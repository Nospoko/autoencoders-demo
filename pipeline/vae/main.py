import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import wandb
from pipeline.vae import evals as vae_evals
from utils.data_loader import prepare_dataset
from models.variational_autoencoder import VariationalAutoencoder


def train(cfg: DictConfig) -> nn.Module:
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    device = cfg.system.device
    input_size = train_dataset.input_size
    model = VariationalAutoencoder(
        input_size=input_size,
        encoder_output_size=cfg.model.output_size,
        embedding_size=cfg.model.embedding_size,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    step = 0
    best_test_loss = float("inf")
    epoch_progress = tqdm(range(cfg.train.epochs))
    for epoch in epoch_progress:
        # Train epoch
        model.train()

        train_loss = []
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for it, batch in train_progress:
            optimizer.zero_grad()
            losses = forward_step(
                model=model,
                batch=batch,
                device=cfg.system.device,
            )
            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            if step % cfg.train.log_interval == 0:
                train_progress.set_postfix(loss=loss.item())
                metrics = {f"train/{key}": value.item() for key, value in losses.items()}
                wandb.log(metrics, step=step)

            train_loss.append(loss.item())

            step += 1

        # Test epoch
        model.eval()
        test_loss = []

        with torch.no_grad():
            for it, batch in enumerate(test_loader):
                losses = forward_step(
                    model=model,
                    batch=batch,
                    device=cfg.system.device,
                )
                loss = losses["loss"]
                test_loss.append(loss.item())

        test_loss = np.mean(test_loss)
        train_loss = np.mean(train_loss)
        wandb.log({"train/loss_epoch": train_loss, "test/loss_epoch": test_loss}, step=step)
        epoch_progress.set_postfix(train_loss=train_loss, test_loss=test_loss)

        if test_loss < best_test_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }
            checkpoint_path = "{}/{}.pt".format(cfg.checkpoint_path, cfg.run_name)
            torch.save(checkpoint, checkpoint_path)
            best_test_loss = test_loss

    print("Best checkpoint:", best_test_loss)
    print("Checkpoint path:", checkpoint_path)

    return model


def forward_step(
    model: nn.Module,
    batch: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    data = batch["image"].to(device) / 255.0

    # Hmm
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    recon_batch, mu, logvar = model(data)
    recon_loss = F.binary_cross_entropy(recon_batch, data, reduction="mean")
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + KLD

    losses = {
        "loss": loss,
        "KLD": KLD,
        "recon": recon_loss,
    }

    return losses


def main(cfg: DictConfig):
    model = train(cfg)
    train_dataset, test_dataset = prepare_dataset(cfg)

    device = cfg.system.device
    images = test_dataset[:20]["image"].to(device) / 255

    # Demo usage
    grid = vae_evals.make_interpolation_grid(model, images, n_interps=12)
    savepath = "tmp/tmp.png"
    save_image(grid, savepath)
    print("Saved an image!", savepath)
