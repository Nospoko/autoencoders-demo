from typing import Callable

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import wandb
from pipeline.vae import evals as vae_evals
from utils.data_loader import prepare_dataset
from utils.train_utils import prepare_loss_function
from models.variational_autoencoder import VariationalAutoencoder


def train(cfg: DictConfig) -> nn.Module:
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    device = cfg.system.device
    input_size = train_dataset.input_size
    model = VariationalAutoencoder(
        input_size=input_size,
        output_size=cfg.model.output_size,
        embedding_size=cfg.model.embedding_size,
    )
    model = model.to(device)

    loss_fn = prepare_loss_function(loss_function_name=cfg.train.loss_function)
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
            loss = forward_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                device=cfg.system.device,
            )
            loss.backward()
            optimizer.step()

            if step % cfg.train.log_interval == 0:
                train_progress.set_postfix(loss=loss.item())
                wandb.log({"train/loss": loss.item()}, step=step)

            train_loss.append(loss.item())

            step += 1

        # Test epoch
        model.eval()
        test_loss = []

        with torch.no_grad():
            for it, batch in enumerate(test_loader):
                loss = forward_step(
                    model=model,
                    batch=batch,
                    loss_fn=loss_fn,
                    device=cfg.system.device,
                )
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
    loss_fn: Callable,
    device: str,
) -> torch.Tensor:
    data = batch["image"].to(device) / 255.0

    # Hmm
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    recon_batch, mu, logvar = model(data)
    loss = loss_fn(recon_batch, data, mu, logvar)

    return loss


def main(cfg: DictConfig):
    model = train(cfg)
    train_dataset, test_dataset = prepare_dataset(cfg)

    device = cfg.system.device
    images = test_dataset[:20]["image"].to(device) / 255

    # Demo usage
    grid = vae_evals.make_interpolation_grid(model, images, n_interps=12)
    save_image(grid, "tmp/tmp.png")
