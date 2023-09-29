import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from utils.data_loader import prepare_dataset
from pipeline.vae_ecg import evals as vae_ecg_evals
from models.autoencoder_ecg import VariationalAutoencoderECG


def train(cfg: DictConfig) -> nn.Module:
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8)

    device = cfg.system.device

    # Fixed samples for visualization
    n_samples = 5
    idxs = np.random.randint(len(test_dataset), size=n_samples)
    fixed_validation_signals = test_dataset[idxs]["signal"].to(device)

    input_size = train_dataset.input_size
    model = VariationalAutoencoderECG(
        encoder_output_size=cfg.model.encoder_output_size,
        embedding_size=cfg.model.embedding_size,
        input_size=input_size,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    step = 0
    best_test_loss = float("inf")
    epoch_progress = tqdm(range(cfg.train.epochs))
    for epoch in epoch_progress:
        model.train()
        train_loss = []

        # Train epoch
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, batch in train_progress:
            data = batch["signal"].to(device)
            optimizer.zero_grad()

            losses = forward_step(
                model=model,
                data=data,
                kl_weight=cfg.train.kl_weight,
                recon_weight=cfg.train.recon_weight,
            )
            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if step % cfg.train.log_interval == 0:
                train_progress.set_postfix(loss=loss.item())
                metrics = {f"train/{key}": value.item() for key, value in losses.items()}
                wandb.log(metrics, step=step)

            step += 1

        # Test epoch
        model.eval()
        test_loss = []

        with torch.no_grad():
            test_progress = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
            for it, batch in test_progress:
                data = batch["signal"].to(device)
                losses = forward_step(
                    model=model,
                    data=data,
                    kl_weight=cfg.train.kl_weight,
                    recon_weight=cfg.train.recon_weight,
                )
                loss = losses["loss"]
                test_loss.append(loss.item())

            # Review reconstructions
            reconstructions, mu, logvar = model(fixed_validation_signals)
            fig = vae_ecg_evals.draw_ecg_reconstructions(fixed_validation_signals, reconstructions)
            wandb.log({"test/reconstruction": wandb.Image(fig)}, step=step)

        # Epoch summary
        test_loss = np.mean(test_loss)
        train_loss = np.mean(train_loss)
        wandb.log({"train/loss_epoch": train_loss, "test/loss_epoch": test_loss}, step=step)
        epoch_progress.set_postfix(train_loss=train_loss, test_loss=test_loss)

        if test_loss < best_test_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_object(cfg),
                "test_loss": test_loss,
            }
            checkpoint_path = "{}/{}.pt".format(cfg.checkpoint_path, cfg.run_name)
            torch.save(checkpoint, checkpoint_path)
            best_test_loss = test_loss

    return model


def forward_step(model: nn.Module, data: torch.Tensor, kl_weight: float, recon_weight: float) -> dict[str, torch.Tensor]:
    recon_batch, mu, logvar = model(data)

    recon_loss = F.mse_loss(recon_batch, data, reduction="mean")
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Balance between reconstruction loss and KLD is a hyperparameter
    # See: https://github.com/Nospoko/autoencoders-demo/pull/7
    loss = recon_weight * recon_loss + kl_weight * KLD

    losses = {
        "loss": loss,
        "KLD": KLD,
        "recon": recon_loss,
    }

    return losses


def main(cfg: DictConfig):
    model = train(cfg)
    model.eval()

    # Data prep
    _, test_dataset = prepare_dataset(cfg)
    device = cfg.system.device
    n_samples = 16
    idxs = np.random.randint(len(test_dataset), size=n_samples)
    signals = test_dataset[idxs]["signal"].to(device)

    # Process
    reconstructions, mu, logvar = model(signals)

    # Review reconstructions
    vae_ecg_evals.draw_ecg_reconstructions(signals, reconstructions)
    savepath = "tmp/vae-ecg-reconstruction.png"
    plt.savefig(savepath)
    print("Saved an image!", savepath)

    # Review embedding based interpolations
    vae_ecg_evals.draw_interpolation_tower(model, signals, 16)
    savepath = "tmp/vae-ecg-interpolation.png"
    plt.savefig(savepath)
