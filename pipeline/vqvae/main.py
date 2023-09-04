import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from models.VQVAE import VQVAE
from utils.data_loader import prepare_dataset
from utils.train_utils import prepare_loss_function


def train(cfg: DictConfig) -> nn.Module:
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    device = cfg.system.device

    model = VQVAE(cfg.vqvae).to(device)
    model.train()

    loss_function = prepare_loss_function(loss_function_name=cfg.train.loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    step = 0
    best_test_loss = float("inf")
    epoch_progress = tqdm(range(cfg.train.epochs))
    for epoch in epoch_progress:
        train_loss = []
        total_recon_error = 0
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for it, batch in train_progress:
            optimizer.zero_grad()
            images = batch["image"].to(device) / 255

            out = model(images)
            recon_error = loss_function(out["x_recon"], images)
            total_recon_error += recon_error.item()

            # cfg.train.beta is the configurable beta term
            loss = recon_error + cfg.vqvae.beta * out["commitment_loss"]

            if not cfg.vqvae.use_ema:
                loss += out["dictionary_loss"]

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if step % cfg.train.log_interval == 0:
                train_progress.set_postfix(loss=loss.item())
                wandb.log({"train/loss": loss.item()}, step=step)

            step += 1

        # Test epoch
        model.eval()
        test_loss = []

        test_progress = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        with torch.no_grad():
            for it, batch in test_progress:
                images = batch["image"].to(device) / 255
                out = model(images)

                recon_error = loss_function(out["x_recon"], images)
                loss = recon_error + cfg.vqvae.beta * out["commitment_loss"]
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

    return model
