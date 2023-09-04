import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from utils.data_loader import prepare_dataset
from utils.train_utils import prepare_loss_function
from models.autoencoder_ecg import VariationalAutoencoderECG


def train(cfg: DictConfig) -> nn.Module:
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    device = cfg.system.device
    input_size = train_dataset.input_size
    model = VariationalAutoencoderECG(
        encoder_output_size=cfg.model.encoder_output_size,
        embedding_size=cfg.model.embedding_size,
        input_size=input_size,
    )
    model = model.to(device)

    loss_fn = prepare_loss_function(loss_function_name=cfg.train.loss_function)
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

            recon_batch, mu, logvar = model(data)
            loss = loss_fn(recon_batch, data, mu, logvar)
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

        with torch.no_grad():
            test_progress = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
            for it, batch in test_progress:
                data = batch["signal"].to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_fn(recon_batch, data, mu, logvar)
                test_loss.append(loss.item())

        # Epoch summary
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


def main(cfg: DictConfig):
    _ = train(cfg)
