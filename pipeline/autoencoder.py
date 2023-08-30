from typing import Callable

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from models.autoencoder import Autoencoder
from utils.data_loader import prepare_dataset
from utils.train_utils import prepare_loss_function


def main(cfg: DictConfig):
    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    input_size = train_dataset.input_size
    model = Autoencoder(embedding_size=cfg.model.embedding_size, input_size=input_size)
    model.to(cfg.system.device)
    loss_fn = prepare_loss_function(loss_function_name=cfg.train.loss_function)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    step = 0
    best_test_loss = float("inf")
    epoch_progrss = tqdm(range(cfg.train.epochs))
    for epoch in epoch_progrss:
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
        epoch_progrss.set_postfix(train_loss=train_loss, test_loss=test_loss)

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


def forward_step(
    model: nn.Module,
    batch: dict,
    loss_fn: Callable,
    device: str,
):
    data = batch["image"].to(device) / 255.0

    # Hmm
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    recon_batch = model(data)
    loss = loss_fn(recon_batch, data)

    return loss
