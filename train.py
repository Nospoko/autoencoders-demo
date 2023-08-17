import os
import time

import hydra
import torch
import psutil
from omegaconf import DictConfig

import wandb
from data_loader import get_data_loaders
from models.autoencoder import Autoencoder
from models.ECG_autoencoder import ECG_autoencoder
from visualizations import draw_interpolation_grid, visualize_ecg_reconstruction
from train_utils import test_epoch, train_epoch, test_epoch_ecg, train_epoch_ecg, prepare_loss_function


def initialize_model(cfg: DictConfig, train_loader, test_loader, input_size):
    if cfg.model.type == "AE":
        model = Autoencoder(cfg, train_loader=train_loader, test_loader=test_loader, input_size=input_size)
    elif cfg.model.type == "ECG_AE":
        model = ECG_autoencoder(cfg, train_loader=train_loader, test_loader=test_loader, input_size=input_size)
    else:
        raise NotImplementedError("Model type not implemented")
    return model


def train(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module, test_loader):
    os.makedirs(cfg.logger.results_path, exist_ok=True)
    os.makedirs(cfg.logger.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.system.seed)

    if cfg.model.type == "ECG_AE":
        train_ecg_autoencoder(cfg, autoencoder, loss_function)
        visualize_ecg_reconstruction(cfg, autoencoder, test_loader)
    else:
        train_autoencoder(cfg, autoencoder, loss_function)
        draw_interpolation_grid(cfg, autoencoder)


def train_ecg_autoencoder(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module):
    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoencoder.model.parameters(), lr=cfg.train.lr)
    for epoch in range(1, cfg.train.epochs + 1):
        avg_loss = train_epoch_ecg(
            autoencoder, autoencoder.train_loader, optimizer, autoencoder.device, cfg.train.log_interval, epoch, loss_function
        )
        avg_test_loss = test_epoch_ecg(autoencoder, autoencoder.test_loader, autoencoder.device, loss_function)

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": autoencoder.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

        memory_usage = process.memory_info().rss / (1024 * 1024)  # in megabytes

        if cfg.logger.enable_wandb:
            wandb.log({"epoch": epoch, "test_loss": avg_test_loss, "train_loss": avg_loss, "memory_usage": memory_usage})

    total_time = time.time() - total_start_time
    if cfg.logger.enable_wandb:
        wandb.log({"total_time": total_time})


def train_autoencoder(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module):
    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoencoder.model.parameters(), lr=cfg.train.lr)
    for epoch in range(1, cfg.train.epochs + 1):
        train_epoch(
            autoencoder, autoencoder.train_loader, optimizer, autoencoder.device, cfg.train.log_interval, epoch, loss_function
        )
        # test_epoch(autoenc, autoenc.test_loader, autoenc.device)
        test_loss = test_epoch(autoencoder, autoencoder.test_loader, autoencoder.device, loss_function)

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": autoencoder.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

        memory_usage = process.memory_info().rss / (1024 * 1024)  # in megabytes

        # Pseudocode for wandb logging
        if cfg.logger.enable_wandb:
            wandb.log({"epoch": epoch, "test_loss": test_loss, "memory_usage": memory_usage})

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    if cfg.logger.enable_wandb:
        wandb.log({"total_training_time": total_training_time})


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        wandb.init(project="autoencoder", config=cfg)

    train_loader, test_loader, input_size = get_data_loaders(cfg)
    model = initialize_model(cfg, train_loader, test_loader, input_size)

    loss_function = prepare_loss_function(cfg.train)
    train(cfg, model, loss_function, test_loader)


if __name__ == "__main__":
    main()
