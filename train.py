import os
import time

import hydra
import torch
import psutil
from omegaconf import OmegaConf, DictConfig

import wandb
from data_loader import get_data_loaders
from models.autoencoder import Autoencoder
from models.ECG_autoencoder import ECG_autoencoder
from visualizations import draw_interpolation_grid, visualize_ecg_reconstruction
from train_utils import test_epoch, train_epoch, test_epoch_ecg, train_epoch_ecg, prepare_loss_function


def initialize_model(cfg: DictConfig, input_size):
    if cfg.model.type == "AE":
        model = Autoencoder(cfg, input_size)
    elif cfg.model.type == "ECG_AE":
        model = ECG_autoencoder(cfg, input_size)
    else:
        raise NotImplementedError("Model type not implemented")
    return model.to(model.device)


def train(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module, train_loader, test_loader):
    os.makedirs(cfg.logger.results_path, exist_ok=True)
    os.makedirs(cfg.logger.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.system.seed)

    if cfg.model.type == "ECG_AE":
        train_ecg_autoencoder(cfg, autoencoder, loss_function, train_loader, test_loader)
    else:
        train_autoencoder(cfg, autoencoder, loss_function, train_loader, test_loader)


def train_ecg_autoencoder(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module, train_loader, test_loader):
    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg.train.lr)
    for epoch in range(1, cfg.train.epochs + 1):
        start_time = time.time()

        avg_loss = train_epoch_ecg(
            autoencoder,
            train_loader,
            optimizer,
            autoencoder.device,
            cfg.train.log_interval,
            epoch,
            loss_function,
        )
        avg_test_loss = test_epoch_ecg(autoencoder, test_loader, autoencoder.device, loss_function)

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": autoencoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

        memory_usage = process.memory_info().rss / (1024 * 1024)  # in megabytes
        end_time = time.time()
        epoch_duration = end_time - start_time  # in megabytes
        # Get CPU utilization
        cpu_utilization = psutil.cpu_percent()

        if torch.cuda.is_available():
            gpu_util = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            gpu_util *= 100  # Convert to percentage
        else:
            gpu_util = 0

        first_layer = autoencoder.encoder.layers[0]

        # Getting the weights of the first convolutional layer
        sample_weights = first_layer.weight.detach().cpu().numpy().flatten()

        if cfg.logger.enable_wandb:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "test/loss": avg_test_loss,
                    "memory_usage": memory_usage,
                    "epoch_duration": epoch_duration,
                    "cpu_utilization": cpu_utilization,
                    "gpu_utilization": gpu_util,
                    "sample_weights": wandb.Histogram(sample_weights),
                }
            )

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    if cfg.logger.enable_wandb:
        wandb.log({"total_training_time": total_training_time})


def train_autoencoder(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module, train_loader, test_loader):
    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg.train.lr)
    for epoch in range(1, cfg.train.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(
            autoencoder=autoencoder,
            train_loader=train_loader,
            optimizer=optimizer,
            device=autoencoder.device,
            log_interval=cfg.train.log_interval,
            epoch=epoch,
            loss_function=loss_function,
        )
        test_loss = test_epoch(
            autoencoder=autoencoder,
            test_loader=test_loader,
            device=autoencoder.device,
            loss_function=loss_function,
        )

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": autoencoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

        memory_usage = process.memory_info().rss / (1024 * 1024)  # in megabytes
        end_time = time.time()
        epoch_duration = end_time - start_time  # in megabytes
        # Get CPU utilization
        cpu_utilization = psutil.cpu_percent()

        # Get GPU utilization (assuming you're using NVIDIA GPUs and torch.cuda is available)
        if torch.cuda.is_available():
            gpu_util = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            gpu_util *= 100  # Convert to percentage
        else:
            gpu_util = 0

        # Accessing the first convolutional layer of the encoder
        first_layer = autoencoder.encoder.conv[0]

        # Getting the weights of the first convolutional layer
        sample_weights = first_layer.weight.detach().cpu().numpy().flatten()

        if cfg.logger.enable_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "test/loss": test_loss,
                    "memory_usage": memory_usage,
                    "epoch_duration": epoch_duration,
                    "cpu_utilization": cpu_utilization,
                    "gpu_utilization": gpu_util,
                    "sample_weights": wandb.Histogram(sample_weights),
                }
            )

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    if cfg.logger.enable_wandb:
        wandb.log({"total_training_time": total_training_time})


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        name = f"{cfg.dataset.name}_{cfg.run_date}"
        wandb.init(
            project="autoencoder",
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    train_loader, test_loader, input_size = get_data_loaders(cfg)
    autoencoder = initialize_model(cfg, input_size)

    loss_function = prepare_loss_function(cfg.train)
    train(cfg, autoencoder, loss_function, train_loader, test_loader)

    if cfg.model.type == "ECG_AE":
        visualize_ecg_reconstruction(cfg, autoencoder, test_loader)
    else:
        draw_interpolation_grid(cfg, autoencoder, test_loader)


if __name__ == "__main__":
    main()
