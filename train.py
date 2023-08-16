import os
import time

import hydra
import torch
import psutil
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torchvision.utils import save_image

import wandb
from utils import get_interpolations
from data_loader import get_data_loaders
from models.autoencoder import Autoencoder
from models.ECG_autoencoder import ECG_autoencoder
from train_utils import test_epoch, train_epoch, test_epoch_ecg, train_epoch_ecg, prepare_loss_function


def initialize_model(cfg: DictConfig, train_loader, test_loader, input_size):
    if cfg.model.type == "AE":
        model = Autoencoder(cfg, train_loader=train_loader, test_loader=test_loader, input_size=input_size)
    elif cfg.model.type == "ECG_AE":
        model = ECG_autoencoder(cfg, train_loader=train_loader, test_loader=test_loader, input_size=input_size)
    else:
        raise NotImplementedError("Model type not implemented")
    return model


def train(cfg: DictConfig, autoencoder: Autoencoder, loss_function: torch.nn.Module):
    os.makedirs(cfg.logger.results_path, exist_ok=True)
    os.makedirs(cfg.logger.checkpoint_path, exist_ok=True)

    torch.manual_seed(cfg.system.seed)

    if cfg.model.type == "ECG_AE":
        train_ecg_autoencoder(cfg, autoencoder, loss_function)
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


@torch.no_grad()
def draw_interpolation_grid(cfg, autoenc):
    for batch in autoenc.test_loader:
        images = batch["image"].to(autoenc.device) / 255.0
        break  # Get the first batch for visualization purposes

    images_per_row = 16
    interpolations = get_interpolations(cfg, autoenc.model, autoenc.device, images, images_per_row)

    img_dim = images.shape[-2]
    channels = 3 if img_dim == 32 else 1
    sample = torch.randn(64, cfg.model.embedding_size).to(autoenc.device)
    sample = autoenc.model.decode(sample).cpu()

    # Adjust the reshape based on channels and img_dim
    save_image(
        sample.view(64, channels, img_dim, img_dim),
        "{}/sample_{}_{}.png".format(cfg.logger.results_path, cfg.model.type, cfg.dataset.name),
    )
    save_image(
        interpolations.view(-1, channels, img_dim, img_dim),
        "{}/interpolations_{}_{}.png".format(cfg.logger.results_path, cfg.model.type, cfg.dataset.name),
        nrow=images_per_row,
    )

    interpolations = interpolations.cpu()
    interpolations = np.reshape(interpolations.data.numpy(), (-1, img_dim, img_dim, channels))
    if channels == 1:  # Convert grayscale to RGB for gif
        interpolations = np.repeat(interpolations, 3, axis=-1)
    interpolations *= 256
    imageio.mimsave(
        "{}/animation_{}_{}.gif".format(cfg.logger.results_path, cfg.model.type, cfg.dataset.name),
        interpolations.astype(np.uint8),
    )


@torch.no_grad()
def visualize_ecg_reconstruction(cfg, autoenc, test_loader):
    # Fetch a batch of ECG signals
    record = next(iter(test_loader))
    signal = record["signal"].to(autoenc.device)

    # Get the reconstructed signal from the autoencoder
    reconstructed_signal_batch = autoenc.model(signal)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))  # Adjusting to 3 rows, 2 columns

    # Just taking first 3 signals from the batch for visualization
    for idx in range(3):
        original_signal = signal[idx].cpu().numpy()
        reconstructed_signal = reconstructed_signal_batch[idx].squeeze().cpu().numpy()

        # Plot original and reconstructed signals on the left axis of the subplot
        sns.lineplot(
            x=range(len(original_signal[0])),
            y=original_signal[0],
            ax=axes[idx, 0],
            label="Original Signal Channel 1",
            color="blue",
        )
        sns.lineplot(
            x=range(len(original_signal[1])),
            y=original_signal[1],
            ax=axes[idx, 0],
            label="Original Signal Channel 2",
            color="green",
        )
        sns.lineplot(
            x=range(len(reconstructed_signal[0])),
            y=reconstructed_signal[0],
            ax=axes[idx, 0],
            label="Reconstructed Signal Channel 1",
            color="red",
            linestyle="--",
        )
        sns.lineplot(
            x=range(len(reconstructed_signal[1])),
            y=reconstructed_signal[1],
            ax=axes[idx, 0],
            label="Reconstructed Signal Channel 2",
            color="orange",
            linestyle="--",
        )
        axes[idx, 0].set_title(f"Original vs Reconstructed Signal for Sample {idx+1}")

        # Right axis just for original signal
        sns.lineplot(x=range(len(original_signal[0])), y=original_signal[0], ax=axes[idx, 1], color="blue")
        sns.lineplot(x=range(len(original_signal[1])), y=original_signal[1], ax=axes[idx, 1], color="green")
        axes[idx, 1].set_title(f"Original Signal for Sample {idx+1}")

    # axes[0, 0].legend() <- trying to make it look cleaner
    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        wandb.init(project="autoencoder", config=cfg)

    train_loader, test_loader, input_size = get_data_loaders(cfg)
    model = initialize_model(cfg, train_loader, test_loader, input_size)

    loss_function = prepare_loss_function(cfg.train)
    train(cfg, model, loss_function)


if __name__ == "__main__":
    main()
