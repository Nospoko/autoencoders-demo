import os
import time

import hydra
import torch
import psutil
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

import wandb
from models.VQVAE import VQVAE
from models.model import Autoencoder
from utils.data_loader import prepare_dataset
from models.ECG_autoencoder import ECG_autoencoder
from pipeline import autoencoder as autoencoder_pipeline
from models.variational_autoencoder import VariationalAutoencoder
from utils.visualizations import draw_interpolation_grid, save_img_tensors_as_grid, visualize_ecg_reconstruction
from utils.train_utils import (
    test_epoch,
    train_epoch,
    test_epoch_ecg,
    train_epoch_ecg,
    test_epoch_vqvae,
    train_epoch_vqvae,
    prepare_loss_function,
)


def initialize_model(cfg: DictConfig, input_size):
    if cfg.model.type == "AE":
        model = Autoencoder(cfg, input_size)
    elif cfg.model.type == "ECG_AE":
        model = ECG_autoencoder(cfg, input_size)
    elif cfg.model.type == "VAE":
        model = VariationalAutoencoder(cfg, input_size)
    elif cfg.model.type == "VQ-VAE":
        model = VQVAE(cfg.vqvae)
    else:
        raise NotImplementedError("Model type not implemented")
    return model.to(cfg.system.device)


def train_model(
    cfg: DictConfig,
    model: Autoencoder,
    loss_function: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
):
    torch.manual_seed(cfg.system.seed)

    if cfg.model.type == "ECG_AE":
        train_ecg_autoencoder(cfg, model, loss_function, train_loader, test_loader)
    elif cfg.model.type == "VQ-VAE":
        train_vqvae(cfg, model, loss_function, train_loader, test_loader)
    else:
        train_autoencoder(cfg, model, loss_function, train_loader, test_loader)


def train_vqvae(cfg: DictConfig, model: Autoencoder, loss_function: torch.nn.Module, train_loader, test_loader):
    total_start_time = time.time()
    best_train_loss = float("inf")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    for epoch in range(cfg.train.epochs):
        start_time = time.time()

        avg_loss = train_epoch_vqvae(
            model,
            train_loader,
            optimizer,
            cfg.system.device,
            cfg.train.log_interval,
            epoch,
            loss_function,
            cfg.vqvae.beta,
            cfg.vqvae.use_ema,
            best_train_loss,
        )
        avg_test_loss = test_epoch_vqvae(model, test_loader, cfg.system.device, loss_function, cfg.vqvae.beta)

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

        end_time = time.time()
        epoch_duration = end_time - start_time

        wandb.log(
            {
                "train/loss": avg_loss,
                "test/loss": avg_test_loss,
                "epoch_duration": epoch_duration,
            }
        )
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    wandb.log({"total_training_time": total_training_time})


def train_ecg_autoencoder(cfg: DictConfig, model: Autoencoder, loss_function: torch.nn.Module, train_loader, test_loader):
    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    for epoch in range(1, cfg.train.epochs + 1):
        start_time = time.time()

        avg_loss = train_epoch_ecg(
            model,
            train_loader,
            optimizer,
            model.device,
            cfg.train.log_interval,
            epoch,
            loss_function,
        )
        avg_test_loss = test_epoch_ecg(model, test_loader, model.device, loss_function)

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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

        first_layer = model.encoder.conv[0]

        # Getting the weights of the first convolutional layer
        sample_weights = first_layer.weight.detach().cpu().numpy().flatten()

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

    wandb.log({"total_training_time": total_training_time})


def train_autoencoder(
    cfg: DictConfig,
    model: Autoencoder,
    loss_function: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    for epoch in range(cfg.train.epochs):
        start_time = time.time()
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=cfg.system.device,
            log_interval=cfg.train.log_interval,
            epoch=epoch,
            loss_function=loss_function,
        )
        test_loss = test_epoch(
            model=model,
            test_loader=test_loader,
            device=cfg.system.device,
            loss_function=loss_function,
        )

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}_embSize_{}.pt".format(
            cfg.logger.checkpoint_path, cfg.model.type, cfg.dataset.name, epoch, cfg.model.embedding_size
        )
        torch.save(checkpoint, checkpoint_path)

        end_time = time.time()
        epoch_duration = end_time - start_time  # in megabytes

        # Accessing the first convolutional layer of the encoder
        first_layer = model.encoder.conv[0]

        # Getting the weights of the first convolutional layer
        sample_weights = first_layer.weight.detach().cpu().numpy().flatten()

        wandb.log(
            {
                "train/loss": train_loss,
                "test/loss": test_loss,
                "epoch_duration": epoch_duration,
                "sample_weights": wandb.Histogram(sample_weights),
            }
        )


def train_multiple_models(cfg: DictConfig, range_begin: int, range_end: int, range_step: int):
    os.makedirs(cfg.logger.checkpoint_path + "/multiple/", exist_ok=True)

    for embedding_size in range(range_begin, range_end + 1, range_step):  # Loop through embedding sizes from 1 to 32
        print(f"Training for embedding_size: {embedding_size}")

        # Override the embedding_size in the original configuration
        cfg.model.embedding_size = embedding_size

        # Reinitialize everything for the new embedding_size
        # FIXME
        train_loader, test_loader, input_size = prepare_dataset(cfg)
        model = initialize_model(cfg, input_size)

        loss_function = prepare_loss_function(cfg.train)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        # Train for 5 epochs only
        for epoch in range(1, cfg.train.epochs + 1):
            train_epoch(
                model,
                train_loader,
                optimizer,
                cfg.system.device,
                cfg.train.log_interval,
                epoch,
                loss_function,
            )
            test_epoch(
                model,
                test_loader,
                cfg.system.device,
                loss_function,
            )

        # Save only one checkpoint per model, after 5 epochs
        checkpoint = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,  # Save the config used for this run
        }
        checkpoint_path = f"{cfg.logger.checkpoint_path}/multiple/{cfg.model.type}_{cfg.dataset.name}_checkpoint_epoch_5_embSize_{embedding_size}.pt"
        torch.save(checkpoint, checkpoint_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Preparations
    os.makedirs(cfg.logger.results_path, exist_ok=True)
    os.makedirs(cfg.logger.checkpoint_path, exist_ok=True)

    wandb.init(
        project="autoencoders",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    if cfg.model.type == "AE":
        autoencoder_pipeline.main(cfg)

    train_dataset, test_dataset = prepare_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    input_size = train_dataset.input_size
    model = initialize_model(cfg, input_size)

    loss_function = prepare_loss_function(loss_function_name=cfg.train.loss_function)
    train_model(
        cfg=cfg,
        model=model,
        loss_function=loss_function,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    # Validation
    if cfg.model.type == "ECG_AE":
        visualize_ecg_reconstruction(cfg, model, test_loader)
    elif cfg.model.type == "VQ-VAE" or cfg.dataset.name == "CIFAR10":
        save_img_tensors_as_grid(
            model=model,
            train_loader=test_loader,
            nrows=4,
            f=f"{cfg.logger.results_path}side_by_side_comparison",
            side_by_side=True,
            type=cfg.model.type,
        )
    else:
        draw_interpolation_grid(cfg, model, test_loader)


if __name__ == "__main__":
    main()
