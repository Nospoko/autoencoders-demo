import os
import time

import hydra
import torch
import psutil
import imageio
import numpy as np
from omegaconf import DictConfig
from torchvision.utils import save_image

from utils import get_interpolations
from models.autoencoder import Autoencoder
from train_utils import test_epoch, train_epoch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    os.makedirs(cfg.logger.results_path, exist_ok=True)

    torch.manual_seed(cfg.system.seed)

    ae = Autoencoder(cfg)
    architectures = {"AE": ae}
    autoenc = architectures[cfg.model.type]

    train_autoencoder(cfg, autoenc)
    draw_interpolation_grid(cfg, autoenc)


def train_autoencoder(args: DictConfig, autoenc):
    # Pseudocode for wandb initialization, since I'm not sure how it works yet
    # wandb.init(project='autoencoder', config=args)

    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoenc.model.parameters(), lr=1e-3)
    for epoch in range(1, args.hyperparameters.epochs + 1):
        train_epoch(autoenc, autoenc.train_loader, optimizer, autoenc.device, args.hyperparameters.log_interval, epoch)
        test_epoch(autoenc, autoenc.test_loader, autoenc.device)
        # test_loss = test_epoch(autoenc, autoenc.test_loader, autoenc.device)

        memory_usage = process.memory_info().rss / (1024 * 1024)  # in megabytes

        # Pseudocode for wandb logging
        # wandb.log({
        #     "epoch": epoch,
        #     "test_loss": test_loss,
        #     "memory_usage": memory_usage
        # })

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    # placeholder for wandb logging
    print("Total training time: {}".format(total_training_time) + " seconds")
    print("Memory usage: {}".format(memory_usage) + " MB")
    # Pseudocode for wandb logging
    # wandb.log({"total_training_time": total_training_time})


@torch.no_grad()
def draw_interpolation_grid(args, autoenc):
    for batch in autoenc.test_loader:
        images = batch["image"].to(autoenc.device) / 255.0
        break  # Get the first batch for visualization purposes

    images_per_row = 16
    interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

    sample = torch.randn(64, args.model.embedding_size).to(autoenc.device)
    sample = autoenc.model.decode(sample).cpu()

    # Save the images and interpolations
    save_image(
        sample.view(64, 1, 28, 28), "{}/sample_{}_{}.png".format(args.logger.results_path, args.model.type, args.dataset.name)
    )
    save_image(
        interpolations.view(-1, 1, 28, 28),
        "{}/interpolations_{}_{}.png".format(args.logger.results_path, args.model.type, args.dataset.name),
        nrow=images_per_row,
    )

    interpolations = interpolations.cpu()
    interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
    interpolations *= 256
    imageio.mimsave(
        "{}/animation_{}_{}.gif".format(args.logger.results_path, args.model.type, args.dataset.name),
        interpolations.astype(np.uint8),
    )


if __name__ == "__main__":
    main()
