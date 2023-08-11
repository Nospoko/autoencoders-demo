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


def initialize_model(args: DictConfig):
    if args.model.type == "AE":
        model = Autoencoder(args)
    else:
        raise NotImplementedError("Model type not implemented")
    return model


def train(args: DictConfig, model: torch.nn.Module):
    os.makedirs(args.logger.results_path, exist_ok=True)
    os.makedirs(args.logger.checkpoint_path, exist_ok=True)

    torch.manual_seed(args.system.seed)

    train_autoencoder(args, model)
    draw_interpolation_grid(args, model)


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

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": autoenc.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": args,  # Saving the config used for this training run
        }
        checkpoint_path = "{}/{}_{}_checkpoint_epoch_{}.pt".format(
            args.logger.checkpoint_path, args.model.type, args.dataset.name, epoch
        )
        torch.save(checkpoint, checkpoint_path)

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = initialize_model(cfg)
    train(cfg, model)


if __name__ == "__main__":
    main()
