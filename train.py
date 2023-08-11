import os
import time
import argparse

import torch
import psutil
import imageio
import numpy as np
from torchvision.utils import save_image

from utils import get_interpolations
from models.autoencoder import Autoencoder
from train_utils import test_epoch, train_epoch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main function to call training for different AutoEncoders")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="input batch size for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="enables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--embedding-size", type=int, default=32, metavar="N", help="how many batches to wait before logging training status"
    )
    parser.add_argument("--results_path", type=str, default="results/", metavar="N", help="Where to store images")
    parser.add_argument("--model", type=str, choices=["AE"], default="AE", metavar="N", help="Which architecture to use")
    parser.add_argument("--dataset", type=str, default="MNIST", metavar="N", help="Which dataset to use")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    return args


def train_autoencoder(args, autoenc):
    # Pseudocode for wandb initialization, since I'm not sure how it works yet
    # wandb.init(project='autoencoder', config=args)

    total_start_time = time.time()
    process = psutil.Process()

    optimizer = torch.optim.Adam(autoenc.model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        train_epoch(autoenc, autoenc.train_loader, optimizer, autoenc.device, autoenc.args.log_interval, epoch)
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


def draw_interpolation_grid(args, autoenc):
    with torch.no_grad():
        for batch in autoenc.test_loader:
            images = batch["image"].to(autoenc.device) / 255.0
            break  # Get the first batch for visualization purposes

        images_per_row = 16
        interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

        sample = torch.randn(64, args.embedding_size).to(autoenc.device)
        sample = autoenc.model.decode(sample).cpu()

        # Save the images and interpolations
        save_image(sample.view(64, 1, 28, 28), "{}/sample_{}_{}.png".format(args.results_path, args.model, args.dataset))
        save_image(
            interpolations.view(-1, 1, 28, 28),
            "{}/interpolations_{}_{}.png".format(args.results_path, args.model, args.dataset),
            nrow=images_per_row,
        )

        interpolations = interpolations.cpu()
        interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
        interpolations *= 256
        imageio.mimsave(
            "{}/animation_{}_{}.gif".format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8)
        )


if __name__ == "__main__":
    args = parse_arguments()  # will eventually be changing that to Hydra

    os.makedirs(args.results_path, exist_ok=True)

    # this is a little obsolete for now, will leave it for possible future use
    ae = Autoencoder(args)
    architectures = {"AE": ae}
    autoenc = architectures[args.model]

    train_autoencoder(args, autoenc)
