import torch
import imageio
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from utils.utils import get_interpolations


@torch.no_grad()
def draw_interpolation_grid(cfg, autoencoder, test_loader):
    for batch in test_loader:
        images = batch["image"].to(autoencoder.device) / 255.0
        break  # Get the first batch for visualization purposes

    images_per_row = 16
    interpolations = get_interpolations(cfg, autoencoder, autoencoder.device, images, images_per_row)

    img_dim = images.shape[-2]
    channels = 3 if img_dim == 32 else 1
    sample = torch.randn(64, cfg.model.embedding_size).to(autoencoder.device)
    sample = autoencoder.decode(sample).cpu()

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
def visualize_ecg_reconstruction(cfg, autoencoder, test_loader):
    to_plot = []

    # Getting batches from the test_loader until we have enough signals
    for batch in test_loader:
        signals = batch["signal"].to(autoencoder.device)
        higher_than = 0.8
        heartbeat_signals = [signal for signal in signals if signal.max() > higher_than]
        to_plot.extend(heartbeat_signals)
        if len(to_plot) >= 6:  # Exit loop when we have at least 6 signals
            break

    # If there are not enough signals, give a message and exit
    if len(to_plot) < 4:
        print("Not enough signals with max value > 0.8 found.")
        return

    # Convert list to tensor and pass through autoencoder
    to_plot_tensor = torch.stack(to_plot[:4])
    reconstructions = autoencoder.model(to_plot_tensor)

    # Convert to CPU for visualization
    to_plot_tensor = to_plot_tensor.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Number of samples to visualize (can be less than 4 if not enough signals found)
    num_samples = min(4, len(to_plot))

    plt.figure(figsize=(20, 6 * num_samples))

    for i in range(num_samples):
        # Channel 0
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.plot(to_plot_tensor[i, 0, :], label="Original Channel 0", color="blue")
        plt.plot(reconstructions[i, 0, :], label="Reconstructed Channel 0", color="red", linestyle="--")

        # Channel 1
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.plot(to_plot_tensor[i, 1, :], label="Original Channel 1", color="green")
        plt.plot(reconstructions[i, 1, :], label="Reconstructed Channel 1", color="orange", linestyle="--")

    plt.tight_layout(pad=5.0)
    # save the plot
    plt.savefig("{}/reconstructions_{}_{}.png".format(cfg.logger.results_path, cfg.model.type, cfg.dataset.name))
    plt.show()


@torch.no_grad()
def visualize_embedding(cfg, autoencoder, test_loader, num_trio=10):
    """
    Visualize the original image, its embedding, and its reconstruction for each label in the dataset.

    :param cfg: The configuration dict.
    :param autoencoder: The trained autoencoder model.
    :param test_loader: The data loader for the test dataset.
    :param num_trio: The number of trios to visualize.
    """

    autoencoder.eval()  # Set the model to evaluation mode
    autoencoder.to(autoencoder.device)

    found_labels = set()
    label_to_image = {}

    for idx in range(len(test_loader.dataset)):
        image = test_loader.dataset[idx]["image"]
        label = test_loader.dataset[idx]["label"]

        if label not in found_labels:
            label_to_image[label] = image
            found_labels.add(label)

        if len(found_labels) == num_trio:
            break

    fig, axs = plt.subplots(num_trio, 3, figsize=(9, 3 * num_trio))

    for idx, (label, image) in enumerate(label_to_image.items()):
        image = image.float().to(autoencoder.device) / 255.0

        if cfg.model.type == "VAE":
            mu, logvar = autoencoder.encode(image.unsqueeze(0))
            z = autoencoder.reparameterize(mu, logvar)
        else:
            z = autoencoder.encode(image.unsqueeze(0))

        reconstructed_image = autoencoder.decode(z).squeeze().cpu().numpy()

        # Original image
        axs[idx, 0].imshow(image.cpu().squeeze().numpy(), cmap="gray")
        axs[idx, 0].set_title(f"Label {label} - Original")

        # Embedding
        embedding_size = z.size(1)
        side_length = int(np.ceil(np.sqrt(embedding_size)))
        padding_needed = side_length * side_length - embedding_size

        padded_z = torch.nn.functional.pad(z, (0, padding_needed), mode="constant", value=0)

        axs[idx, 1].imshow(padded_z.cpu().squeeze().numpy().reshape(side_length, side_length), cmap="gray")
        axs[idx, 1].set_title(f"Label {label} - Embedding")

        # Reconstructed image
        axs[idx, 2].imshow(reconstructed_image, cmap="gray")
        axs[idx, 2].set_title(f"Label {label} - Reconstructed")

        for j in range(3):
            axs[idx, j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{cfg.logger.results_path}/reconstructions_{cfg.model.type}_{cfg.dataset.name}.png")
    plt.show()
