import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from utils.utils import get_interpolations


@torch.no_grad()
def draw_interpolation_grid(cfg, autoencoder, test_loader):
    for batch in test_loader:
        if cfg.dataset.name == "MNIST" or cfg.dataset.name == "FashionMNIST" or cfg.dataset.name == "AmbiguousMNIST":
            images = batch["image"].to(autoencoder.device) / 255.0
        elif cfg.dataset.name == "CIFAR10":
            images = batch["image"].to(autoencoder.device)
        break  # Get the first batch for visualization purposes

    images_per_row = 16
    interpolations = get_interpolations(cfg, autoencoder, autoencoder.device, images, images_per_row)
    interpolations = (interpolations - interpolations.min()) / (interpolations.max() - interpolations.min())

    print("Max:", interpolations.max())
    print("Min:", interpolations.min())

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
    print("Interpolation grid saved.")


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
def visualize_embedding(cfg, autoencoder, test_loader, num_trio=10, rgb=False):
    """
    Visualize the original image, its embedding, and its reconstruction for each label in the dataset.

    :param cfg: The configuration dict.
    :param autoencoder: The trained autoencoder model.
    :param test_loader: The data loader for the test dataset.
    :param num_trio: The number of trios to visualize.
    :param display_2d: A flag to display embedding in 2D or 1D.
    :param rgb: Whether the images are RGB.

    """

    found_labels = set()
    label_to_image = {}

    for idx in range(len(test_loader.dataset)):
        image = test_loader.dataset["image"][idx]
        label = test_loader.dataset["label"][idx]

        if label not in found_labels:
            label_to_image[label] = image
            found_labels.add(label)

        if len(found_labels) == num_trio:
            break

    fig, axs = plt.subplots(num_trio, 3, figsize=(9, 3 * num_trio))

    for idx, (label, image) in enumerate(label_to_image.items()):
        image = image.float() / 255.0

        if rgb:
            image = image.permute(1, 2, 0)  # CxHxW -> HxWxC if RGB
            cmap = None  # Default colormap for RGB images
        else:
            cmap = "gray"

        # Create an embedding for the image
        autoencoder.eval()  # Set the model to evaluation mode
        embedding = autoencoder.encode(image.unsqueeze(0).to(autoencoder.device))

        # Original image
        axs[idx, 0].imshow(image.cpu().numpy(), cmap=cmap)
        axs[idx, 0].set_title(f"Label {label} - Original Image")

        # Embedding

        if cfg.model.type == "VAE":
            mu, log_var = embedding
            image_embedding = mu.cpu().numpy().squeeze()
        else:
            image_embedding = embedding.squeeze().detach().cpu().numpy()
        axs[idx, 1].hist(image_embedding, bins=20, range=[-3, 3], color="blue", edgecolor="black")
        axs[idx, 1].set_xlim([-3, 3])

        axs[idx, 1].set_title(f"Label {label} - {'1D Embedding'}")

        # Reconstructed image
        if cfg.model.type == "VAE":
            reconstructed_image, _, _ = autoencoder(image.unsqueeze(0).to(autoencoder.device))
            reconstructed_image = reconstructed_image.squeeze().cpu().detach().numpy()
        else:
            reconstructed_image = autoencoder.decode(embedding).squeeze().cpu().detach().numpy()

        if rgb:
            reconstructed_image = reconstructed_image.transpose(1, 2, 0)  # If it's a numpy array
            # or if it's a tensor
            # reconstructed_image = reconstructed_image.permute(1, 2, 0).cpu().numpy()

        # Clipping if needed, example for float type
        reconstructed_image = np.clip(reconstructed_image, 0, 1)

        axs[idx, 2].imshow(reconstructed_image, cmap=cmap)
        axs[idx, 2].set_title(f"Label {label} - Reconstructed Image")

        for j in range(3):
            if not (j == 1):
                axs[idx, j].set_xticks([])
            axs[idx, j].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{cfg.logger.results_path}/reconstructions_{cfg.model.type}_{cfg.dataset.name}.png")
    plt.show()


@torch.no_grad()
def save_img_tensors_as_grid(model, train_loader, nrows, f, side_by_side=False, reconstructed=False, type="VQ-VAE"):
    if side_by_side:
        reconstructed = False

    for batch in train_loader:
        img_tensors = batch["image"].to(model.device)
        if type == "VAE":
            reconstructed_tensors, _, _ = model(img_tensors)
        else:
            reconstructed_tensors = model(img_tensors)["x_recon"]
        break

    if reconstructed:
        img_tensors = reconstructed_tensors

    img_tensors = img_tensors.permute(0, 2, 3, 1).detach().cpu().numpy()
    if side_by_side:
        reconstructed_tensors = reconstructed_tensors.permute(0, 2, 3, 1).detach().cpu().numpy()

    # Preprocess the image tensors
    img_tensors = np.clip(255 * (img_tensors + 0.5), 0, 255).astype(np.uint8)
    if side_by_side:
        reconstructed_tensors = np.clip(255 * (reconstructed_tensors + 0.5), 0, 255).astype(np.uint8)

    batch_size, img_size, _, _ = img_tensors.shape
    ncols = batch_size // nrows

    if side_by_side:
        img_arr = np.zeros((nrows * img_size, 2 * ncols * img_size, 3), dtype=np.uint8)
    else:
        img_arr = np.zeros((nrows * img_size, ncols * img_size, 3), dtype=np.uint8)

    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols

        row_start = row_idx * img_size
        row_end = row_start + img_size

        if side_by_side:
            col_start = 2 * col_idx * img_size
            col_end = col_start + img_size
            img_arr[row_start:row_end, col_start:col_end] = img_tensors[idx]
            img_arr[row_start:row_end, col_end : col_end + img_size] = reconstructed_tensors[idx]
        else:
            col_start = col_idx * img_size
            col_end = col_start + img_size
            img_arr[row_start:row_end, col_start:col_end] = img_tensors[idx]

    Image.fromarray(img_arr, "RGB").save(f"{f}.jpg")
