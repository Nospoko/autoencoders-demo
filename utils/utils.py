import torch
import numpy as np


def get_interpolations(cfg, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        img_dim = images.shape[-2]  # For 28 (MNIST) or 32 (CIFAR10)
        channels = 3 if img_dim == 32 else 1
        flattened_dim = channels * img_dim * img_dim

        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps + 2)
            interps = []
            for a in alpha:
                interps.append(a * t2.reshape(1, -1) + (1 - a) * t1.reshape(1, -1))
            return torch.cat(interps, 0)

        if cfg.model.type == "VAE":
            mu, logvar = model.encode(images.view(-1, flattened_dim))
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif cfg.model.type == "AE":
            embeddings = model.encode(images.view(-1, flattened_dim))

        interps = []
        for i in range(0, images_per_row + 1, 1):
            interp = interpolate(embeddings[i], embeddings[i + 1], images_per_row - 4)
            interp = interp.to(device)
            interp_dec = model.decode(interp)

            if img_dim == 32:  # CIFAR10 or any RGB image with 32x32 dimensions
                line = torch.cat((images[i].unsqueeze(0), interp_dec, images[i + 1].unsqueeze(0)), 0)
            else:  # Grayscale images like MNIST
                # Add channel dimension
                line = torch.cat((images[i].unsqueeze(0).unsqueeze(1), interp_dec, images[i + 1].unsqueeze(0).unsqueeze(1)), 0)

            interps.append(line)

        # Complete the loop and append the first image again
        interp = interpolate(embeddings[i + 1], embeddings[0], images_per_row - 4)
        interp = interp.to(device)
        interp_dec = model.decode(interp)

        if img_dim == 32:  # CIFAR10 or any RGB image with 32x32 dimensions
            line = torch.cat((images[i].unsqueeze(0), interp_dec, images[i + 1].unsqueeze(0)), 0)
        else:  # Grayscale images like MNIST
            # Add channel dimension
            line = torch.cat((images[i].unsqueeze(0).unsqueeze(1), interp_dec, images[i + 1].unsqueeze(0).unsqueeze(1)), 0)

        interps.append(line)

        interps = torch.cat(interps, 0).to(device)
    return interps
