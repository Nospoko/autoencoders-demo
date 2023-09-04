import torch
import torch.nn as nn

from utils.visualizations import interpolate_embeddings


def make_interpolation_grid(
    model: nn.Module,
    images: torch.Tensor,
    n_interps: int,
) -> torch.Tensor:
    n_images = images.shape[0]

    model_input = images.view(n_images, -1).float()
    mu, logvar = model.encode(model_input)
    embeddings = model.reparameterize(mu, logvar)

    interpolated_lines = []
    for it in range(n_images - 1):
        left = embeddings[it]
        right = embeddings[it + 1]
        interpolated_embeddings = interpolate_embeddings(
            left=left,
            right=right,
            num_interps=n_interps,
        )

        decoded_interpolations = model.decode(interpolated_embeddings)
        line = [images[it].unsqueeze(0)]
        line += decoded_interpolations.split(1)
        line += [images[it + 1].unsqueeze(0)]

        # Last axis is width
        line = torch.cat(line, dim=-1).squeeze(0)

        interpolated_lines.append(line)

    # Second to last axis is height
    grid = torch.cat(interpolated_lines, dim=-2)
    return grid
