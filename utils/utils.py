import torch


@torch.no_grad()
def get_interpolations(model, device, images, images_per_row=20):
    model.eval()

    img_dim = images.shape[-2]
    channels = 1 if len(images.shape) == 3 else images.shape[1]
    flattened_dim = channels * img_dim * img_dim

    def interpolate(left: torch.Tensor, right: torch.Tensor, num_interps: int) -> torch.Tensor:
        alpha = torch.linspace(0, 1, num_interps)[:, None].to(device)

        interps = left * (1 - alpha) + right * alpha
        return interps

    if model.model_type == "VAE":
        mu, logvar = model.encode(images.view(-1, flattened_dim).float())
        embeddings = model.reparameterize(mu, logvar).cpu()
    elif model.model_type == "AE":
        embeddings = model.encode(images.view(-1, flattened_dim).float())

    interps = []

    # Left and right are not interps, but input images
    num_interps = images_per_row - 2
    for it in range(0, images_per_row + 1, 1):
        interps = interpolate(
            left=embeddings[it],
            right=embeddings[it + 1],
            num_interps=num_interps,
        )
        interp_dec = model.decode(interps)

        line = torch.cat((images[it].unsqueeze(0), interp_dec, images[it + 1].unsqueeze(0)), 0)

        interps.append(line)

    # Complete the loop and append the first image again
    interp = interpolate(embeddings[it + 1], embeddings[0], images_per_row - 4)
    interp = interp.to(device)
    interp_dec = model.decode(interp)

    if img_dim == 32:  # CIFAR10 or any RGB image with 32x32 dimensions
        line = torch.cat((images[it].unsqueeze(0), interp_dec, images[it + 1].unsqueeze(0)), 0)
    else:  # Grayscale images like MNIST
        # Add channel dimension
        line = torch.cat((images[it].unsqueeze(0), interp_dec.squeeze(1), images[it + 1].unsqueeze(0)), 0)

    interps.append(line)

    interps = torch.cat(interps, 0).to(device)
    return interps
