import torch

from models.variational_autoencoder import VAELoss


def train_epoch(autoencoder, train_loader, optimizer, device, log_interval, epoch, loss_function):
    autoencoder.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["image"].to(device) / 255.0

        if len(data.shape) == 4:
            data = data.permute(0, 3, 1, 2)
        if len(data.shape) == 3:
            data = data.unsqueeze(1)

        optimizer.zero_grad()

        if isinstance(loss_function, VAELoss):
            recon_batch, mu, logvar = autoencoder(data)
            loss = loss_function(recon_batch, data, mu, logvar)
        else:
            recon_batch = autoencoder(data)
            loss = loss_function(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_epoch(autoencoder, test_loader, device, loss_function):
    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device) / 255.0

            if len(data.shape) == 4:
                data = data.permute(0, 3, 1, 2)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            if isinstance(loss_function, VAELoss):
                recon_batch, mu, logvar = autoencoder(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
            else:
                recon_batch = autoencoder(data)
                test_loss += loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def prepare_loss_function(train_cfg):
    if train_cfg.loss_function == "BCE":
        return torch.nn.BCELoss(reduction="sum")
    elif train_cfg.loss_function == "MSE":
        return torch.nn.MSELoss(reduction="sum")
    elif train_cfg.loss_function == "VAE":
        return VAELoss()

    raise ValueError(f"Invalid loss function: {train_cfg.loss_function}. Available options are: 'BCE', 'MSE', 'VAE'.")


def train_epoch_ecg(autoencoder, train_loader, optimizer, device, log_interval, epoch, loss_function, verbose=True):
    autoencoder.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["signal"].to(device)
        optimizer.zero_grad()
        recon_batch = autoencoder(data)

        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if verbose and batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    train_loss / (batch_idx + 1),
                )
            )

    avg_loss = train_loss / len(train_loader)
    return avg_loss


def test_epoch_ecg(autoencoder, test_loader, device, loss_function, verbose=True):
    autoencoder.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["signal"].to(device)
            recon_batch = autoencoder(data)

            loss = loss_function(recon_batch, data)
            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)

    if verbose:
        print(f"====> Test set loss: {avg_loss:.4f}")

    return avg_loss
