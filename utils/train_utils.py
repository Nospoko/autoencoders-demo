from typing import Callable

import torch
import torch.nn as nn
from torch.utisl.data import DataLoader

from models.variational_autoencoder import VAELoss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_interval: int,
    loss_function: Callable,
):
    model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["image"].to(device) / 255.0

        if len(data.shape) == 3:
            data = data.unsqueeze(1)

        optimizer.zero_grad()

        if isinstance(loss_function, VAELoss):
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
        else:
            recon_batch = model(data)
            loss = loss_function(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Training [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    train_loss /= len(train_loader.dataset)
    return train_loss


def test_epoch(model, test_loader, device, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device) / 255.0

            if len(data.shape) == 3:
                data = data.unsqueeze(1)

            if isinstance(loss_function, VAELoss):
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
            else:
                recon_batch = model(data)
                test_loss += loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def prepare_loss_function(loss_function_name: str):
    if loss_function_name == "BCE":
        return torch.nn.BCELoss(reduction="sum")
    elif loss_function_name == "MSE":
        return torch.nn.MSELoss()
    elif loss_function_name == "VAE":
        return VAELoss()
    elif loss_function_name == "VAE_MSE":
        return VAELoss(recon_loss="MSE")

    raise ValueError(f"Invalid loss function: {loss_function_name}. Available options are: 'BCE', 'MSE', 'VAE'.")


def train_epoch_ecg(model, train_loader, optimizer, device, log_interval, epoch, loss_function, verbose=True):
    model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["signal"].to(device)
        optimizer.zero_grad()
        recon_batch = model(data)

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


def test_epoch_ecg(model, test_loader, device, loss_function, verbose=True):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["signal"].to(device)
            recon_batch = model(data)

            loss = loss_function(recon_batch, data)
            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)

    if verbose:
        print(f"====> Test set loss: {avg_loss:.4f}")

    return avg_loss


def train_epoch_vqvae(
    model,
    train_loader,
    optimizer,
    device,
    log_interval,
    epoch,
    loss_function,
    beta,
    use_ema,
    best_train_loss,
):
    model.train()
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for batch_idx, train_tensors in enumerate(train_loader):
        optimizer.zero_grad()

        imgs = train_tensors["image"].to(device)

        out = model(imgs)
        recon_error = loss_function(out["x_recon"], imgs)
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]  # cfg.train.beta is the configurable beta term

        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_train_loss = total_train_loss / n_train
            avg_recon_error = total_recon_error / n_train

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(imgs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]"
            )
            print(f"Avg Train Loss: {avg_train_loss}")
            print(f"Best Train Loss: {best_train_loss}")
            print(f"Avg Recon Error: {avg_recon_error}")

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0
    return avg_train_loss


def test_epoch_vqvae(model, test_loader, device, loss_function, beta):
    model.eval()
    total_test_loss = 0
    n_test = 0
    with torch.no_grad():
        for batch_idx, test_tensors in enumerate(test_loader):
            imgs = test_tensors["image"].to(device)
            out = model(imgs)
            recon_error = loss_function(out["x_recon"], imgs)
            loss = recon_error + beta * out["commitment_loss"]

            if "dictionary_loss" in out and out["dictionary_loss"] is not None:
                loss += out["dictionary_loss"]

            total_test_loss += loss.item()
            n_test += 1

    avg_test_loss = total_test_loss / n_test
    print(f"====> Test set loss: {avg_test_loss:.4f}")
    return avg_test_loss
