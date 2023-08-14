import torch


def train_epoch(autoencoder, train_loader, optimizer, device, log_interval, epoch, loss_function):
    autoencoder.model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["image"].to(device) / 255.0

        # There is some issue with chanel ordering after introduction of cifar10
        if len(data.shape) == 4:
            # permute chanel order to NCHW from NHWC
            data = data.permute(0, 3, 1, 2)
        if len(data.shape) == 3:
            # add chanel dim
            data = data.unsqueeze(1)

        optimizer.zero_grad()
        recon_batch = autoencoder.model(data)

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


def test_epoch(autoencoder, test_loader, device, loss_function):
    autoencoder.model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device) / 255.0
            recon_batch = autoencoder.model(data)
            # ordering issue
            if len(data.shape) == 4:
                # permute chanel order to NCHW from NHWC
                data = data.permute(0, 3, 1, 2)
            if len(data.shape) == 3:
                # add chanel dim
                data = data.unsqueeze(1)

            test_loss += loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def prepare_loss_function(train_cfg):
    if train_cfg.loss_function == "BCE":
        return torch.nn.BCELoss(reduction="sum")
    elif train_cfg.loss_function == "MSE":
        return torch.nn.MSELoss(reduction="sum")

    raise ValueError("Invalid loss function: {}. Available options are: 'BCE', 'MSE'.".format(train_cfg.loss_function))
