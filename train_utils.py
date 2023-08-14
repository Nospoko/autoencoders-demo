import torch


def train_epoch(autoencoder, train_loader, optimizer, device, log_interval, epoch):
    autoencoder.model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        data = batch["image"].to(device) / 255.0

        batch_size, channels, height, width = data.shape

        # Ensure data is in the shape: [batch_size, channels, height, width]
        assert data.shape == (batch_size, channels, height, width)  # These assertions will catch any shape mismatches

        optimizer.zero_grad()
        recon_batch = autoencoder.model(data)

        # Ensure recon_batch is in the shape: [batch_size, channels, height, width]
        # if recon_batch.shape != (batch_size, channels, height, width):
        #     print(f"Expected shape: {(batch_size, channels, height, width)}")
        #     print(f"Actual shape: {recon_batch.shape}")
        #     assert False

        loss = autoencoder.loss_function(recon_batch, data)
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


def test_epoch(autoencoder, test_loader, device):
    autoencoder.model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch["image"].to(device) / 255.0
            recon_batch = autoencoder.model(data)
            test_loss += autoencoder.loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss