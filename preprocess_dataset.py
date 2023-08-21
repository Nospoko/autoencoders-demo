import os

import torch
import numpy as np
from datasets import Dataset, DatasetDict

from train import initialize_model
from utils.data_loader import get_data_loaders


@torch.no_grad()
def create_embeddings(cfg, autoencoder, data_loader):
    """
    Create embeddings for the given data loader using the given autoencoder.
    :param cfg: configuration file
    :param autoencoder: autoencoder instance
    :param data_loader: data loader
    """
    autoencoder.eval()
    autoencoder.to(autoencoder.device)
    embeddings = []
    labels = []

    for batch_idx, batch in enumerate(data_loader):
        data = batch["image"].float().to(autoencoder.device) / 255.0
        target = batch["label"].to(autoencoder.device)

        embedding = autoencoder.encode(data)
        embeddings.append(embedding.cpu().numpy())
        labels.append(target.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels


if __name__ == "__main__":
    checkpoint_path = "checkpoints/AE_MNIST_checkpoint_epoch_10_embSize_32.pt"
    checkpoint = torch.load(checkpoint_path)

    train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"], return_targets=True)

    autoencoder_instance = initialize_model(checkpoint["config"], input_size)
    autoencoder_instance.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(autoencoder_instance.parameters(), checkpoint["config"].train.lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_embeddings, test_labels = create_embeddings(checkpoint["config"], autoencoder_instance, test_loader)

    train_embeddings, train_labels = create_embeddings(checkpoint["config"], autoencoder_instance, train_loader)

    DATASET_NAME = "SneakyInsect/MNIST-preprocessed"
    HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

    test_data = {
        "image": test_loader.dataset["image"].numpy(),
        "label": test_labels,
        "embedding": test_embeddings,
    }
    train_data = {
        "image": train_loader.dataset["image"].numpy(),
        "label": train_labels,
        "embedding": train_embeddings,
    }

    test_dataset = Dataset.from_dict(test_data)
    train_dataset = Dataset.from_dict(train_data)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.push_to_hub(DATASET_NAME, token=HUGGINGFACE_TOKEN)
