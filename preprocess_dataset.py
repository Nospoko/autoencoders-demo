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
    checkpoint_path_list = [
        "checkpoints/AE_MNIST_checkpoint_epoch_10_embSize_32.pt",
        "checkpoints/AE_MNIST_checkpoint_epoch_10_embSize_16.pt",
        "checkpoints/AE_MNIST_checkpoint_epoch_10_embSize_8.pt",
    ]
    test_embeddings_list = []
    test_labels_list = []
    train_embeddings_list = []
    train_labels_list = []

    # make sure the loaders are the same
    checkpoint = torch.load(checkpoint_path_list[0])
    train_loader, test_loader, input_size = get_data_loaders(checkpoint["config"], return_targets=True)

    for checkpoint_path in checkpoint_path_list:
        if checkpoint_path != checkpoint_path_list[0]:
            checkpoint = torch.load(checkpoint_path)

        autoencoder_instance = initialize_model(checkpoint["config"], input_size)
        autoencoder_instance.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.Adam(autoencoder_instance.parameters(), checkpoint["config"].train.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        test_embeddings, test_labels = create_embeddings(checkpoint["config"], autoencoder_instance, test_loader)
        test_embeddings_list.append(test_embeddings)
        test_labels_list.append(test_labels)
        train_embeddings, train_labels = create_embeddings(checkpoint["config"], autoencoder_instance, train_loader)
        train_embeddings_list.append(train_embeddings)
        train_labels_list.append(train_labels)

    DATASET_NAME = "SneakyInsect/MNIST-preprocessed"
    HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

    test_data = {
        "image": test_loader.dataset["image"].numpy(),
        "label": test_labels,
        "embedding32": test_embeddings_list[0],
        "embedding16": test_embeddings_list[1],
        "embedding8": test_embeddings_list[2],
    }
    train_data = {
        "image": train_loader.dataset["image"].numpy(),
        "label": train_labels,
        "embedding32": train_embeddings_list[0],
        "embedding16": train_embeddings_list[1],
        "embedding8": train_embeddings_list[2],
    }
    # print(test_data["image"].shape, test_data["label"].shape, test_data["embedding"].shape)
    # print(train_data["image"].shape, train_data["label"].shape, train_data["embedding"].shape)
    test_dataset = Dataset.from_dict(test_data)
    train_dataset = Dataset.from_dict(train_data)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    # dataset_dict.save_to_disk(DATASET_NAME)
    dataset_dict.push_to_hub(DATASET_NAME, token=HUGGINGFACE_TOKEN)
