import sys

import torch
from datasets import load_dataset


def get_data_loaders(cfg):
    """
    Returns the data loaders for the dataset specified in cfg.dataset
    """
    if cfg.dataset.name == "MNIST":
        data = load_dataset("mnist")
        input_size = (1, 28, 28)
    elif cfg.dataset.name == "AmbiguousMNIST":
        data = load_dataset("mweiss/mnist_ambiguous")
        input_size = (1, 28, 28)
    elif cfg.dataset.name == "FashionMNIST":
        data = load_dataset("fashion_mnist")
        input_size = (1, 28, 28)
    elif cfg.dataset.name == "CIFAR10":
        data = load_dataset("cifar10")
        data["train"] = data["train"].rename_column("img", "image")
        data["test"] = data["test"].rename_column("img", "image")
        input_size = (3, 32, 32)
    elif cfg.dataset.name == "ltafdb":
        data = load_dataset("roszcz/ecg-segmentation-ltafdb")
        input_size = (2, 1000)
        data["train"].set_format("torch", columns=["signal"])
        data["test"].set_format("torch", columns=["signal"])
        return data["train"], data["test"], input_size
    else:
        print("Dataset not supported")
        sys.exit()

    data["train"].set_format("torch", columns=["image"])
    data["test"].set_format("torch", columns=["image"])

    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data["test"], batch_size=cfg.train.batch_size, shuffle=False)

    return train_loader, test_loader, input_size
