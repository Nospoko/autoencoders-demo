import sys

import torch
from datasets import load_dataset


def get_data_loaders(cfg):
    """
    Returns the data loaders for the dataset specified in cfg.dataset
    """
    if cfg.dataset.name == "MNIST":
        data = load_dataset("mnist")
    elif cfg.dataset.name == "AmbiguousMNIST":
        data = load_dataset("mweiss/mnist_ambiguous")
    elif cfg.dataset.name == "FashionMNIST":
        data = load_dataset("fashion_mnist")
    elif cfg.dataset.name == "CIFAR10":
        data = load_dataset("cifar10")
        data["train"] = data["train"].rename_column("img", "image")
        data["test"] = data["test"].rename_column("img", "image")
    else:
        print("Dataset not supported")
        sys.exit()

    data["train"].set_format("torch", columns=["image"])
    data["test"].set_format("torch", columns=["image"])

    input_size = data["train"][0]["image"].shape
    print("Input size: ", tuple(input_size))

    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data["test"], batch_size=cfg.train.batch_size, shuffle=False)

    return train_loader, test_loader
