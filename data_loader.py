import sys

import torch
from datasets import load_dataset


def get_data_loaders(args):
    """
    Returns the data loaders for the dataset specified in args.dataset
    """
    if args.dataset.name == "MNIST":
        data = load_dataset("mnist")
    elif args.dataset.name == "AmbiguousMNIST":
        data = load_dataset("mweiss/mnist_ambiguous")
    elif args.dataset.name == "FashionMNIST":
        data = load_dataset("fashion_mnist")
    elif args.dataset.name == "CIFAR10":
        data = load_dataset("cifar10")
        data["train"] = data["train"].rename_column("img", "image")
        data["test"] = data["test"].rename_column("img", "image")
    else:
        print("Dataset not supported")
        sys.exit()

    data["train"].set_format("torch", columns=["image"])
    data["test"].set_format("torch", columns=["image"])

    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=args.hyperparameters.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data["test"], batch_size=args.hyperparameters.batch_size, shuffle=False)

    return train_loader, test_loader
