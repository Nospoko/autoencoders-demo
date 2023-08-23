import sys

import torch
from datasets import load_dataset
from torchvision import transforms

# Define the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
transform_1_channel = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])


# Apply transformations using the map function
def apply_transform(batch):
    transformed_image = transform(batch["image"])
    batch["image"] = transformed_image
    return batch


def apply_transform_1_channel(batch):
    transformed_image = transform_1_channel(batch["image"])
    batch["image"] = transformed_image
    return batch


def get_data_loaders(cfg, return_targets=False, return_data_variance=False):
    """
    Returns the data loaders for the dataset specified in cfg.dataset.
    If return_targets is True, it also returns the targets.
    """
    columns_to_load = ["image"] if not return_targets else ["image", "label"]

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
        columns_to_load = ["signal"] if not return_targets else ["signal", "label"]
        data["train"].set_format("torch", columns=columns_to_load)
        data["test"].set_format("torch", columns=columns_to_load)

        train_loader = torch.utils.data.DataLoader(data["train"], batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
        test_loader = torch.utils.data.DataLoader(data["test"], batch_size=cfg.train.batch_size, shuffle=False)
        return train_loader, test_loader, input_size

    else:
        print("Dataset not supported")
        sys.exit()

    if cfg.dataset.name == "MNIST" or cfg.dataset.name == "AmbiguousMNIST" or cfg.dataset.name == "FashionMNIST":
        transform_to_apply = apply_transform_1_channel  # Use the 1-channel transform
    elif cfg.dataset.name == "CIFAR10":
        transform_to_apply = apply_transform
    else:
        transform_to_apply = None

    if transform_to_apply is not None:
        data["train"] = data["train"].map(transform_to_apply)
        data["test"] = data["test"].map(transform_to_apply)

    data["train"].set_format("torch", columns=columns_to_load)
    data["test"].set_format("torch", columns=columns_to_load)

    train_loader = torch.utils.data.DataLoader(data["train"], batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data["test"], batch_size=cfg.train.batch_size, shuffle=False)

    return train_loader, test_loader, input_size
