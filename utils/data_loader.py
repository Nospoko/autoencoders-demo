from omegaconf import DictConfig
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

# Define the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class SizedDataset(Dataset):
    def __init__(self, dataset: HFDataset, input_size: tuple[int, int, int]):
        self.dataset = dataset.with_format("torch")
        self.input_size = input_size

    @property
    def n_channels(self) -> int:
        return self.input_size[0]

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        image = record["image"]
        label = record["label"]

        if self.n_channels == 1:
            axis = 1 if isinstance(idx, slice) else 0
            image = image.unsqueeze(axis)

        out = {
            "image": image,
            "label": label,
        }

        return out

    def __len__(self) -> int:
        return len(self.dataset)


# Apply transformations using the map function
def apply_transform(batch):
    transformed_image = transform(batch["image"])
    batch["image"] = transformed_image
    return batch


def prepare_dataset(cfg: DictConfig) -> tuple[SizedDataset, SizedDataset]:
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

    elif cfg.dataset.name == "CIFAR10":
        data["train"] = data["train"].map(apply_transform)
        data["test"] = data["test"].map(apply_transform)

    else:
        raise ValueError("Dataset not supported")

    # train_loader = DataLoader(data["train"], batch_size=cfg.train.batch_size, shuffle=True)
    # test_loader = DataLoader(data["test"], batch_size=cfg.train.batch_size, shuffle=False)

    train_dataset = SizedDataset(dataset=data["train"], input_size=input_size)
    test_dataset = SizedDataset(dataset=data["test"], input_size=input_size)

    return train_dataset, test_dataset
