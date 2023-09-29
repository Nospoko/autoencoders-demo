from omegaconf import DictConfig
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

# Define the transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class SizedDataset(Dataset):
    def __init__(self, dataset: HFDataset, input_size: tuple):
        self.dataset = dataset.with_format("torch")
        self.input_size = input_size

        # I don't like this but what can you do
        self.data_key = "image" if len(input_size) == 3 else "signal"

    def __rich_repr__(self):
        yield "SizedDataset"
        yield "data_key", self.data_key
        yield "input_size", self.input_size
        yield "n_samples", len(self)

    @property
    def n_channels(self) -> int:
        return self.input_size[0]

    def __getitem__(self, idx: int) -> dict:
        record = self.dataset[idx]
        data = record[self.data_key]

        # Grey images are missing the channel dim
        if self.data_key == "image" and self.n_channels == 1:
            axis = 1 if isinstance(idx, slice) else 0
            data = data.unsqueeze(axis)

        # Special treatment for CIFAR
        if self.dataset.builder_name == "cifar10":
            if isinstance(idx, slice):
                data = data.permute(0, 3, 2, 1)
            else:
                data = data.permute(2, 1, 0)

        out = {
            self.data_key: data,
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

    train_dataset = SizedDataset(dataset=data["train"], input_size=input_size)
    test_dataset = SizedDataset(dataset=data["test"], input_size=input_size)

    return train_dataset, test_dataset
