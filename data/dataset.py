import os
import kagglehub
import shutil
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from .augmentation import get_transforms


class SubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Grab the raw image path and label directly from ImageFolder underlying samples
        original_index = self.subset.indices[index]
        path, label = self.subset.dataset.samples[original_index]

        img = self.subset.dataset.loader(path)

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def prepare_dataset(dataset_handle, download_dir):
    """Handles auto-downloading and organizing Kaggle files locally."""
    if not os.path.exists(download_dir):
        print(f"Downloading dataset '{dataset_handle}' from Kaggle...")
        cache_path = kagglehub.dataset_download(dataset_handle)
        os.makedirs(download_dir, exist_ok=True)
        for item in os.listdir(cache_path):
            s_path = os.path.join(cache_path, item)
            d_path = os.path.join(download_dir, item)
            if os.path.isdir(s_path):
                shutil.copytree(s_path, d_path, dirs_exist_ok=True)
            else:
                shutil.copy2(s_path, d_path)
        print(f"Data localized to: {download_dir}")
    else:
        print(f"Dataset directory ready at '{download_dir}'.")


def get_dataloaders(data_config):
    """Main pipeline setup. Returns train, val, and test dataloaders along with class metadata."""
    prepare_dataset(data_config["dataset_handle"], data_config["download_dir"])
    transforms_dict = get_transforms()
    base_dir = data_config["download_dir"]

    # 1. Load the full training directory using the base dataset settings
    train_dir = os.path.join(base_dir, "train")
    # Use None initially so we can apply separate transforms to splits via Wrapper
    raw_train_dataset = datasets.ImageFolder(train_dir, transform=None)

    # Get metadata
    class_names = raw_train_dataset.classes
    num_classes = len(class_names)

    # 2. Compute partition sizes
    total_train_len = len(raw_train_dataset)
    train_size = int(data_config["train_split_ratio"] * total_train_len)
    val_size = total_train_len - train_size

    # 3. Random Split
    train_subset, val_subset = random_split(
        raw_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config["seed"]),
    )

    # 4. Wrap subsets to safely isolate train/val transforms
    train_dataset = SubsetWrapper(train_subset, transform=transforms_dict["train"])
    val_dataset = SubsetWrapper(val_subset, transform=transforms_dict["val"])
    test_dataset = datasets.ImageFolder(
        os.path.join(base_dir, "test"), transform=transforms_dict["val"]
    )
    dataset_sizes = {"train": train_size, "val": val_size, "test": len(test_dataset)}

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=data_config["batch_size"], shuffle=True
        ),
        "val": DataLoader(
            val_dataset, batch_size=data_config["batch_size"], shuffle=False
        ),
        "test": DataLoader(
            test_dataset, batch_size=data_config["batch_size"], shuffle=False
        ),
    }

    print(f"Classes Identified ({num_classes}): {class_names}")
    print(
        f"Dataset splits sizes -> Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}"
    )

    return loaders, class_names, num_classes, dataset_sizes
