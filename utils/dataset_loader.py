import torch
import os

from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from loguru import logger


def load_split_dataset(device="cpu", root="./data/"):
    """
    Loads the MUTAG dataset. If it already exists, skips re-download.
    Splits into train/validation/test and returns DataLoaders.
    """
    dataset_path = os.path.join(root, "MUTAG")

    # Only download if not already present
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        logger.info(f"Downloading MUTAG dataset to {dataset_path}...")
        dataset = TUDataset(root=root, name="MUTAG")
    else:
        logger.info(f"Loading existing MUTAG dataset from {dataset_path}")
        dataset = TUDataset(root=root, name="MUTAG")

    dataset = dataset.shuffle()

    # Split into training, validation, and test
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, (100, 44, 44), generator=rng
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=100)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader = DataLoader(test_dataset, batch_size=44)

    return train_loader, validation_loader, test_loader


def load_full_dataset(device="cpu", root="./data/"):
    """
    Loads the MUTAG dataset and returns a DataLoader for the entire dataset.
    """
    dataset = TUDataset(root=root, name="MUTAG").shuffle()
    loader = DataLoader(dataset, batch_size=len(dataset))
    return loader
