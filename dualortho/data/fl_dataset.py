"""Federated Learning dataset builder.

Builds per-client DataLoaders from a torchvision dataset with Non-IID partitioning.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dualortho.data.noniid_partition import (
    partition_dirichlet,
    partition_shard,
    compute_class_distribution,
)

logger = logging.getLogger(__name__)

# Standard CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Standard CIFAR-100 normalization
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar_transforms(dataset: str):
    """Get standard train/test transforms for CIFAR-10/100."""
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform


def build_fl_clients(
    dataset_name: str,
    num_clients: int,
    partition: str = "dirichlet",
    alpha: float = 0.5,
    shards_per_client: int = 2,
    batch_size: int = 64,
    seed: int = 42,
    data_root: str = "./data",
) -> Tuple[List[DataLoader], DataLoader, int, int, dict]:
    """Build federated client DataLoaders with Non-IID partitioning.

    Args:
        dataset_name: 'cifar10' or 'cifar100'.
        num_clients: Number of federated clients.
        partition: 'dirichlet' or 'shard'.
        alpha: Dirichlet alpha (only used when partition='dirichlet').
        shards_per_client: Shards per client (only used when partition='shard').
        batch_size: Batch size for client DataLoaders.
        seed: Random seed for partitioning.
        data_root: Root directory for dataset download.

    Returns:
        client_loaders: List of DataLoaders, one per client.
        test_loader: Shared test DataLoader.
        num_classes: Number of classes.
        in_channels: Number of input channels (3 for CIFAR).
    """
    train_transform, test_transform = get_cifar_transforms(dataset_name)

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            data_root, train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            data_root, train=False, download=True, transform=test_transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Extract labels for partitioning
    labels = np.array(train_dataset.targets)

    # Partition
    if partition == "dirichlet":
        client_indices = partition_dirichlet(
            labels, num_clients, alpha=alpha, seed=seed
        )
    elif partition == "shard":
        client_indices = partition_shard(
            labels, num_clients, shards_per_client=shards_per_client, seed=seed
        )
    else:
        raise ValueError(f"Unknown partition: {partition}")

    # Log distribution
    dist = compute_class_distribution(labels, client_indices, num_classes)
    for cid in range(min(3, num_clients)):  # log first 3 clients
        nonzero = {c: n for c, n in enumerate(dist[cid]) if n > 0}
        logger.info(f"  Client {cid}: {len(client_indices[cid])} samples, classes={nonzero}")

    # Build DataLoaders
    client_loaders = []
    for cid in range(num_clients):
        subset = Subset(train_dataset, client_indices[cid])
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False,
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    in_channels = 3
    return client_loaders, test_loader, num_classes, in_channels, client_indices

