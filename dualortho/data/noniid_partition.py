"""Non-IID data partitioning for federated learning.

Supports:
    - Dirichlet allocation (dir): controls heterogeneity via alpha
    - Shard-based partition (shard): each client gets a limited number of class shards
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 10,
) -> Dict[int, np.ndarray]:
    """Dirichlet-based Non-IID partition.

    Smaller alpha = more heterogeneous. alpha -> inf = IID.

    Args:
        labels: [N] array of integer labels.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed.
        min_samples: Minimum samples per client.

    Returns:
        Dict mapping client_id -> array of sample indices.
    """
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    N = len(labels)

    # Initialize empty index lists for each client
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Balance: clip clients that already have too many samples
        proportions = np.array([
            p * (len(client_indices[i]) < N / num_clients)
            for i, p in enumerate(proportions)
        ])
        proportions /= proportions.sum() + 1e-10

        # Split class indices according to proportions
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        class_splits = np.split(class_idx, splits)

        for i, split in enumerate(class_splits):
            client_indices[i].extend(split.tolist())

    # Convert to arrays and shuffle
    result = {}
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        result[i] = idx

    return result


def partition_shard(
    labels: np.ndarray,
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Shard-based Non-IID partition.

    Sorts data by label, divides into shards, assigns shards to clients.
    shards_per_client=2 is the standard non-IID setting from McMahan et al.

    Args:
        labels: [N] array of integer labels.
        num_clients: Number of federated clients.
        shards_per_client: Number of shards per client.
        seed: Random seed.

    Returns:
        Dict mapping client_id -> array of sample indices.
    """
    rng = np.random.RandomState(seed)
    N = len(labels)
    num_shards = num_clients * shards_per_client
    shard_size = N // num_shards

    # Sort by label
    sorted_idx = np.argsort(labels)

    # Create shards
    shards = [
        sorted_idx[i * shard_size: (i + 1) * shard_size]
        for i in range(num_shards)
    ]
    rng.shuffle(shards)

    # Assign shards to clients
    result = {}
    for i in range(num_clients):
        client_shards = shards[i * shards_per_client: (i + 1) * shards_per_client]
        result[i] = np.concatenate(client_shards)

    return result


def compute_class_distribution(
    labels: np.ndarray,
    client_indices: Dict[int, np.ndarray],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    """Compute per-client class distribution for analysis/logging.

    Returns:
        Dict mapping client_id -> [num_classes] array of sample counts.
    """
    dist = {}
    for cid, idx in client_indices.items():
        counts = np.zeros(num_classes, dtype=int)
        for c in range(num_classes):
            counts[c] = int((labels[idx] == c).sum())
        dist[cid] = counts
    return dist
