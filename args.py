"""Centralized argument parsing for DualOrtho federated fine-tuning."""
from __future__ import annotations

import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DualOrtho: Dual-Orthogonal Projection Federated Fine-Tuning"
    )

    # ---- Device & Seed ----
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Dataset ----
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./data")

    # ---- Model ----
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--nf", type=int, default=64,
                        help="Base width multiplier for ResNet")

    # ---- Federated Learning ----
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--global_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--join_ratio", type=float, default=1.0,
                        help="Fraction of clients per round")
    parser.add_argument("--partition", type=str, default="dirichlet",
                        choices=["dirichlet", "shard"])
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet alpha (smaller = more non-IID)")
    parser.add_argument("--batch_size", type=int, default=64)

    # ---- Training ----
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--local_momentum", type=float, default=0.9)
    parser.add_argument("--local_weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    # ---- Stage 0: GPM Memory ----
    parser.add_argument("--memory_layers", type=str, default="layer4",
                        help="Comma-separated layer names for memory extraction")
    parser.add_argument("--memory_var_th", type=float, default=0.97,
                        help="SVD variance threshold for memory basis")
    parser.add_argument("--memory_max_rank", type=int, default=512,
                        help="Max rank for memory basis")
    parser.add_argument("--memory_max_batches", type=int, default=20,
                        help="Max batches for activation collection")

    # ---- Stage 1-2: Drift Probe ----
    parser.add_argument("--toxic_layer", type=str, default="layer4",
                        help="Layer to apply toxic forward projection")
    parser.add_argument("--drift_var_th", type=float, default=0.9,
                        help="SVD variance threshold for drift subspace")
    parser.add_argument("--drift_max_rank", type=int, default=32,
                        help="Max rank for drift subspace")
    parser.add_argument("--regularization", type=float, default=1e-2,
                        help="Ridge regression lambda for AFL closed-form")

    # ---- Logging ----
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=5,
                        help="Log every N rounds")

    args = parser.parse_args()

    # Auto-derive num_classes from dataset
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100

    return args
