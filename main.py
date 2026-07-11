"""DualOrtho Federated Fine-Tuning -- Main Entry Point.

Full federated learning loop:
    1. Initialize global model and partition data across clients
    2. Each round: select clients, run DualOrtho 4-stage pipeline, aggregate
    3. Evaluate global model on test set after each round

Usage:
    python main.py --alg dualortho --dataset cifar10 --num_clients 10 --global_rounds 50
    python main.py --alg dualortho --dataset cifar100 --alpha 0.1 --drift_max_rank 64
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from args import get_args
from algorithms import dualortho_alg
from dualortho.data.fl_dataset import build_fl_clients
from dualortho.data.noniid_partition import compute_class_distribution
from dualortho.models.resnet_cifar import resnet18_cifar, resnet34_cifar


# ============================================================
# Setup helpers
# ============================================================

def setup_logging(log_dir: str, run_name: str):
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, log_file


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    """Build ResNet model."""
    if args.model == "resnet18":
        return resnet18_cifar(args.num_classes, nf=args.nf)
    elif args.model == "resnet34":
        return resnet34_cifar(args.num_classes, nf=args.nf)
    raise ValueError(f"Unknown model: {args.model}")


def log_model_summary(logger, model, model_name: str = "unknown"):
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}, Total params: {total_params:,}")


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)

    # Use paths relative to main.py's location (unless user explicitly set them)
    _main_dir = os.path.dirname(os.path.abspath(__file__))
    if args.log_dir == "./logs":
        args.log_dir = os.path.join(_main_dir, "logs")
    if args.save_dir == "./checkpoints":
        args.save_dir = os.path.join(_main_dir, "checkpoints")

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.model}_{args.partition}_a{args.alpha}_{timestamp}"
    args.run_name = run_name
    logger, log_file = setup_logging(args.log_dir, run_name)

    logger.info("=" * 70)
    logger.info(f"Run: {run_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Algorithm: {args.alg}")
    logger.info("-" * 70)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)

    # ---- Data ----
    logger.info("Preparing federated data...")
    client_loaders, test_loader, num_classes, in_channels, client_indices = build_fl_clients(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        partition=args.partition,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        data_root=args.data_root,
    )
    args.num_classes = num_classes

    labels = np.array(client_loaders[0].dataset.dataset.targets)
    logger.info(f"Clients: {args.num_clients}, Test: {len(test_loader.dataset)} samples")
    logger.info(f"Classes: {num_classes}, Model: {args.model}")

    # ---- Global model ----
    global_model = build_model(args).to(device)
    log_model_summary(logger, global_model, model_name=args.model)

    # ---- Client models ----
    nets = {i: build_model(args).to('cpu') for i in range(args.num_clients)}

    # ---- Per-round client selection ----
    n_clients_per_round = max(1, int(args.num_clients * args.join_ratio))
    party_list_rounds = []
    for _ in range(args.global_rounds):
        party_list_rounds.append(random.sample(range(args.num_clients), n_clients_per_round))

    # ---- Algorithm dispatch ----
    if args.alg == "dualortho":
        logger.info("Starting DualOrtho training...")
        record_test_acc_list, best_test_acc = dualortho_alg(
            args=args,
            n_rounds=args.global_rounds,
            nets=nets,
            global_model=global_model,
            party_list_rounds=party_list_rounds,
            net_dataidx_map=client_indices,
            train_local_dls=client_loaders,
            test_dl=test_loader,
            traindata_cls_counts=None,
            device=device,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.alg}")

    logger.info(f"Best test accuracy: {best_test_acc*100:.2f}%")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
