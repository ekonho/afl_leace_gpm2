"""DualOrtho Federated Fine-Tuning -- Main Entry Point.

Full federated learning loop:
    1. Initialize global model and partition data across clients
    2. Each round: select clients, run DualOrtho 4-stage pipeline, aggregate
    3. Evaluate global model on test set after each round

Usage:
    python main.py --dataset cifar10 --num_clients 10 --global_rounds 50
    python main.py --dataset cifar100 --alpha 0.1 --drift_max_rank 64
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

from args import get_args
from dualortho.data.fl_dataset import build_fl_clients
from dualortho.data.noniid_partition import compute_class_distribution
from dualortho.models.resnet_cifar import resnet18_cifar, resnet34_cifar
from dualortho.pipeline import DualOrthoPipeline


# ============================================================
# Setup helpers
# ============================================================

def setup_logging(log_dir: str, run_name: str):
    """Setup logging to both file and console. Returns (logger, log_file_path)."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # Console handler
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
    """Build ResNet backbone + head."""
    if args.model == "resnet18":
        model = resnet18_cifar(args.num_classes, nf=args.nf)
    elif args.model == "resnet34":
        model = resnet34_cifar(args.num_classes, nf=args.nf)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model


def log_model_summary(logger, model):
    """Log model architecture and parameter counts."""
    total_params = 0
    logger.info("Model architecture:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        total_params += n_params
        logger.info(f"  {name}: {n_params:,} params")
    logger.info(f"  Total: {total_params:,} params")


def log_data_distribution(logger, labels, client_indices, num_classes, num_clients):
    """Log per-client data distribution."""
    dist = compute_class_distribution(labels, client_indices, num_classes)
    logger.info("Data distribution:")
    for cid in range(num_clients):
        counts = dist[cid]
        total = int(counts.sum())
        nonzero = {int(c): int(n) for c, n in enumerate(counts) if n > 0}
        logger.info(f"  Client {cid}: {total} samples, classes={nonzero}")


# ============================================================
# Federated Averaging
# ============================================================

def federated_averaging(global_model, client_states, client_counts):
    """Weighted FedAvg aggregation of client model state dicts."""
    global_state = global_model.state_dict()
    total = sum(client_counts)
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for state, count in zip(client_states, client_counts):
            weight = count / total
            global_state[key] += state[key].float() * weight
    global_model.load_state_dict(global_state)


def extract_global_head_weight(global_model):
    """Extract the global classification head weight matrix [D_in, D_out]."""
    return global_model.head.fc.weight.data.T.clone()


# ============================================================
# Main FL Loop
# ============================================================

def main():
    args = get_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.model}_{args.partition}_a{args.alpha}_{timestamp}"
    logger, log_file = setup_logging(args.log_dir, run_name)

    # ---- Log all args ----
    logger.info("=" * 70)
    logger.info(f"Run: {run_name}")
    logger.info(f"Log file: {log_file}")
    logger.info("-" * 70)
    logger.info("Arguments:")
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

    # Log data distribution
    labels = np.array(client_loaders[0].dataset.dataset.targets)  # full labels
    log_data_distribution(logger, labels, client_indices, num_classes, args.num_clients)
    logger.info(f"Test set: {len(test_loader.dataset)} samples")

    # ---- Global model ----
    global_model = build_model(args).to(device)
    log_model_summary(logger, global_model)

    # JSON log setup
    os.makedirs(args.log_dir, exist_ok=True)
    json_log_path = os.path.join(args.log_dir, f"{run_name}.json")
    run_record = {
        "run_name": run_name,
        "timestamp": timestamp,
        "args": vars(args),
        "data": {
            "num_clients": args.num_clients,
            "num_classes": num_classes,
            "partition": args.partition,
            "alpha": args.alpha,
            "client_sample_counts": [len(client_indices[i]) for i in range(args.num_clients)],
        },
        "model": {
            "name": args.model,
            "nf": args.nf,
            "total_params": sum(p.numel() for p in global_model.parameters()),
        },
        "rounds": [],
    }

    # ---- FL Training ----
    best_acc = 0.0
    memory_layers = [
        "backbone." + s.strip() if not s.strip().startswith("backbone.") else s.strip()
        for s in args.memory_layers.split(",")
    ]

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Training begins: {args.global_rounds} rounds, {args.num_clients} clients")
    logger.info(f"Memory layers: {memory_layers}")
    logger.info("=" * 70)

    for rnd in range(args.global_rounds):
        round_start = time.time()
        logger.info("")
        logger.info("-" * 50)
        logger.info(f"Round {rnd+1}/{args.global_rounds}")
        logger.info("-" * 50)

        # Select clients
        num_selected = max(1, int(args.num_clients * args.join_ratio))
        selected = sorted(random.sample(range(args.num_clients), num_selected))
        logger.info(f"Selected clients: {selected} ({num_selected}/{args.num_clients})")

        global_W = extract_global_head_weight(global_model)
        client_states = []
        client_sample_counts = []
        round_client_details = []

        for cid in selected:
            client_start = time.time()
            logger.info(f"  Client {cid} ({len(client_loaders[cid].dataset)} samples):")

            local_model = copy.deepcopy(global_model).to(device)
            pipe = DualOrthoPipeline(
                args, local_model, device,
                memory_layers=memory_layers,
                backbone_name="backbone",
                grad_proj_layers=memory_layers,
            )

            # Stages 0-2
            pipe.prepare(
                client_loaders[cid],
                global_W.to(device),
                memory_max_batches=args.memory_max_batches,
            )

            # Read stage results
            memory_ranks = {name: mem.rank for name, mem in pipe.memories.items()}
            memory_energy = {name: round(mem.explained_variance, 4) for name, mem in pipe.memories.items()}
            drift_rank = pipe.drift.rank if pipe.drift else 0
            drift_energy = round(pipe.drift.explained_variance, 4) if pipe.drift else 0.0

            logger.info(f"    Stage 0 memory: ranks={memory_ranks}, energy={memory_energy}")
            logger.info(f"    Stage 1-2 drift: rank={drift_rank}, energy={drift_energy}")

            # Stage 3
            optimizer = torch.optim.SGD(
                local_model.parameters(), lr=args.local_lr,
                momentum=args.local_momentum, weight_decay=args.local_weight_decay,
            )
            criterion = torch.nn.CrossEntropyLoss()
            pipe.set_optimizer(optimizer)
            pipe.set_criterion(criterion)

            epoch_metrics = []
            for ep in range(args.local_epochs):
                metrics = pipe.train_one_epoch(client_loaders[cid], grad_clip=args.grad_clip)
                epoch_metrics.append({"loss": round(metrics["loss"], 4), "acc": round(metrics["acc"], 4)})
                logger.info(
                    f"    Epoch {ep+1}/{args.local_epochs}: "
                    f"loss={metrics['loss']:.4f}, acc={metrics['acc']*100:.1f}%"
                )

            pipe.cleanup()
            client_time = time.time() - client_start

            # Collect results
            client_states.append(copy.deepcopy(local_model.state_dict()))
            client_sample_counts.append(len(client_loaders[cid].dataset))

            client_detail = {
                "client_id": cid,
                "n_samples": len(client_loaders[cid].dataset),
                "memory_ranks": memory_ranks,
                "memory_energy": memory_energy,
                "drift_rank": drift_rank,
                "drift_energy": drift_energy,
                "epoch_metrics": epoch_metrics,
                "final_loss": epoch_metrics[-1]["loss"],
                "final_acc": epoch_metrics[-1]["acc"],
                "time_sec": round(client_time, 1),
            }
            round_client_details.append(client_detail)
            logger.info(f"    Done in {client_time:.1f}s")

            del local_model, pipe
            torch.cuda.empty_cache()

        # FedAvg
        logger.info("  Aggregating...")
        federated_averaging(global_model, client_states, client_sample_counts)

        # Evaluate
        global_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = global_model(images)
                loss = criterion(outputs, labels_batch)
                test_loss += float(loss) * images.size(0)
                correct += int((outputs.argmax(1) == labels_batch).sum())
                total += images.size(0)

        acc = correct / total
        avg_loss = test_loss / total
        round_time = time.time() - round_start

        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                global_model.state_dict(),
                os.path.join(args.save_dir, f"{run_name}_best.pt"),
            )

        logger.info(f"  Global test: loss={avg_loss:.4f}, acc={acc*100:.2f}%")
        logger.info(f"  Best so far: {best_acc*100:.2f}%")
        logger.info(f"  Round time: {round_time:.1f}s")

        # JSON log (append round)
        run_record["rounds"].append({
            "round": rnd + 1,
            "test_loss": round(avg_loss, 4),
            "test_acc": round(acc, 4),
            "best_acc": round(best_acc, 4),
            "time_sec": round(round_time, 1),
            "selected_clients": selected,
            "client_details": round_client_details,
        })
        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)

    # ---- Final summary ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"  Best test accuracy: {best_acc*100:.2f}%")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  JSON log: {json_log_path}")
    logger.info(f"  Checkpoint: {os.path.join(args.save_dir, run_name + '_best.pt')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
