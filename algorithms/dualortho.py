"""DualOrtho federated learning algorithm.

Full FL loop: select clients → DualOrtho 4-stage pipeline → FedAvg → evaluate.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from algorithms.client import local_train_net
from dualortho.models.resnet_cifar import resnet18_cifar, resnet34_cifar

logger = logging.getLogger(__name__)


def build_model(args) -> nn.Module:
    """Build ResNet backbone + head."""
    if args.model == "resnet18":
        model = resnet18_cifar(args.num_classes, nf=args.nf)
    elif args.model == "resnet34":
        model = resnet34_cifar(args.num_classes, nf=args.nf)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model


def federated_averaging(global_model: nn.Module, client_states: List[Dict], client_counts: List[int]):
    """Weighted FedAvg aggregation."""
    global_state = global_model.state_dict()
    device = next(global_model.parameters()).device
    total = sum(client_counts)
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        for state, count in zip(client_states, client_counts):
            global_state[key] += state[key].float().to(device) * (count / total)
    global_model.load_state_dict(global_state)


def evaluate(global_model: nn.Module, test_loader, device: torch.device, criterion: nn.Module):
    """Evaluate on test set."""
    global_model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = global_model(images)
            loss = criterion(outputs, labels_batch)
            test_loss += float(loss) * images.size(0)
            correct += int((outputs.argmax(1) == labels_batch).sum())
            total += images.size(0)
    acc = correct / total if total > 0 else 0.0
    return test_loss / total if total > 0 else 0.0, acc


def dualortho_alg(
    args,
    n_rounds: int,
    nets: Dict[int, nn.Module],
    global_model: nn.Module,
    party_list_rounds: List[List[int]],
    net_dataidx_map: Dict[int, List[int]],
    train_local_dls: List,
    test_dl,
    traindata_cls_counts: np.ndarray,
    device: torch.device,
    logger: logging.Logger = None,
):
    """Run the full DualOrtho federated learning loop.

    Args:
        args: Argument namespace.
        n_rounds: Number of communication rounds.
        nets: Dict of client models (updated in-place).
        global_model: Global model (updated in-place).
        party_list_rounds: per-round list of selected client IDs.
        net_dataidx_map: {client_id: list of data indices}.
        train_local_dls: List of per-client DataLoaders.
        test_dl: Global test DataLoader.
        traindata_cls_counts: Per-client class counts.
        device: Torch device.
        logger: Logger instance.

    Returns:
        (record_test_acc_list, best_test_acc)
    """
    if logger is None:
        logger = logging.getLogger()

    record_test_acc_list = []
    best_test_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    # JSON log
    run_record = {"run_name": args.run_name if hasattr(args, 'run_name') else "dualortho",
                  "args": vars(args), "data": {}, "model": {}, "rounds": []}
    json_log_path = os.path.join(args.log_dir, "result.json")

    logger.info("=" * 70)
    logger.info(f"Algorithm: {args.alg}")
    logger.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logger.info(f"Rounds: {n_rounds}, Clients: {args.num_clients}")
    logger.info("=" * 70)

    for rnd in range(n_rounds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Round {rnd+1}/{n_rounds}")
        logger.info(f"{'='*50}")

        party_list_this_round = party_list_rounds[rnd]
        logger.info(f"Selected clients: {party_list_this_round} ({len(party_list_this_round)}/{args.num_clients})")

        # Broadcast global model to clients
        global_w = global_model.state_dict()
        nets_this_round = {cid: nets[cid] for cid in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        round_start = time.time()

        # Local training with DualOrtho
        client_details, client_sample_counts = local_train_net(
            nets_this_round, args, global_model, train_local_dls,
            device=device, logger=logger,
        )

        # FedAvg aggregation
        logger.info("  Aggregating...")
        client_states = [nets_this_round[cid].state_dict() for cid in party_list_this_round]
        federated_averaging(global_model, client_states, client_sample_counts)

        # Evaluate
        avg_loss, acc = evaluate(global_model, test_dl, device, criterion)
        round_time = time.time() - round_start

        if acc > best_test_acc:
            best_test_acc = acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                global_model.state_dict(),
                os.path.join(args.save_dir, f"{args.alg}_best.pt"),
            )

        record_test_acc_list.append(acc)
        logger.info(f"  Global test: loss={avg_loss:.4f}, acc={acc*100:.2f}%")
        logger.info(f"  Best so far: {best_test_acc*100:.2f}%")
        logger.info(f"  Round time: {round_time:.1f}s")

        # Append to JSON
        run_record["rounds"].append({
            "round": rnd + 1,
            "test_loss": round(avg_loss, 4),
            "test_acc": round(acc, 4),
            "best_acc": round(best_test_acc, 4),
            "time_sec": round(round_time, 1),
            "selected_clients": party_list_this_round,
            "client_details": [
                {k: v for k, v in cd.items() if k not in ("epoch_metrics",)}
                for cd in client_details
            ],
        })

    # Final
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"  Best test accuracy: {best_test_acc*100:.2f}%")
    logger.info("=" * 70)

    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON log: {json_log_path}")

    return record_test_acc_list, best_test_acc
