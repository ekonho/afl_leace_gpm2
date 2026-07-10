"""Client-level local training adapters.

Dispatches to the appropriate local training method based on `args.alg`.
Currently only supports "dualortho" — other algorithms can be added by
extending the `local_train_net()` dispatcher.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from dualortho.pipeline import DualOrthoPipeline

logger = logging.getLogger(__name__)


def train_client_dualortho(
    cid: int,
    net: nn.Module,
    global_model: nn.Module,
    train_dl,
    test_dl,
    args,
    device: torch.device,
    *,
    logger: logging.Logger = None,
) -> Tuple[nn.Module, Dict]:
    """Train one client with DualOrtho 4-stage pipeline.

    Returns:
        (updated_model, client_detail): Trained model and metrics dict.
    """
    if logger is None:
        logger = logging.getLogger()

    local_model = copy.deepcopy(global_model).to(device)
    logger.info(f"  Client {cid} ({len(train_dl.dataset)} samples):")

    # ---- Pipeline setup ----
    memory_layers = [f"backbone.{l}" for l in args.memory_layers.split(",")]
    pipe = DualOrthoPipeline(
        args,
        local_model,
        device,
        memory_layers=memory_layers,
        backbone_name="backbone",
        grad_proj_layers=memory_layers,
    )

    # Extract global head weight for drift detection
    global_W = global_model.head.fc.weight.data.T.clone()

    # ---- Stages 0-2 ----
    pipe.prepare(
        train_dl,
        global_W.to(device),
        memory_max_batches=args.memory_max_batches,
    )

    memory_ranks = {name: mem.rank for name, mem in pipe.memories.items()}
    memory_energy = {name: round(mem.explained_variance, 4) for name, mem in pipe.memories.items()}
    drift_rank = pipe.drift.rank if pipe.drift else 0
    drift_energy = round(pipe.drift.explained_variance, 4) if pipe.drift else 0.0

    logger.info(f"    Stage 0 memory: ranks={memory_ranks}, energy={memory_energy}")
    logger.info(f"    Stage 1-2 drift: rank={drift_rank}, energy={drift_energy}")

    # ---- Stage 3: Fine-tuning ----
    optimizer = optim.SGD(
        local_model.parameters(), lr=args.local_lr,
        momentum=args.local_momentum, weight_decay=args.local_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    pipe.set_optimizer(optimizer)
    pipe.set_criterion(criterion)

    epoch_metrics = []
    for ep in range(args.local_epochs):
        metrics = pipe.train_one_epoch(train_dl, grad_clip=args.grad_clip)
        epoch_metrics.append({"loss": round(metrics["loss"], 4), "acc": round(metrics["acc"], 4)})
        logger.info(
            f"    Epoch {ep+1}/{args.local_epochs}: "
            f"loss={metrics['loss']:.4f}, acc={metrics['acc']*100:.1f}%"
        )

    pipe.cleanup()

    client_detail = {
        "client_id": cid,
        "n_samples": len(train_dl.dataset),
        "memory_ranks": memory_ranks,
        "memory_energy": memory_energy,
        "drift_rank": drift_rank,
        "drift_energy": drift_energy,
        "epoch_metrics": epoch_metrics,
        "final_loss": epoch_metrics[-1]["loss"],
        "final_acc": epoch_metrics[-1]["acc"],
    }
    return local_model, client_detail


def local_train_net(
    nets_this_round: Dict[int, nn.Module],
    args,
    global_model: nn.Module,
    train_dls: List,
    net_dataidx_map: Dict[int, List[int]] = None,
    device: torch.device = None,
    *,
    logger: logging.Logger = None,
) -> Tuple[List[Dict], List[int]]:
    """Dispatch local training to the appropriate algorithm.

    Args:
        nets_this_round: {client_id: model} — models will be updated in-place.
        args: Full argument namespace.
        global_model: Global model (read-only reference).
        train_dls: List of per-client DataLoaders, indexed by client_id.
        device: Torch device.
        logger: Logger instance.

    Returns:
        (client_details, client_sample_counts)
    """
    client_details = []
    client_sample_counts = []

    for cid, net in nets_this_round.items():
        client_start = time.time()
        train_dl = train_dls[cid]

        if args.alg == "dualortho":
            local_model, detail = train_client_dualortho(
                cid, net, global_model, train_dl, None, args, device, logger=logger
            )
        else:
            raise ValueError(f"Unknown algorithm: {args.alg}")

        detail["time_sec"] = round(time.time() - client_start, 1)
        client_details.append(detail)
        client_sample_counts.append(len(train_dl.dataset))

        # Copy weights back to the net dict
        nets_this_round[cid].load_state_dict(local_model.state_dict())

        logger.info(f"    Done in {detail['time_sec']:.1f}s")
        del local_model
        torch.cuda.empty_cache()

    return client_details, client_sample_counts
