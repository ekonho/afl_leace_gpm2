"""DualOrtho Pipeline — Orchestrates the 4-stage dual-orthogonal fine-tuning.

This is the top-level API that ties together:
    Stage 0: GPM memory extraction (extractor.py)
    Stage 1: AFL closed-form drift detection (afl_drift_probe.py)
    Stage 2: Toxic subspace construction (afl_drift_probe.py)
    Stage 3: Dual-orthogonal fine-tuning (dualortho_trainer.py)

Usage:
    pipe = DualOrthoPipeline(args, model, device)

    # Stages 0-2: prepare constraints (frozen model)
    pipe.prepare(client_loader, global_weight)

    # Stage 3: constrained fine-tuning
    for epoch in range(args.local_epochs):
        metrics = pipe.train_one_epoch(client_loader)

    # Clean up hooks
    pipe.cleanup()
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from dualortho.memory.extractor import MemoryExtractor, LayerMemory
from dualortho.probe.afl_drift_probe import AFLDriftProbe, DriftSpace
from dualortho.training.dualortho_trainer import DualOrthoTrainer

logger = logging.getLogger(__name__)


class DualOrthoPipeline:
    """Top-level orchestrator for the 4-stage dual-orthogonal fine-tuning.

    Architecture assumptions:
        model.backbone -> [B, D] features (e.g., ResNet after avg_pool)
        model.head     -> [B, num_classes] logits

    Stage 0 collects activations from backbone.layer4 (spatial [B,C,H,W]),
    flattens to [N, C], and computes SVD -> memory basis [C, k].

    Stage 1-2 computes drift in the backbone output space [D, k'].

    Stage 3:
        - Forward hook on backbone: applies toxic projection on [B, D] output
        - Gradient projection on backbone.layer4: protects memory directions
    """

    def __init__(
        self,
        args,
        model: nn.Module,
        device: torch.device,
        *,
        memory_layers: Optional[Sequence[str]] = None,
        backbone_name: str = 'backbone',
        grad_proj_layers: Optional[List[str]] = None,
    ):
        self.args = args
        self.model = model
        self.device = device

        self.memory_layers = memory_layers or ['backbone.layer4']
        self.backbone_name = backbone_name
        self.grad_proj_layers = grad_proj_layers or self.memory_layers

        # Stage 0: Memory extractor
        self.extractor = MemoryExtractor(
            target_layers=self.memory_layers,
            variance_threshold=args.memory_var_th,
            max_rank=args.memory_max_rank,
        )

        # Stage 1-2: Drift probe
        self.probe = AFLDriftProbe(
            regularization=args.regularization,
            variance_threshold=args.drift_var_th,
            max_rank=args.drift_max_rank,
        )

        # Stage 3: Trainer (initialized in prepare())
        self.trainer: Optional[DualOrthoTrainer] = None

        # Intermediate results
        self.memories: Dict[str, LayerMemory] = {}
        self.drift: Optional[DriftSpace] = None

    def prepare(
        self,
        client_loader,
        global_head_weight: Tensor,
        *,
        memory_max_batches: int = 20,
    ) -> None:
        """Run Stages 0-2: extract memory and detect drift (frozen model).

        Args:
            client_loader: Local client data loader.
            global_head_weight: [D_in, D_out] global classification head weight.
            memory_max_batches: Max batches for activation collection.
        """
        logger.info("=" * 50)
        logger.info("Stage 0: Extracting GPM memory bases...")
        self.memories = self.extractor.extract(
            self.model, client_loader, self.device,
            max_batches=memory_max_batches,
        )
        for name, mem in self.memories.items():
            logger.info(
                "  Layer '%s': rank=%d, explained_var=%.4f",
                name, mem.rank, mem.explained_variance,
            )

        logger.info("Stage 1: Computing local closed-form weight via AFL...")
        W_local = self._compute_local_weight(client_loader, max_batches=memory_max_batches)

        logger.info("Stage 2: Detecting drift and building toxic projector...")
        self.drift = self.probe.detect_drift(W_local, global_head_weight)
        logger.info(
            "  Drift rank=%d, explained_var=%.4f",
            self.drift.rank, self.drift.explained_variance,
        )

        # Build trainer
        logger.info("Building DualOrtho Trainer...")
        self.trainer = DualOrthoTrainer(
            model=self.model,
            memory_bases={name: mem.basis for name, mem in self.memories.items()},
            toxic_basis=self.drift.u_drift,
            backbone_name=self.backbone_name,
            grad_proj_layers=self.grad_proj_layers,
            device=self.device,
        )
        logger.info("Pipeline ready for Stage 3 fine-tuning.")
        logger.info("=" * 50)

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self.trainer is not None:
            self.trainer.optimizer = optimizer

    def set_criterion(self, criterion: nn.Module) -> None:
        if self.trainer is not None:
            self.trainer.criterion = criterion

    def train_one_epoch(self, client_loader, **kwargs) -> dict:
        if self.trainer is None:
            raise RuntimeError("Pipeline not initialized. Call prepare() first.")
        return self.trainer.train_one_epoch(client_loader, **kwargs)

    def evaluate(self, test_loader, **kwargs) -> dict:
        if self.trainer is None:
            raise RuntimeError("Pipeline not initialized. Call prepare() first.")
        return self.trainer.evaluate(test_loader, **kwargs)

    def cleanup(self) -> None:
        if self.trainer is not None:
            self.trainer.remove_hooks()

    def _compute_local_weight(self, dataloader, max_batches: int = 20) -> Tensor:
        """Compute local closed-form weight using AFL analytic solution."""
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for i, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Get features from backbone
                feats = self.model.backbone(images)
                if feats.dim() > 2:
                    feats = feats.flatten(1)

                features_list.append(feats)
                num_classes = self.args.num_classes
                one_hot = torch.nn.functional.one_hot(
                    targets, num_classes=num_classes
                ).float()
                labels_list.append(one_hot)

                if i + 1 >= max_batches:
                    break

        X = torch.cat(features_list, dim=0)
        Y = torch.cat(labels_list, dim=0)
        W_local = self.probe.solve_closed_form_weight(X, Y)
        return W_local
