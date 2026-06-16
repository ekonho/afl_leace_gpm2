"""Stage 3: Dual-Orthogonal Projection Fine-Tuning.

Implements the core fine-tuning loop with two orthogonal constraints:

    1. Forward constraint (LEACE / toxic projection):
       At the backbone output, apply h_clean = (I - U U^T) h
       to block drift information from flowing into the classification head.

    2. Backward constraint (GPM / memory projection):
       After loss.backward(), before optimizer.step(), project out memory
       directions from parameter gradients:
           grad_new = grad - grad @ M @ M^T
       This matches GPM's original train_projected() approach.

Hook Strategy:
    - Forward hook on backbone module (outputs [B, D] after avg_pool):
        Applies toxic projection on the feature vector before the head.
    - Gradient projection after backward, before step:
        Iterates over target layer parameters and projects out memory subspace.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from dualortho.utils.rank_utils import project_out


class DualOrthoTrainer:
    """Stage 3 trainer with forward (toxic) and backward (memory) projection.

    Usage:
        trainer = DualOrthoTrainer(
            model=model,
            memory_bases={'backbone.layer4': M_basis},  # from Stage 0
            toxic_basis=U_drift,                         # from Stage 1-2
            backbone_name='backbone',
            grad_proj_layers=['backbone.layer4'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        metrics = trainer.train_one_epoch(dataloader)
        trainer.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        memory_bases: Dict[str, Tensor],       # layer_name -> [D, k] memory basis
        toxic_basis: Tensor,                    # [D, k'] drift directions
        backbone_name: str = 'backbone',        # module whose output is [B, D]
        grad_proj_layers: Optional[List[str]] = None,  # layers to protect gradients
        optimizer: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Move bases to device
        self.memory_bases = {k: v.to(device).float() for k, v in memory_bases.items()}
        self.toxic_basis = toxic_basis.to(device).float()
        self.backbone_name = backbone_name
        self.grad_proj_layers = grad_proj_layers or list(memory_bases.keys())

        self._fwd_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._registered = False

    def _find_module(self, name: str) -> nn.Module:
        modules = dict(self.model.named_modules())
        if name not in modules:
            raise ValueError(
                f"Layer '{name}' not found. Available: {list(modules.keys())}"
            )
        return modules[name]

    def _register_forward_hook(self) -> None:
        """Register forward hook on backbone to apply toxic projection.

        The backbone outputs [B, D] (e.g., [B, 256] for ResNet after avg_pool).
        The hook applies: h_clean = h - h @ U_drift @ U_drift^T
        """
        backbone = self._find_module(self.backbone_name)
        U_drift = self.toxic_basis  # [D, k']

        def toxic_forward_hook(module, input, output):
            # output is [B, D] from backbone
            h = output
            # Apply toxic projection: h_clean = h - h @ U_drift @ U_drift^T
            h_clean = project_out(h, U_drift)
            return h_clean

        self._fwd_handle = backbone.register_forward_hook(toxic_forward_hook)

    def _project_memory_gradients(self) -> None:
        """Project out memory subspace from parameter gradients.

        Called after loss.backward(), before optimizer.step().
        For each target layer's parameters, applies:
            grad_new = grad - grad @ M @ M^T

        This matches GPM's original gradient projection approach.
        """
        for layer_name in self.grad_proj_layers:
            M_basis = self.memory_bases.get(layer_name)
            if M_basis is None:
                continue

            layer = self._find_module(layer_name)
            for name, param in layer.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.data
                # Only project if the last dimension matches memory basis
                if grad.dim() >= 1 and grad.shape[-1] == M_basis.shape[0]:
                    param.grad.data = project_out(grad.float(), M_basis).to(grad.dtype)

    def _ensure_hooks(self) -> None:
        if not self._registered:
            self._register_forward_hook()
            self._registered = True

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        self._registered = False

    def train_one_epoch(
        self,
        dataloader: Iterable,
        *,
        max_batches: Optional[int] = None,
        grad_clip: float = 5.0,
    ) -> dict:
        """Run one epoch of dual-orthogonal projected fine-tuning.

        Flow per batch:
            1. Forward (toxic hook intercepts backbone output)
            2. Compute loss
            3. loss.backward()
            4. Project memory directions out of gradients (GPM)
            5. optimizer.step()
        """
        self._ensure_hooks()
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward -- toxic projection applied via hook on backbone
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward
            loss.backward()

            # GPM memory projection: project out memory directions from gradients
            self._project_memory_gradients()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += float(loss) * batch_size
            total_correct += int((logits.argmax(1) == labels).sum())
            total_samples += batch_size

            if max_batches is not None and (i + 1) >= max_batches:
                break

        return {
            "loss": total_loss / max(1, total_samples),
            "acc": total_correct / max(1, total_samples),
            "num_samples": total_samples,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Iterable,
        *,
        max_batches: Optional[int] = None,
    ) -> dict:
        """Evaluate with toxic projection hook active."""
        self._ensure_hooks()
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += float(loss) * batch_size
            total_correct += int((logits.argmax(1) == labels).sum())
            total_samples += batch_size

            if max_batches is not None and (i + 1) >= max_batches:
                break

        return {
            "loss": total_loss / max(1, total_samples),
            "acc": total_correct / max(1, total_samples),
            "num_samples": total_samples,
        }
