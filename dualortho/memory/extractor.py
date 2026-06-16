"""Stage 0: Extract core memory subspaces from a frozen global model (GPM).

Workflow:
    1. Register forward hooks on target layers (e.g., layer4)
    2. Run a few batches of local data through the frozen backbone
    3. Collect activation matrices R^l per layer
    4. Compute truncated SVD on R^l -> memory basis M^l = U[:, :k]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn

from dualortho.utils.rank_utils import svd_truncate


@dataclass(frozen=True)
class LayerMemory:
    """Memory basis for a single layer, extracted via GPM-style SVD."""
    layer_name: str
    basis: Tensor            # [D, k] orthonormal columns (the memory subspace)
    rank: int                # k: number of retained singular vectors
    explained_variance: float  # fraction of activation variance captured


class MemoryExtractor:
    """Stage 0 — Extract core memory basis matrices M^l from a frozen backbone.

    This implements the GPM (Gradient Projection Memory) idea:
        - Collect activations R^l from key layers during a forward pass
        - Perform SVD on R^l to extract the principal directions
        - These directions represent the model's 'core knowledge' that
          should be protected during local fine-tuning

    Usage:
        extractor = MemoryExtractor(['layer3', 'layer4'], variance_threshold=0.97)
        memories = extractor.extract(frozen_model, local_dataloader, device='cuda')
        # memories['layer4'].basis -> [512, k] tensor
    """

    def __init__(
        self,
        target_layers: Sequence[str],
        *,
        variance_threshold: float = 0.97,
        max_rank: Optional[int] = None,
    ):
        """
        Args:
            target_layers: Names of layers whose output activations to collect.
                           E.g., ['layer4'] for ResNet.
            variance_threshold: SVD variance threshold (0, 1]. Higher = more
                                components retained = larger memory basis.
            max_rank: Hard cap on memory basis rank. None = no limit.
        """
        self.target_layers = list(target_layers)
        self.variance_threshold = variance_threshold
        self.max_rank = max_rank

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[str, List[Tensor]] = {}

    def _make_hook(self, layer_name: str):
        """Create a forward hook that captures the output tensor."""
        def hook_fn(module: nn.Module, input, output):
            # output may be a tuple (e.g., from some layer implementations)
            t = output[0] if isinstance(output, tuple) else output
            # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
            if t.dim() == 4:
                B, C, H, W = t.shape
                t = t.permute(0, 2, 3, 1).reshape(-1, C)
            elif t.dim() == 3:
                B, N, D = t.shape
                t = t.reshape(-1, D)
            # dim==2: already [B, D], no reshape needed
            self._activations[layer_name].append(t.detach())
        return hook_fn

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on all target layers."""
        named_modules = dict(model.named_modules())
        for name in self.target_layers:
            if name not in named_modules:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(named_modules.keys())}"
                )
            mod = named_modules[name]
            self._hooks.append(mod.register_forward_hook(self._make_hook(name)))
            self._activations[name] = []

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(
        self,
        model: nn.Module,
        dataloader: Iterable,
        device: torch.device,
        *,
        max_batches: int = 20,
    ) -> Dict[str, LayerMemory]:
        """Extract memory bases from a frozen model.

        Args:
            model: Frozen global backbone (model.eval() + no grad).
            dataloader: Yields (images, ...) or (images, labels, ...).
            device: Device for forward pass.
            max_batches: How many batches to use for activation collection.

        Returns:
            Dict mapping layer_name -> LayerMemory with the SVD basis.
        """
        was_training = model.training
        model.eval()

        self._register_hooks(model)
        self._activations = {name: [] for name in self.target_layers}

        collected = 0
        for batch in dataloader:
            images = batch[0].to(device)
            model(images)
            collected += 1
            if collected >= max_batches:
                break

        self._remove_hooks()

        # Compute SVD basis for each layer
        memories: Dict[str, LayerMemory] = {}
        for name, act_list in self._activations.items():
            if not act_list:
                raise RuntimeError(f"No activations collected for layer '{name}'")

            R = torch.cat(act_list, dim=0).float()  # [N_total, D]
            # GPM convention: SVD on R^T [D, N] to get feature-space basis U [D, k]
            result = svd_truncate(
                R.T,
                variance_threshold=self.variance_threshold,
                max_rank=self.max_rank,
            )
            explained = float(result.S.pow(2).sum().item() / (R.norm().pow(2).item() + 1e-12))
            memories[name] = LayerMemory(
                layer_name=name,
                basis=result.U.contiguous(),  # [D, k]
                rank=result.rank,
                explained_variance=explained,
            )

        if was_training:
            model.train()

        return memories
