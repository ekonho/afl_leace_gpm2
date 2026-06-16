"""Stage 1-2: Unsupervised drift detection and toxic subspace construction.

Combines AFL's closed-form analytic solution with LEACE-style subspace projection:
    1. Compute local closed-form weight: W_local = (X^T X + lambda*I)^{-1} X^T Y
    2. Compute aggregated global weight: W_agg (from AFL aggregation)
    3. Extract drift directions: SVD(Delta_W) -> U_drift
    4. Build toxic projector: P_toxic = I - U_drift @ U_drift^T
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dualortho.utils.rank_utils import svd_truncate, build_projector_matrix


@dataclass(frozen=True)
class DriftSpace:
    """Result of drift detection: the toxic subspace and its projector."""
    delta_weight: Tensor     # [D_out, D_in] raw weight difference
    u_drift: Tensor          # [D_out, k'] drift directions (orthonormal columns)
    rank: int                # k' number of drift components
    explained_variance: float
    projector: Tensor        # [D_out, D_out] full P_toxic = I - U U^T


class AFLDriftProbe:
    """Stage 1-2 — Detect drift via AFL closed-form and build LEACE-style projector.

    Usage:
        probe = AFLDriftProbe(variance_threshold=0.9, max_rank=32)

        # Stage 1: compute local closed-form head weight
        W_local = probe.solve_closed_form_weight(features, one_hot_labels)

        # Stage 1: compute aggregated global head weight (from multiple clients)
        W_agg = probe.aggregate_weights(local_weights, local_sample_counts)

        # Stage 2: detect drift and build toxic projector
        drift = probe.detect_drift(W_local, W_agg)
        # drift.projector -> [D_out, D_out] matrix P_toxic
        # drift.u_drift   -> [D_out, k'] drift directions
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-2,
        variance_threshold: float = 0.9,
        max_rank: Optional[int] = 32,
    ):
        """
        Args:
            regularization: Ridge regression lambda for closed-form solve.
            variance_threshold: Fraction of drift variance to retain in U_drift.
            max_rank: Hard cap on drift subspace rank.
        """
        self.regularization = regularization
        self.variance_threshold = variance_threshold
        self.max_rank = max_rank

    def solve_closed_form_weight(
        self,
        features: Tensor,
        one_hot_labels: Tensor,
        *,
        reg: Optional[float] = None,
    ) -> Tensor:
        """AFL analytic solution: W = (X^T X + lambda I)^{-1} X^T Y.

        Args:
            features: [N, D_in] feature matrix from frozen backbone.
            one_hot_labels: [N, D_out] one-hot label matrix.
            reg: Override regularization. None = use default.

        Returns:
            Weight matrix [D_in, D_out].
        """
        if reg is None:
            reg = self.regularization
        X = features.detach().float()
        Y = one_hot_labels.detach().float()
        XtX = X.T @ X  # [D_in, D_in]
        ridge = XtX + reg * torch.eye(XtX.shape[0], device=X.device, dtype=X.dtype)
        W = torch.linalg.solve(ridge, X.T @ Y)  # [D_in, D_out]
        return W

    @torch.no_grad()
    def aggregate_weights(
        self,
        local_weights: list[Tensor],
        sample_counts: list[int],
    ) -> Tensor:
        """Weighted average of client weights (AFL absolute aggregation law).

        Args:
            local_weights: List of [D_in, D_out] weight matrices.
            sample_counts: Number of samples per client.

        Returns:
            Aggregated weight [D_in, D_out].
        """
        total = sum(sample_counts)
        W_agg = torch.zeros_like(local_weights[0])
        for W_i, n_i in zip(local_weights, sample_counts):
            W_agg += W_i * (n_i / total)
        return W_agg

    @torch.no_grad()
    def detect_drift(
        self,
        w_local: Tensor,
        w_agg: Tensor,
    ) -> DriftSpace:
        """Detect drift between local and aggregated weights, build toxic projector.

        Args:
            w_local: [D_in, D_out] local closed-form weight.
            w_agg: [D_in, D_out] aggregated global weight.

        Returns:
            DriftSpace with drift subspace and projector.
        """
        assert w_local.shape == w_agg.shape, (
            f"Shape mismatch: {w_local.shape} vs {w_agg.shape}"
        )

        delta_w = (w_local - w_agg).float()  # [D_in, D_out]

        # SVD on delta_w to extract drift directions
        # We use the left singular vectors (rows of delta_w = feature space)
        result = svd_truncate(
            delta_w,
            variance_threshold=self.variance_threshold,
            max_rank=self.max_rank,
        )

        U_drift = result.U  # [D_in, k'] — drift directions in feature space

        # Build full projector matrix P_toxic = I - U_drift @ U_drift^T
        projector = build_projector_matrix(U_drift)

        total_var = float(delta_w.norm().pow(2).item())
        explained = float(result.S.pow(2).sum().item() / (total_var + 1e-12))

        return DriftSpace(
            delta_weight=delta_w,
            u_drift=U_drift.contiguous(),
            rank=result.rank,
            explained_variance=explained,
            projector=projector.contiguous(),
        )

    def pseudo_one_hot_from_global_head(
        self,
        backbone: nn.Module,
        head: nn.Module,
        images: Tensor,
    ) -> Tensor:
        """Generate pseudo one-hot labels using the global model's predictions.

        Useful for unsupervised drift detection when true labels are unavailable.

        Args:
            backbone: Frozen feature extractor.
            head: Frozen global classification head.
            images: [B, C, H, W] input images.

        Returns:
            [B, num_classes] one-hot matrix from argmax predictions.
        """
        with torch.no_grad():
            feats = backbone(images)
            if feats.dim() > 2:
                feats = feats.flatten(1)
            logits = head(feats)
            labels = logits.argmax(dim=-1)
            one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
        return one_hot
