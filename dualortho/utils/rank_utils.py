"""SVD truncation and orthogonal projection primitives for DualOrtho.

Provides:
    - SvdResult: lightweight container for truncated SVD output
    - svd_truncate(): truncated SVD with variance-threshold / max-rank control
    - project_out(): efficiently project a vector out of a low-rank subspace
    - build_projector_matrix(): construct full (I - UU^T) when needed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class SvdResult:
    """Container for truncated SVD decomposition."""
    U: Tensor      # [m, k] left singular vectors (orthonormal columns)
    S: Tensor      # [k] singular values (descending)
    Vh: Tensor     # [k, n] right singular vectors (orthonormal rows)
    rank: int      # number of retained components
    total_energy: float  # fraction of variance explained


def svd_truncate(
    A: Tensor,
    *,
    variance_threshold: float = 0.95,
    max_rank: Optional[int] = None,
    min_rank: int = 1,
) -> SvdResult:
    """Compute truncated SVD of matrix A with automatic rank selection.

    Retains the smallest number of components k such that:
        sum(S[:k]^2) / sum(S^2) >= variance_threshold
    subject to min_rank <= k <= max_rank.

    Args:
        A: Input matrix [m, n].
        variance_threshold: Fraction of total variance to retain (0, 1].
        max_rank: Hard upper bound on k. None = no limit.
        min_rank: Hard lower bound on k (default 1).

    Returns:
        SvdResult with U[:, :k], S[:k], Vh[:k, :].
    """
    # Full economy SVD — returns min(m, n) singular values
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    total_energy = float(S.pow(2).sum().item())
    if total_energy == 0:
        k = min_rank
        return SvdResult(U=U[:, :k], S=S[:k], Vh=Vh[:k, :], rank=k, total_energy=0.0)

    # Cumulative variance explained
    cumulative = S.pow(2).cumsum(dim=0) / total_energy

    # Find smallest k meeting threshold
    indices = torch.where(cumulative >= variance_threshold)[0]
    if len(indices) == 0:
        k = S.shape[0]
    else:
        k = int(indices[0].item()) + 1  # +1 because cumsum is 0-indexed

    # Clamp to [min_rank, max_rank]
    k = max(k, min_rank)
    if max_rank is not None:
        k = min(k, max_rank)
    k = min(k, S.shape[0])  # cannot exceed available singular values

    retained_energy = float(S[:k].pow(2).sum().item() / total_energy)
    return SvdResult(
        U=U[:, :k].contiguous(),
        S=S[:k].contiguous(),
        Vh=Vh[:k, :].contiguous(),
        rank=k,
        total_energy=retained_energy,
    )


def project_out(x: Tensor, basis: Tensor) -> Tensor:
    """Project x OUT of the subspace spanned by columns of basis.

    Computes: x_clean = x - (x @ U) @ U^T  where U = basis

    This is the core operation for both:
        - Forward: h_clean = h - h @ U_drift @ U_drift^T  (toxic projection)
        - Backward: g_clean = g - g @ M @ M^T  (memory protection)

    Args:
        x: Input tensor [..., D]. Arbitrary leading batch dims; projection
           is applied along the last dimension.
        basis: Orthonormal basis [D, k] where k << D.

    Returns:
        x with the subspace component removed, same shape as x.
    """
    # x @ U -> [..., k],  then (... @ U^T) -> [..., D]
    return x - (x @ basis) @ basis.T


def build_projector_matrix(basis: Tensor) -> Tensor:
    """Build the full projection matrix P = I - U @ U^T.

    Only use this when you need the explicit D x D matrix (e.g., for
    debugging or when D is small). For production, prefer project_out().

    Args:
        basis: Orthonormal basis [D, k].

    Returns:
        Symmetric projection matrix [D, D].
    """
    D = basis.shape[0]
    eye = torch.eye(D, device=basis.device, dtype=basis.dtype)
    P = eye - basis @ basis.T
    # Enforce exact symmetry (guards against floating-point drift)
    return 0.5 * (P + P.T)
