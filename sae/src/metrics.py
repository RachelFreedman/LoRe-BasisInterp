"""SAE reconstruction and LoRe preservation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    residual_var = torch.var(x - x_hat)
    total_var = torch.var(x)
    return 1.0 - residual_var / total_var.clamp_min(1e-12)


def active_feature_counts(z: torch.Tensor) -> torch.Tensor:
    return (z != 0).sum(dim=-1)


def gini_coefficient(values: torch.Tensor) -> float:
    """Gini coefficient of a non-negative 1D distribution.

    0 = perfectly equal usage across features.
    1 = one feature owns all mass (maximum inequality).
    """
    x = values.detach().float().flatten()
    x = torch.clamp(x, min=0.0)
    if x.numel() == 0:
        return float("nan")
    total = float(x.sum().item())
    if total <= 0.0:
        return 0.0
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    index = torch.arange(1, n + 1, device=x_sorted.device, dtype=x_sorted.dtype)
    # Standard sorted Gini: (2 * sum(i * x_i)) / (n * sum(x)) - (n + 1) / n
    gini = (2.0 * (index * x_sorted).sum() / (n * x_sorted.sum())) - (n + 1.0) / n
    return float(gini.clamp(0.0, 1.0).item())


def pearson_corr_by_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=0)
    denominator = torch.linalg.norm(x_centered, dim=0) * torch.linalg.norm(y_centered, dim=0)
    return numerator / denominator.clamp_min(1e-12)


def rank_columns(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(x.shape[0], device=x.device, dtype=torch.float32).unsqueeze(1).expand_as(ranks)
    ranks.scatter_(0, order, base)
    return ranks


def spearman_corr_by_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pearson_corr_by_column(rank_columns(x), rank_columns(y))
