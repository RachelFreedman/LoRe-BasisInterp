"""Numeric LoRe basis–feature attribution helpers (observational, not causal)."""

from __future__ import annotations

import torch


def decoder_basis_alignment(
    decoder_weight: torch.Tensor,
    basis_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and cosine alignment matrices of shape [dict_size, n_bases].

    decoder_weight: [input_dim, dict_size] (nn.Linear weight)
    basis_v: [input_dim, n_bases]
    """
    alignment = decoder_weight.T @ basis_v
    v_norms = basis_v.norm(dim=0).clamp_min(1e-8)
    cosine = decoder_weight.T @ (basis_v / v_norms)
    return alignment, cosine


def operational_kept_mask(user_w: torch.Tensor, threshold: float = 1e-2) -> torch.Tensor:
    """Bases kept for personalized LoRe scores: max user weight >= threshold."""
    return user_w.max(dim=0).values >= threshold


def feature_activation_stats(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """mean |z_i| and activation frequency over examples. z: [n, dict]."""
    mean_abs = z.abs().mean(dim=0)
    freq = (z.abs() > 0).float().mean(dim=0)
    return mean_abs, freq


def mean_abs_contribution(
    mean_abs_activation: torch.Tensor,
    alignment: torch.Tensor,
) -> torch.Tensor:
    """contribution_ij = mean|z_i| * |alignment_ij| → [dict, n_bases]."""
    return mean_abs_activation.unsqueeze(1) * alignment.abs()


def top_features_by_contribution(
    alignment: torch.Tensor,
    cosine_alignment: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    activation_frequency: torch.Tensor,
    contribution: torch.Tensor,
    *,
    top_n: int = 50,
    kept_mask: torch.Tensor | None = None,
    max_user_weight: torch.Tensor | None = None,
) -> list[dict]:
    """Top-n positive and negative features per basis, ranked by contribution."""
    dict_size, n_bases = alignment.shape
    if kept_mask is None:
        kept_mask = torch.ones(n_bases, dtype=torch.bool)
    if max_user_weight is None:
        max_user_weight = torch.zeros(n_bases)

    rows: list[dict] = []
    for j in range(n_bases):
        align_j = alignment[:, j]
        cos_j = cosine_alignment[:, j]
        contrib_j = contribution[:, j]
        is_kept = bool(kept_mask[j].item())

        for sign, sign_mask in (
            ("positive", align_j > 0),
            ("negative", align_j < 0),
        ):
            scores = contrib_j.clone()
            scores[~sign_mask] = -1.0
            order = torch.argsort(scores, descending=True)[:top_n]
            for rank, i in enumerate(order.tolist()):
                if sign == "positive" and align_j[i] <= 0:
                    continue
                if sign == "negative" and align_j[i] >= 0:
                    continue
                rows.append(
                    {
                        "basis_id": j,
                        "is_operational_kept": int(is_kept),
                        "max_user_weight": float(max_user_weight[j].item()),
                        "rank": rank,
                        "sign": sign,
                        "feature_id": i,
                        "alignment": float(align_j[i].item()),
                        "cosine_alignment": float(cos_j[i].item()),
                        "mean_abs_activation": float(mean_abs_activation[i].item()),
                        "activation_frequency": float(activation_frequency[i].item()),
                        "mean_abs_contribution": float(contrib_j[i].item()),
                    }
                )
    return rows
