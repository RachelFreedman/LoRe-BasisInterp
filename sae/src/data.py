"""Dataset loading and metadata helpers for SAE training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


EMBEDDING_DIM = 4096


@dataclass(frozen=True)
class SAEArtifactRows:
    embeddings: torch.Tensor
    metadata: list[dict[str, Any]]


def _as_embedding(value: Any, *, label: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu().to(torch.float32)
    if tensor.ndim != 1 or tensor.numel() != EMBEDDING_DIM:
        raise ValueError(f"{label} has shape {tuple(tensor.shape)}, expected ({EMBEDDING_DIM},)")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains NaN or Inf values")
    return tensor


def load_prism_response_embeddings(path: str | Path, *, source_split: str) -> SAEArtifactRows:
    """Load chosen/rejected response embeddings from a Phase 1 PRISM artifact."""
    rows = torch.load(path, map_location="cpu")
    embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for original_index, row in enumerate(rows):
        extra = row.get("extra_info", {})
        if not isinstance(extra, dict):
            raise ValueError(f"{source_split}[{original_index}] missing dict extra_info")

        pair_id = f"{source_split}:{original_index}"
        common = {
            "source_split": extra.get("split", source_split),
            "pair_id": pair_id,
            "user_id": extra.get("user_id"),
            "dialog_id": extra.get("dialog_id"),
            "is_seen_user": extra.get("seen"),
            "original_index": original_index,
        }

        for response_role, key in (
            ("chosen", "chosen_conv_embedding"),
            ("rejected", "rejected_conv_embedding"),
        ):
            label = f"{source_split}[{original_index}].{key}"
            embeddings.append(_as_embedding(extra[key], label=label))
            metadata.append(
                {
                    **common,
                    "embedding_id": len(metadata),
                    "response_role": response_role,
                }
            )

    return SAEArtifactRows(embeddings=torch.stack(embeddings), metadata=metadata)


def split_indices(n_rows: int, train_fraction: float, val_fraction: float, seed: int) -> dict[str, torch.Tensor]:
    if n_rows <= 0:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_rows, generator=generator)
    train_end = int(n_rows * train_fraction)
    val_end = train_end + int(n_rows * val_fraction)
    return {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:],
    }
