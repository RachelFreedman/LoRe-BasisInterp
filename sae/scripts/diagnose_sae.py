#!/usr/bin/env python3
"""Compute SAE activation and decoder diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAE_DIR.parent
sys.path.append(str(REPO_ROOT))

from sae.src.io import ensure_dir, write_json  # noqa: E402
from sae.src.metrics import gini_coefficient  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    parser.add_argument("--data-dir", default="sae/data")
    parser.add_argument("--results-dir", default="sae/results/d3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_split_stats(
    model: TopKSAE,
    x: torch.Tensor,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    activation_counts = torch.zeros(model.dict_size, dtype=torch.long)
    activation_sums = torch.zeros(model.dict_size, dtype=torch.float64)
    max_activation = torch.zeros(model.dict_size, dtype=torch.float32)
    active_per_example_sum = 0
    n_examples = 0

    loader = DataLoader(TensorDataset(x), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z = model.encode(batch).cpu()
            active = z != 0
            activation_counts += active.sum(dim=0)
            activation_sums += z.double().sum(dim=0)
            max_activation = torch.maximum(max_activation, z.max(dim=0).values)
            active_per_example_sum += int(active.sum().item())
            n_examples += z.shape[0]

    live_mask = activation_counts > 0
    dead_features = int((~live_mask).sum().item())
    freq = activation_counts.float() / max(1, n_examples)
    gini_all = gini_coefficient(freq)
    gini_live = gini_coefficient(freq[live_mask]) if bool(live_mask.any()) else float("nan")
    stats = {
        "split": split,
        "n_examples": n_examples,
        "dict_size": model.dict_size,
        "k": model.k,
        "sparsity_mode": model.sparsity_mode,
        "average_active_features": active_per_example_sum / max(1, n_examples),
        "live_features": int(live_mask.sum().item()),
        "dead_features": dead_features,
        "dead_feature_rate": dead_features / model.dict_size,
        "max_activation_count": int(activation_counts.max().item()),
        "median_activation_count": float(torch.median(activation_counts.float()).item()),
        "gini_activation_frequency_all": gini_all,
        "gini_activation_frequency_live": gini_live,
    }

    rows = []
    for rank, feature_idx in enumerate(torch.argsort(activation_counts, descending=True).tolist()):
        count = int(activation_counts[feature_idx].item())
        mean_when_active = float((activation_sums[feature_idx] / count).item()) if count > 0 else 0.0
        rows.append(
            {
                "split": split,
                "rank": rank,
                "feature": feature_idx,
                "activation_count": count,
                "activation_frequency": count / max(1, n_examples),
                "mean_activation_when_active": mean_when_active,
                "max_activation": float(max_activation[feature_idx].item()),
            }
        )
    return stats, rows


def main() -> int:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    train_cfg = config.get("training", {})
    model = TopKSAE(
        input_dim=int(config["input_dim"]),
        dict_size=int(config["dict_size"]),
        k=int(config["k"]),
        normalize_decoder=bool(train_cfg.get("normalize_decoder", True)),
        aux_k=int(train_cfg.get("aux_k", config["k"])),
        sparsity_mode=str(train_cfg.get("sparsity_mode", "topk")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    all_stats = []
    all_top_rows = []
    for split in ("train", "val", "test"):
        x = torch.load(Path(args.data_dir) / f"sae_{split}.pt", map_location="cpu").float()
        stats, rows = collect_split_stats(model, x, split, args.batch_size, device)
        all_stats.append(stats)
        all_top_rows.extend(rows[: args.top_n])

    decoder_weight = model.decoder.weight.detach().cpu()
    decoder_norms = torch.linalg.norm(decoder_weight, dim=0)
    decoder_stats = {
        "decoder_norm_mean": float(decoder_norms.mean().item()),
        "decoder_norm_std": float(decoder_norms.std(unbiased=False).item()),
        "decoder_norm_min": float(decoder_norms.min().item()),
        "decoder_norm_max": float(decoder_norms.max().item()),
    }

    write_json(results_dir / "sae_diagnostics_summary.json", {"splits": all_stats, **decoder_stats})
    write_rows(
        results_dir / "top_active_features.csv",
        [
            "split",
            "rank",
            "feature",
            "activation_count",
            "activation_frequency",
            "mean_activation_when_active",
            "max_activation",
        ],
        all_top_rows,
    )
    print({"splits": all_stats, **decoder_stats})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
