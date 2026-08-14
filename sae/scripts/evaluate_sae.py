#!/usr/bin/env python3
"""Evaluate SAE reconstruction quality and LoRe preservation."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAE_DIR.parent
sys.path.append(str(REPO_ROOT))

from sae.src.io import ensure_dir, read_jsonl, write_json  # noqa: E402
from sae.src.metrics import (  # noqa: E402
    active_feature_counts,
    explained_variance,
    gini_coefficient,
    pearson_corr_by_column,
    reconstruction_mse,
    spearman_corr_by_column,
)
from sae.src.topk_sae import TopKSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    parser.add_argument("--data-dir", default="sae/data")
    parser.add_argument("--basis-matrices", default="PRISM/basis_matrices.pt")
    parser.add_argument("--run-key", default="PART2_K10_seed42")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--results-dir", default="sae/results/d3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reconstruct(model: TopKSAE, x: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs_hat = []
    zs = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            xs_hat.append(x_hat.cpu())
            zs.append(z.cpu())
    return torch.cat(xs_hat), torch.cat(zs)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_scores(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    metadata: list[dict],
    basis_v: torch.Tensor,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    by_pair: dict[str, dict[str, int]] = defaultdict(dict)
    for row in metadata:
        if row["sae_split"] == split:
            by_pair[row["pair_id"]][row["response_role"]] = row["sae_split_index"]

    original_scores = []
    reconstructed_scores = []
    for roles in by_pair.values():
        if "chosen" not in roles or "rejected" not in roles:
            continue
        d_pair = x[roles["chosen"]] - x[roles["rejected"]]
        d_pair_hat = x_hat[roles["chosen"]] - x_hat[roles["rejected"]]
        original_scores.append(d_pair @ basis_v)
        reconstructed_scores.append(d_pair_hat @ basis_v)
    return torch.stack(original_scores), torch.stack(reconstructed_scores)


def personalized_lore_accuracy(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    metadata: list[dict],
    basis_v: torch.Tensor,
    user_w: torch.Tensor,
    split: str,
) -> tuple[float, float, int]:
    kept_mask = user_w.max(dim=0).values >= 1e-2
    basis_kept = basis_v[:, kept_mask]
    weights_kept = user_w[:, kept_mask]

    train_seen_users = sorted(
        {
            row["user_id"]
            for row in metadata
            if row["source_split"] == "train" and row["is_seen_user"] is True
        }
    )
    user_to_row = {user_id: idx for idx, user_id in enumerate(train_seen_users)}

    by_pair: dict[str, dict[str, int | str]] = defaultdict(dict)
    for row in metadata:
        if row["sae_split"] == split and row["user_id"] in user_to_row:
            by_pair[row["pair_id"]][row["response_role"]] = row["sae_split_index"]
            by_pair[row["pair_id"]]["user_id"] = row["user_id"]

    original_correct = []
    reconstructed_correct = []
    for roles in by_pair.values():
        if "chosen" not in roles or "rejected" not in roles or "user_id" not in roles:
            continue
        user_idx = user_to_row[str(roles["user_id"])]
        reward_direction = basis_kept @ weights_kept[user_idx]
        d_pair = x[int(roles["chosen"])] - x[int(roles["rejected"])]
        d_pair_hat = x_hat[int(roles["chosen"])] - x_hat[int(roles["rejected"])]
        original_correct.append(float((d_pair @ reward_direction) > 0))
        reconstructed_correct.append(float((d_pair_hat @ reward_direction) > 0))

    if not original_correct:
        return float("nan"), float("nan"), 0
    return (
        float(torch.tensor(original_correct).mean().item()),
        float(torch.tensor(reconstructed_correct).mean().item()),
        len(original_correct),
    )


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
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

    x = torch.load(data_dir / f"sae_{args.split}.pt", map_location="cpu").float()
    x_hat, z = reconstruct(model, x, args.batch_size, device)

    matrices = torch.load(args.basis_matrices, map_location="cpu")
    run_data = matrices[args.run_key]
    basis_v = run_data["V"].float()
    user_w = run_data["W"].float()

    original_scores = x @ basis_v
    reconstructed_scores = x_hat @ basis_v
    pearson = pearson_corr_by_column(original_scores, reconstructed_scores)
    spearman = spearman_corr_by_column(original_scores, reconstructed_scores)
    score_abs_error = torch.mean(torch.abs(original_scores - reconstructed_scores), dim=0)
    score_rel_error = score_abs_error / torch.mean(torch.abs(original_scores), dim=0).clamp_min(1e-12)

    metadata = read_jsonl(data_dir / "metadata.jsonl")
    pair_scores, pair_scores_hat = pairwise_scores(x, x_hat, metadata, basis_v, args.split)
    pair_pearson = pearson_corr_by_column(pair_scores, pair_scores_hat)
    pair_spearman = spearman_corr_by_column(pair_scores, pair_scores_hat)
    original_acc, reconstructed_acc, accuracy_pair_count = personalized_lore_accuracy(
        x,
        x_hat,
        metadata,
        basis_v,
        user_w,
        args.split,
    )

    active_counts = active_feature_counts(z)
    activation_frequency = (z != 0).float().mean(dim=0)
    live_mask = activation_frequency > 0
    gini_all = gini_coefficient(activation_frequency)
    gini_live = (
        gini_coefficient(activation_frequency[live_mask])
        if bool(live_mask.any())
        else float("nan")
    )
    summary = {
        "split": args.split,
        "n_embeddings": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "dict_size": int(config["dict_size"]),
        "k": int(config["k"]),
        "sparsity_mode": str(train_cfg.get("sparsity_mode", "topk")),
        "mse": float(reconstruction_mse(x, x_hat).item()),
        "explained_variance": float(explained_variance(x, x_hat).item()),
        "mean_embedding_norm": float(torch.linalg.norm(x, dim=1).mean().item()),
        "mean_reconstruction_norm": float(torch.linalg.norm(x_hat, dim=1).mean().item()),
        "mean_reconstruction_error_norm": float(torch.linalg.norm(x - x_hat, dim=1).mean().item()),
        "average_active_features": float(active_counts.float().mean().item()),
        "dead_feature_rate": float((activation_frequency == 0).float().mean().item()),
        "live_features": int(live_mask.sum().item()),
        "gini_activation_frequency_all": gini_all,
        "gini_activation_frequency_live": gini_live,
        "mean_basis_score_pearson": float(pearson.mean().item()),
        "min_basis_score_pearson": float(pearson.min().item()),
        "mean_pair_score_pearson": float(pair_pearson.mean().item()),
        "min_pair_score_pearson": float(pair_pearson.min().item()),
        "lore_accuracy_original": original_acc,
        "lore_accuracy_reconstructed": reconstructed_acc,
        "lore_accuracy_drop": original_acc - reconstructed_acc,
        "lore_accuracy_pair_count": accuracy_pair_count,
    }
    write_json(results_dir / "sae_eval_summary.json", summary)

    basis_rows = []
    for basis_idx in range(basis_v.shape[1]):
        basis_rows.append(
            {
                "basis": basis_idx,
                "pearson": float(pearson[basis_idx].item()),
                "spearman": float(spearman[basis_idx].item()),
                "mean_abs_score_error": float(score_abs_error[basis_idx].item()),
                "relative_score_error": float(score_rel_error[basis_idx].item()),
                "pair_pearson": float(pair_pearson[basis_idx].item()),
                "pair_spearman": float(pair_spearman[basis_idx].item()),
            }
        )
    write_rows(
        results_dir / "basis_score_correlations.csv",
        [
            "basis",
            "pearson",
            "spearman",
            "mean_abs_score_error",
            "relative_score_error",
            "pair_pearson",
            "pair_spearman",
        ],
        basis_rows,
    )

    activation_rows = [
        {
            "feature": feature_idx,
            "activation_frequency": float(activation_frequency[feature_idx].item()),
        }
        for feature_idx in range(z.shape[1])
    ]
    write_rows(results_dir / "activation_stats.csv", ["feature", "activation_frequency"], activation_rows)

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
