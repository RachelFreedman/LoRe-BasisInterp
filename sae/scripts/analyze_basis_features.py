#!/usr/bin/env python3
"""Numeric SAE feature attribution onto LoRe bases (observational, not causal)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))

from sae.src.attribution import (  # noqa: E402
    decoder_basis_alignment,
    feature_activation_stats,
    mean_abs_contribution,
    operational_kept_mask,
    top_features_by_contribution,
)
from sae.src.io import ensure_dir, write_json  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    parser.add_argument("--data-dir", default="sae/data")
    parser.add_argument("--basis-matrices", default="PRISM/basis_matrices.pt")
    parser.add_argument("--run-key", default="PART2_K10_seed42")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--results-dir", default="sae/results/d3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-n", type=int, default=50)
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


def collect_latents(
    model: TopKSAE,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    zs = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            zs.append(model.encode(batch.to(device)).cpu())
    return torch.cat(zs, dim=0)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    device = choose_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
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

    x = torch.load(Path(args.data_dir) / f"sae_{args.split}.pt", map_location="cpu").float()
    z = collect_latents(model, x, args.batch_size, device)
    mean_abs, act_freq = feature_activation_stats(z)

    matrices = torch.load(args.basis_matrices, map_location="cpu")
    if args.run_key not in matrices:
        raise KeyError(f"run key {args.run_key!r} not in {args.basis_matrices}")
    run = matrices[args.run_key]
    basis_v = run["V"].float()
    user_w = run["W"].float()
    kept = operational_kept_mask(user_w)
    max_user_w = user_w.max(dim=0).values

    dec = model.decoder.weight.detach().cpu().float()
    alignment, cosine = decoder_basis_alignment(dec, basis_v)
    contrib = mean_abs_contribution(mean_abs, alignment)

    rows = top_features_by_contribution(
        alignment,
        cosine,
        mean_abs,
        act_freq,
        contrib,
        top_n=args.top_n,
        kept_mask=kept,
        max_user_weight=max_user_w,
    )
    fieldnames = list(rows[0].keys()) if rows else []
    write_rows(results_dir / "top_features_per_basis.csv", fieldnames, rows)

    ops_rows = [r for r in rows if r["is_operational_kept"] == 1]
    write_rows(results_dir / "top_features_per_basis_operational.csv", fieldnames, ops_rows)

    meta = {
        "run_key": args.run_key,
        "split": args.split,
        "n_embeddings": int(x.shape[0]),
        "dict_size": int(config["dict_size"]),
        "n_bases": int(basis_v.shape[1]),
        "operational_kept_basis_ids": kept.nonzero(as_tuple=False).view(-1).tolist(),
        "bases_kept_metadata": run.get("bases_kept"),
        "top_n_per_sign": args.top_n,
        "primary_sort": "mean_abs_contribution within sign(alignment)",
        "note": (
            "Attribution is observational (decoder·V projections), not causal. "
            "No semantic feature labels are assigned."
        ),
    }
    write_json(results_dir / "attribution_meta.json", meta)

    print(
        {
            "wrote": str(results_dir / "top_features_per_basis.csv"),
            "operational_rows": len(ops_rows),
            "all_rows": len(rows),
            "operational_kept_basis_ids": meta["operational_kept_basis_ids"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
