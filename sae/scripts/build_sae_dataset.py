#!/usr/bin/env python3
"""Build SAE tensor datasets from Phase 1 PRISM embedding artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SAE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SAE_DIR.parent
sys.path.append(str(REPO_ROOT))

from sae.src.data import load_prism_response_embeddings, split_indices  # noqa: E402
from sae.src.io import ensure_dir, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", default="PRISM/data/prism/train_embeddings.pkl")
    parser.add_argument("--test-embeddings", default="PRISM/data/prism/test_embeddings.pkl")
    parser.add_argument("--output-dir", default="sae/data")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    train_rows = load_prism_response_embeddings(args.train_embeddings, source_split="train")
    test_rows = load_prism_response_embeddings(args.test_embeddings, source_split="test")
    embeddings = torch.cat([train_rows.embeddings, test_rows.embeddings], dim=0)
    metadata = train_rows.metadata + test_rows.metadata

    for embedding_id, row in enumerate(metadata):
        row["embedding_id"] = embedding_id

    pair_ids = list(dict.fromkeys(row["pair_id"] for row in metadata))
    pair_splits = split_indices(
        len(pair_ids),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    pair_id_to_split = {
        pair_ids[pair_index]: split_name
        for split_name, indices in pair_splits.items()
        for pair_index in indices.tolist()
    }
    split_row_indices = {split_name: [] for split_name in pair_splits}
    for metadata_index, row in enumerate(metadata):
        split_name = pair_id_to_split[row["pair_id"]]
        split_row_indices[split_name].append(metadata_index)

    splits = {
        split_name: torch.tensor(row_indices, dtype=torch.long)
        for split_name, row_indices in split_row_indices.items()
    }

    for split_name, indices in splits.items():
        split_embeddings = embeddings.index_select(0, indices)
        torch.save(split_embeddings, output_dir / f"sae_{split_name}.pt")
        for split_index, metadata_index in enumerate(indices.tolist()):
            metadata[metadata_index]["sae_split"] = split_name
            metadata[metadata_index]["sae_split_index"] = split_index

    write_jsonl(output_dir / "metadata.jsonl", metadata)
    summary = {
        "embedding_dim": int(embeddings.shape[1]),
        "total_embeddings": int(embeddings.shape[0]),
        "source_train_comparisons": len(train_rows.metadata) // 2,
        "source_test_comparisons": len(test_rows.metadata) // 2,
        "source_train_response_embeddings": len(train_rows.metadata),
        "source_test_response_embeddings": len(test_rows.metadata),
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "total_pairs": len(pair_ids),
        "sae_train_pairs": int(pair_splits["train"].numel()),
        "sae_val_pairs": int(pair_splits["val"].numel()),
        "sae_test_pairs": int(pair_splits["test"].numel()),
        "sae_train_embeddings": int(splits["train"].numel()),
        "sae_val_embeddings": int(splits["val"].numel()),
        "sae_test_embeddings": int(splits["test"].numel()),
    }
    write_json(output_dir / "dataset_summary.json", summary)

    print(f"Wrote SAE tensors and metadata to {output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
