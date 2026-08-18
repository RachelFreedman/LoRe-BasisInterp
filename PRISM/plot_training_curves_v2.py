#!/usr/bin/env python3
"""
USAGE
-----
    python plot_training_curves_v2.py training_curves_v2.csv

Produces a single PNG with two panels (train_acc, test_acc), one line per rank, x-axis =
training step. base_test_acc is drawn as a dashed reference line on both panels.
"""
import sys
import csv
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # no display needed; just write the PNG
import matplotlib.pyplot as plt

FIELDS = ["train_acc", "test_acc"]
TITLES = {
    "train_acc": "Train accuracy",
    "test_acc": "Test accuracy",
}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", help="one or more log_training_curves_v2.py CSV files")
    p.add_argument("--out", default="training_curves_v2.png", help="output PNG path")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.csvs)
    if not rows:
        print("No rows found in the given CSV(s).")
        sys.exit(1)

    by_rank = defaultdict(list)
    for r in rows:
        by_rank[int(float(r["rank"]))].append(r)
    for k in by_rank:
        by_rank[k].sort(key=lambda r: int(r["step"]))
    ranks = sorted(by_rank.keys())
    base_test_acc = float(rows[0]["base_test_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmap = plt.get_cmap("viridis")

    for idx, field in enumerate(FIELDS):
        ax = axes[idx]
        for i, rank in enumerate(ranks):
            sub = by_rank[rank]
            steps = [int(r["step"]) for r in sub]
            vals = [float(r[field]) for r in sub]
            color = cmap(i / max(len(ranks) - 1, 1))
            ax.plot(steps, vals, marker="o", markersize=2, linewidth=1.3,
                    label=f"rank={rank}", color=color)
        ax.axhline(base_test_acc, linestyle="--", color="gray", linewidth=1,
                   label="base RM test_acc")
        ax.set_title(TITLES[field])
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    # single shared legend instead of one per subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("LoReV2 training curves by rank", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out} ({len(ranks)} rank(s): {ranks})")


if __name__ == "__main__":
    main()
