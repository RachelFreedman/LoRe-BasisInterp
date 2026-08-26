#!/usr/bin/env python3
"""
Plot metric-vs-step curves from one or more log_training_curves.py CSVs.
"""
import sys
import csv
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display needed; just write the PNG
import matplotlib.pyplot as plt


FIELDS = ["train_acc", "test_acc", "mean_abs_basis_cos", "mean_cos_vsft"]
TITLES = {
    "train_acc": "Train accuracy",
    "test_acc": "Test accuracy",
    "mean_abs_basis_cos": "Mean |cos| between basis columns (collapse)",
    "mean_cos_vsft": "Mean |cos| of bases to v_sft",
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
    p.add_argument("csvs", nargs="+", help="one or more log_training_curves.py CSV files")
    p.add_argument("--out", default="training_curves.png", help="output PNG path")
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.csvs)
    if not rows:
        print("No rows found in the given CSV(s).")
        sys.exit(1)

    by_alpha = defaultdict(list)
    for r in rows:
        by_alpha[float(r["alpha"])].append(r)
    for a in by_alpha:
        by_alpha[a].sort(key=lambda r: int(r["step"]))

    alphas = sorted(by_alpha.keys())
    base_test_acc = float(rows[0]["base_test_acc"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    cmap = plt.get_cmap("coolwarm")

    for idx, field in enumerate(FIELDS):
        ax = axes[idx]
        for i, alpha in enumerate(alphas):
            sub = by_alpha[alpha]
            steps = [int(r["step"]) for r in sub]
            vals = [float(r[field]) for r in sub]
            color = cmap(i / max(len(alphas) - 1, 1))
            ax.plot(steps, vals, marker="o", markersize=2, linewidth=1.3,
                    label=f"alpha={alpha:g}", color=color)
        ax.set_title(TITLES[field])
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    # single shared legend instead of one per subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training curves by alpha", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out} ({len(alphas)} alpha(s): {alphas})")


if __name__ == "__main__":
    main()
