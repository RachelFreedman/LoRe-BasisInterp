#!/usr/bin/env python3
"""
Shared random-direction baseline utilities.

  random_unit_directions(...)  -- for direction-vs-direction comparisons
                                   (e.g. stability_check.py's cosine check)
  pairwise_accuracy_null(...)  -- for accuracy-vs-chance comparisons (e.g.
                                   "is this residual's held-out accuracy
                                   above what a random direction gets?" --
                                   the null Daglawy said doesn't exist yet)
"""
import numpy as np
import torch
import torch.nn.functional as F


def random_unit_directions(num_features, n, seed=0):
    """n isotropic random unit vectors in R^num_features, as a list of [F]
    tensors. Use for direction-vs-direction cosine comparisons."""
    g = torch.Generator().manual_seed(seed)
    R = F.normalize(torch.randn(num_features, n, generator=g), dim=0)
    return [R[:, i] for i in range(n)]


def pairwise_accuracy_null(feats, n_null=300, seed=0):
    """Null distribution for PAIRWISE ACCURACY: score n_null random
    directions on the same held-out feature diffs an experiment scores its
    real direction on, and return their accuracies.

    feats: list of [m_u, F] per-user (or per-group) diff tensors, same
    shape/format eval_acc() and acc() elsewhere in this repo expect.

    Use this to answer "is this direction's accuracy actually above
    chance in this data, or would a random direction do about as well?" --
    e.g. for the v = c*v_sft + r decomposition, score r ALONE against this
    null before concluding anything from a bare accuracy number.

    Returns a list of n_null accuracies (each already averaged across the
    groups in `feats`, matching eval_acc()'s per-user-then-mean convention).
    """
    num_features = feats[0].shape[1]
    dirs = random_unit_directions(num_features, n_null, seed=seed)
    accs = []
    for d in dirs:
        per_group = [(X @ d > 0).float().mean().item() for X in feats]
        accs.append(float(np.mean(per_group)))
    return accs


def percentile_of(value, null_values):
    """Where does `value` fall inside the null distribution, as a
    percentile (0-100)? e.g. 97.3 means the real value beats 97.3% of
    random directions -- useful for a one-line significance read-out."""
    null_values = np.asarray(null_values)
    return float((null_values < value).mean() * 100)


if __name__ == "__main__":
    # quick self-check, no data files needed
    torch.manual_seed(0)
    dirs = random_unit_directions(16, 5, seed=0)
    assert len(dirs) == 5 and dirs[0].shape == (16,)
    assert abs(dirs[0].norm().item() - 1.0) < 1e-5, "should be unit norm"

    fake_feats = [torch.randn(20, 16) for _ in range(4)]
    accs = pairwise_accuracy_null(fake_feats, n_null=200, seed=0)
    assert len(accs) == 200
    print(f"self-check OK: {len(dirs)} unit dirs (norm={dirs[0].norm():.4f}), "
          f"{len(accs)} null accuracies, mean={np.mean(accs):.4f} "
          f"(should be close to 0.5 for pure noise)")
    print(f"percentile_of(0.55, accs) = {percentile_of(0.55, accs):.1f}  "
          f"(sanity: should be well below 100, above 0)")
