"""Does the SAE's advantage survive resampling the DATA, not just the model?

Pre-registered before running:
  Testing  -- whether the SAE's win over the direct-correlation baseline (9/10 seeds
              at the frozen top-32 window) holds when each seed also resamples the
              contrast items, rather than reusing one fixed set of 50 pairs.
  How      -- per seed, split each concept's 50 contrast pairs at random: plant users
              from half A's embeddings, build the concept fingerprints from half B's
              TEXT. Score SAE@top32 (frozen from the width run, NOT re-tuned) against
              the no-SAE correlation baseline and a random-direction null.
  Outcomes -- SAE still ahead on most seeds => the advantage is real and not an
              artifact of one fixed item set. Collapse => the earlier 9/10 was
              driven by those particular 50 pairs.

This also subsumes the earlier disjoint-text control: planting text and scoring text
never overlap, by construction, on every seed.

Window is FROZEN. Nothing here is tuned on the evaluation concepts.
CPU only: utils.py fixes its device at import.
"""
import argparse, csv, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "PRISM"))

from PRISM.synthetic_concept_users import build_users, set_seed  # noqa
from PRISM.planted_directions import group_mean_directions  # noqa
from PRISM.random_baseline import random_unit_directions  # noqa
from sae.scripts.feature_text_profiles import PROXIES, prism_texts, load_sae  # noqa

EVAL = ["fluency", "repetition", "confidence", "creativity", "formatting", "diversity"]
FROZEN_WINDOW = 32   # selected on the 5 tuning concepts in readout_width_check.py


def unit(v):
    return v / (v.norm() + 1e-8)


def text_signatures(pairs, names, idx_by_concept):
    """Concept fingerprints from contrast TEXT only, over the given pair indices."""
    rows = []
    for c in names:
        idx = idx_by_concept[c]
        hi = np.array([[PROXIES[k](pairs[c][i]["high_response"]) for k in PROXIES] for i in idx])
        lo = np.array([[PROXIES[k](pairs[c][i]["low_response"]) for k in PROXIES] for i in idx])
        rows.append((hi - lo).mean(0))
    M = np.stack(rows)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def match(sig, Csig):
    p = sig / (np.linalg.norm(sig) + 1e-9)
    Cn = Csig / (np.linalg.norm(Csig, axis=1, keepdims=True) + 1e-9)
    s = Cn @ p
    i = int(np.argmax(np.abs(s)))
    return i, (1 if s[i] > 0 else -1)


def sign_test(a, b):
    w = sum(x > y for x, y in zip(a, b)); l = sum(x < y for x, y in zip(a, b))
    n = w + l
    if n == 0:
        return w, l, len(a) - n, 1.0
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n)
    return w, l, len(a) - n, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(REPO / "data/prism/contrastive_pair_embeddings.pt"))
    ap.add_argument("--pairs", default=str(REPO / "data/prism/contrastive_pairs.json"))
    ap.add_argument("--checkpoint", default=str(REPO / "sae/checkpoints/d3/model.pt"))
    ap.add_argument("--profiles", default=str(REPO / "results/planted/feature_text_profiles.pt"))
    ap.add_argument("--sae-data", default=str(REPO / "sae/data"))
    ap.add_argument("--prism-train", default=str(REPO / "PRISM/data/prism/train_embeddings.pkl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--window", type=int, default=FROZEN_WINDOW)
    ap.add_argument("--users_per_group", type=int, default=20)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--subset_frac", type=float, default=0.7)
    ap.add_argument("--high_frac", type=float, default=0.5)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--fixed-split", action="store_true",
                    help="use the SAME pair split on every seed. Disambiguates the two "
                         "things the resampled run changed at once: resampling the "
                         "contrast items, and halving the data. Fixed-split holds volume "
                         "at 25/25 while removing the resampling.")
    ap.add_argument("--out", default=str(REPO / "results/planted/resampled_seeds.json"))
    ap.add_argument("--rows", default=str(REPO / "results/planted/resampled_seeds_rows.csv"))
    a = ap.parse_args()

    device = torch.device("cpu")
    emb = torch.load(a.emb, weights_only=False)
    pairs = json.load(open(a.pairs))
    model, cfg = load_sae(Path(a.checkpoint), device)
    dec = model.decoder.weight.detach().cpu().float()
    blob = torch.load(a.profiles, weights_only=False)
    P = blob["profiles"]; names = blob["concepts"]
    w = blob["activity"].numpy() * (np.abs(P).sum(1) > 0)
    n_groups = 2 * len(names)
    print(f"pair split: {'FIXED (same 25 every seed)' if a.fixed_split else 'RESAMPLED per seed'}")
    print(f"window FROZEN at top{a.window} (from the width run; not re-tuned here)")
    print(f"candidates: {len(names)} concepts x 2 signs = {n_groups}, chance {1/n_groups:.3f}\n")

    # no-SAE baseline machinery (PRISM side is fixed across seeds)
    meta = [json.loads(l) for l in open(Path(a.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == a.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    x = torch.load(Path(a.sae_data) / f"sae_{a.split}.pt", map_location="cpu").float()
    texts = prism_texts(Path(a.prism_train), meta, a.split)
    keep = [i for i, t in enumerate(texts) if t]
    X = x[keep]
    Pr = np.array([[PROXIES[q](texts[i]) for q in PROXIES] for i in keep], dtype=np.float32)
    Pz = (Pr - Pr.mean(0)) / (Pr.std(0) + 1e-9)

    def sae_sig(v):
        al = (dec.T @ F.normalize(v, dim=0)).numpy()
        top = np.argsort(np.abs(al) * w)[::-1][:a.window]
        return al[top] @ P[top]

    def nosae_sig(v):
        s = (X @ F.normalize(v, dim=0)).numpy()
        s = (s - s.mean()) / (s.std() + 1e-9)
        return (Pz * s[:, None]).mean(0)

    from utils import LoReV2
    rows, sae_acc, base_acc = [], [], []
    for seed in a.seeds:
        g = torch.Generator().manual_seed(1000 if a.fixed_split else 1000 + seed)
        idxA, idxB = {}, {}
        embA = {}
        for c in names:
            n = len(pairs[c]); perm = torch.randperm(n, generator=g)
            idxA[c] = perm[: n // 2].tolist(); idxB[c] = perm[n // 2:].tolist()
        for c in EVAL:
            ia = idxA[c]
            embA[c] = {"high": emb[c]["high"][ia], "low": emb[c]["low"][ia]}
        # fingerprints from the OTHER half's text
        Csig = text_signatures(pairs, names, idxB)

        cvsA = {c: unit(embA[c]["high"].float().mean(0) - embA[c]["low"].float().mean(0))
                for c in EVAL}
        V = torch.stack([cvsA[c].float() for c in EVAL], dim=1)
        set_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        tf, _, gid = build_users(embA, EVAL, a.users_per_group, a.test_frac, gen,
                                 a.high_frac, a.subset_frac)
        m = LoReV2(len(tf), V.shape[0], 2 * len(EVAL), lam_pop=a.lam_pop, lam_d=a.lam_d,
                   num_iterations=a.iters, learning_rate=a.lr, verbose=False)
        m.train(tf)
        dirs = group_mean_directions(m.reward_dirs().detach().cpu().T, gid, 2 * len(EVAL))

        sh = bh = 0
        for gi in range(len(dirs)):
            tc, ts = EVAL[gi // 2], (1 if gi % 2 == 0 else -1)
            si, ss = match(sae_sig(dirs[gi]), Csig)
            bi, bs = match(nosae_sig(dirs[gi]), Csig)
            sok = int(names[si] == tc and ss == ts); bok = int(names[bi] == tc and bs == ts)
            sh += sok; bh += bok
            rows.append({"seed": seed, "true_concept": tc,
                         "true_sign": "high" if ts > 0 else "low",
                         "sae_pred": names[si], "sae_sign": "high" if ss > 0 else "low",
                         "nosae_pred": names[bi], "nosae_sign": "high" if bs > 0 else "low",
                         "sae_correct": sok, "nosae_correct": bok})
        sae_acc.append(sh / len(dirs)); base_acc.append(bh / len(dirs))
        print(f"   seed {seed}: SAE {sae_acc[-1]:.3f}   no-SAE {base_acc[-1]:.3f}")

    # null at the frozen window, against the last seed's fingerprints
    nulls = torch.stack(random_unit_directions(dec.shape[0], a.n_null, seed=0))
    truth = [(EVAL[i // 2], 1 if i % 2 == 0 else -1) for i in range(2 * len(EVAL))]
    npred = [match(sae_sig(v), Csig) for v in nulls]
    null = float(np.mean([[int(names[c] == tc and s == ts) for tc, ts in truth]
                          for c, s in npred]))

    W, L, T, p = sign_test(sae_acc, base_acc)
    per = {}
    for r in rows:
        d = per.setdefault(r["true_concept"], [0, 0, 0])
        d[0] += r["sae_correct"]; d[1] += r["nosae_correct"]; d[2] += 1

    with open(a.rows, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    out = {"window": a.window, "seeds": a.seeds, "chance": 1 / n_groups, "null": null,
           "sae_mean": float(np.mean(sae_acc)), "sae_std": float(np.std(sae_acc)),
           "nosae_mean": float(np.mean(base_acc)), "nosae_std": float(np.std(base_acc)),
           "sae_per_seed": sae_acc, "nosae_per_seed": base_acc,
           "sign_test": {"wins": W, "losses": L, "ties": T, "p": p},
           "per_concept": {k: {"sae": v[0], "nosae": v[1], "n": v[2]} for k, v in per.items()},
           "fixed_split": bool(a.fixed_split),
           "design": ("FIXED pair split" if a.fixed_split else "per-seed resample") +
                     " of contrast pairs; plant on half A embeddings, "
                     "fingerprints from half B text; window frozen, not re-tuned"}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n=== {len(a.seeds)} seeds, data resampled per seed ===")
    print(f"  chance {1/n_groups:.3f}   null {null:.3f}")
    print(f"  SAE     {np.mean(sae_acc):.3f} +/- {np.std(sae_acc):.3f}")
    print(f"  no-SAE  {np.mean(base_acc):.3f} +/- {np.std(base_acc):.3f}")
    print(f"  sign test: {W}W-{L}L-{T}T   p = {p:.3f}")
    print("\n  per concept (SAE / no-SAE out of n):")
    for c in EVAL:
        v = per[c]; print(f"    {c:<12} {v[0]:>3} / {v[1]:>3}   of {v[2]}")
    print(f"\nSaved {a.out} and {a.rows}")


if __name__ == "__main__":
    main()
