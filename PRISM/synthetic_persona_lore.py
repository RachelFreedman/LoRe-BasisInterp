"""
Run LoRe (vanilla) and mLoRe/LoReV2 on the synthetic persona dataset.

This is a positive control with known ground truth: every user's reward is a known signed
combination of concept directions, and users genuinely disagree (38% of distinct response pairs
are ordered one way by some user and the other way by another). If a method cannot recover
per-user structure HERE, its null on real data says nothing about the data.

SPLITTING. Splits are by PROMPT, never by pair. The pool has 30 prompts x 20 concept-poles, and
each user draws 120 pairs from them, so a pair-level split would put the same prompt -- and often
the identical response texts -- on both sides. A trained model could then memorise "for this
prompt, response X beats Y" and score its held-out sibling for free, inflating LoRe against an
untrained base_rm. That is exactly the leak found and fixed in Community Alignment, where it
turned a +0.177 result into -0.019.

Reported on held-out prompts:
  base_rm         Skywork's true reward head
  global_meandiff one shared direction fitted across all users
  personal        each user's own mean-diff direction
  other_user      a random OTHER user's direction (the control)
  oracle          the user's TRUE planted direction (ceiling; synthetic only)
  LoRe / mLoRe    test accuracy at several ranks, plus recovery of the planted structure

Read-out: personal >> other_user and mLoRe > base_rm means per-user structure is recoverable.
Since labels are linear in the embedding by construction, this bounds what the method can do in
the best case -- it does not show personalization is recoverable in real data.

CPU-only.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized, LoReV2, canonical_variation_axes   # noqa: E402
from rm_head_utils import load_reward_head                             # noqa: E402
from synthetic_recovery import collapse_metrics, eval_acc              # noqa: E402


def unit(v):
    return v / (v.norm() + 1e-8)


def acc(D, d):
    return (D @ d > 0).float().mean().item() if len(D) else float("nan")


def build_splits(pairs, emb, val_frac, test_frac, seed, max_train_prompts=None):
    """Per user: (train, val, test) difference matrices, split by whole prompt."""
    rng = random.Random(seed)
    prompts = sorted({p["prompt_idx"] for p in pairs})
    shuffled = prompts[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(test_frac * len(prompts))))
    n_val = max(1, int(round(val_frac * len(prompts)))) if val_frac else 0
    test_p = set(shuffled[:n_test])
    val_p = set(shuffled[n_test:n_test + n_val])
    train_p = shuffled[n_test + n_val:]
    if max_train_prompts:
        # Restricting TRAIN prompts while holding the test set fixed isolates the effect of
        # data-per-user: if the metric degrades as train prompts shrink, more prompts would help.
        dropped = set(train_p[max_train_prompts:])
        train_p = train_p[:max_train_prompts]
    else:
        dropped = set()
    print(f"prompt split: {len(prompts) - n_test - n_val} train / {n_val} val / {n_test} test")

    by_user = defaultdict(lambda: {"train": [], "val": [], "test": []})
    for p in pairs:
        d = emb[p["chosen_key"]] - emb[p["rejected_key"]]
        if p["prompt_idx"] in dropped:
            continue
        bucket = "test" if p["prompt_idx"] in test_p else (
            "val" if p["prompt_idx"] in val_p else "train")
        by_user[p["user_id"]][bucket].append(d)

    out = {}
    for u, b in by_user.items():
        if not b["train"] or not b["test"]:
            continue
        out[u] = (torch.stack(b["train"]).float(),
                  torch.stack(b["val"]).float() if b["val"] else None,
                  torch.stack(b["test"]).float())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/synthetic_personas/pairs.json")
    ap.add_argument("--embeddings", default="data/synthetic_personas/pool_embeddings.pt")
    ap.add_argument("--personas", default="data/synthetic_personas/personas.json")
    ap.add_argument("--vectors", default="data/prism/concept_vectors_v2.pt")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16])
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max_train_prompts", type=int, default=None,
                    help="cap training prompts (test set held fixed) to probe data-per-user")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/synthetic_personas/results.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with open(args.pairs) as f:
        pairs = json.load(f)
    with open(args.personas) as f:
        blob = json.load(f)
    emb = torch.load(args.embeddings, weights_only=False)
    cv = torch.load(args.vectors, weights_only=False)
    concepts = blob["concepts"]

    users = build_splits(pairs, emb, args.val_frac, args.test_frac, args.seed,
                         args.max_train_prompts)
    uids = sorted(users)
    ntr = [len(users[u][0]) for u in uids]
    nte = [len(users[u][2]) for u in uids]
    print(f"{len(uids)} users; train/user median {int(np.median(ntr))}, "
          f"test/user median {int(np.median(nte))}")

    # ---- reference directions -------------------------------------------------
    C = F.normalize(torch.stack([cv[c].float().reshape(-1) for c in concepts], 1), dim=0)
    # Labels were generated from STANDARDISED projections: utility = sum_c w_c * (e.c_c - mu_c)/sd_c.
    # As a function of e that is e @ (C @ (w / sd)), so the true planted direction carries the 1/sd
    # factor. Using C @ w instead gives a mis-specified "oracle" that the empirical mean-diff can
    # beat -- which is what happened on the first run (personal 0.919 > oracle 0.826).
    P = torch.stack([emb[k] for k in sorted(emb)]).float() @ C
    sd = P.std(0).clamp_min(1e-6)
    truth = {}
    for s in blob["personas"]:
        w = torch.tensor([s["weights"].get(c, 0.0) for c in concepts])
        truth[s["user_id"]] = unit(C @ (w / sd))

    head = unit(load_reward_head().reshape(-1).float())
    glob = unit(torch.cat([users[u][0] for u in uids]).mean(0))
    personal = {u: unit(users[u][0].mean(0)) for u in uids}

    rng = random.Random(args.seed)
    rows = {}
    rows["base_rm"] = float(np.mean([acc(users[u][2], head) for u in uids]))
    rows["global_meandiff"] = float(np.mean([acc(users[u][2], glob) for u in uids]))
    rows["personal"] = float(np.mean([acc(users[u][2], personal[u]) for u in uids]))
    rows["other_user"] = float(np.mean([
        acc(users[u][2], personal[rng.choice([v for v in uids if v != u])]) for u in uids]))
    rows["oracle_planted"] = float(np.mean([acc(users[u][2], truth[u]) for u in uids]))

    print("\nREFERENCE DIRECTIONS (held-out prompts)")
    for k in ("base_rm", "global_meandiff", "other_user", "personal", "oracle_planted"):
        print(f"  {k:<18}{rows[k]:.4f}")
    print(f"  personal - other_user  {rows['personal'] - rows['other_user']:+.4f}")

    # ---- LoRe vs mLoRe --------------------------------------------------------
    X = [users[u][0] for u in uids]
    Xval = [users[u][1] for u in uids] if all(users[u][1] is not None for u in uids) else None
    Xte = [users[u][2] for u in uids]
    results = []
    print(f"\n{'model':<9}{'K':>4}{'test':>9}{'vs base':>10}{'min|cos|':>10}{'user->axis':>12}")
    for model in ("vanilla", "v2"):
        for K in args.ranks:
            torch.manual_seed(args.seed)
            if model == "vanilla":
                anchor = load_reward_head().reshape(-1, 1)
                m = LoRe_regularized(anchor, args.alpha, len(uids), 4096, K,
                                     args.iters, args.lr)
                W, V = m.train(X)
            else:
                m = LoReV2(len(uids), 4096, K, lam_pop=args.lam_pop, lam_d=args.lam_d,
                           num_iterations=args.iters, learning_rate=args.lr, verbose=False)
                W, V = m.train(X, val=Xval)
            V = V.detach().cpu() if torch.is_tensor(V) else V
            W = W.detach().cpu() if torch.is_tensor(W) else W
            te = eval_acc(W, V, Xte)
            dirs = [unit(V @ W[i]) for i in range(len(uids))]
            mc = collapse_metrics(V)   # float; 1.0 by convention at K=1
            # does each user's recovered direction match their own planted direction better
            # than any other user's? transform-invariant, so it survives V -> VR.
            T = torch.stack([truth[u] for u in uids])
            S = torch.stack([d for d in dirs]) @ T.T
            hit = float((S.argmax(1) == torch.arange(len(uids))).float().mean())
            print(f"{model:<9}{K:>4}{te:>9.4f}{te - rows['base_rm']:>+10.4f}"
                  f"{mc:>10.4f}{hit:>12.3f}")
            results.append({"model": model, "K": K, "test_acc": te,
                            "vs_base": te - rows["base_rm"], "min_abs_cos": mc,
                            "user_axis_match": hit})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference": rows, "lore": results,
                   "config": vars(args), "n_users": len(uids)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
