"""
Run LoRe on Community Alignment -- the dataset that actually has enough data per user.

Why this dataset: synthetic_recovery.py found a sharp phase transition at ~50 pairs/user (below ~25
LoRe cannot recover even clean planted axes; at 100+ it recovers them near-perfectly). PRISM has ~15
pairs/user, which is why it nulled out. Community Alignment (English) has a MEDIAN of 201 pairs/user
with 941 users above 100 -- comfortably inside the regime where LoRe demonstrably works.

So this is the real test of the question PRISM could not answer: with adequate data per user, does
LoRe find genuine personalization structure that beats a single global reward direction?

Reported, all on HELD-OUT pairs (split per user, so test prompts are unseen):
  * base_rm        : Skywork's true reward head -- the bar LoRe must clear
  * global_meandiff: one shared direction fitted on everyone
  * personal       : each user's own mean-diff direction
  * other_user     : a random OTHER user's direction (the key control; if personal ~= other_user
                     there is no user-specific signal, only a shared quality axis)
  * LoRe test acc at several ranks, plus basis collapse metrics

Read-out:
  * LoRe > base_rm AND personal > other_user -> real personalization exists here; PRISM's null was a
    data-volume artifact and LoRe works when the data supports it.
  * LoRe ~= base_rm and personal ~= other_user -> even with ~200 pairs/user there is no separable
    per-user structure, which would point at preference data/representation rather than data volume.

CPU-only once embeddings exist.
"""
import argparse
import csv
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
from utils import LoRe_regularized                                   # noqa: E402
from rm_head_utils import load_reward_head                           # noqa: E402
from synthetic_recovery import collapse_metrics, eval_acc            # noqa: E402
from embed_community_alignment import text_key                       # noqa: E402


def unit(v):
    return v / (v.norm() + 1e-8)


def acc(D, direction):
    return (D @ direction > 0).float().mean().item()


def build_user_diffs(pairs, emb_lookup, min_pairs, test_frac, gen, split_by_turn=True):
    """{user: (train_diffs [n,4096], test_diffs [m,4096])}, split per user.

    split_by_turn is essential for a valid held-out set. Each turn yields 3 pairs (the preferred
    response vs each of the other 3), so those pairs share the SAME prompt and the SAME chosen
    response. Splitting by pair would put siblings on both sides, letting a trained model memorise
    "for this prompt, response_d wins" and score the test sibling for free -- inflating LoRe (which
    trains) while leaving base_rm (which does not) untouched, i.e. an unfair comparison. Grouping by
    (conversation, turn) keeps all siblings on one side, so test prompts are genuinely unseen.
    """
    by_user = defaultdict(lambda: defaultdict(list))   # user -> turn key -> [diffs]
    missing = 0
    for p in pairs:
        kc, kr = text_key(p["prompt"], p["chosen"]), text_key(p["prompt"], p["rejected"])
        if kc not in emb_lookup or kr not in emb_lookup:
            missing += 1
            continue
        turn_key = (p.get("conversation_id"), p.get("turn")) if split_by_turn else len(
            by_user[p["user_id"]])
        by_user[p["user_id"]][turn_key].append(emb_lookup[kc] - emb_lookup[kr])
    if missing:
        print(f"[warn] {missing} pairs skipped (embedding not found -- partial embed run?)")

    out = {}
    for u, groups in by_user.items():
        keys = list(groups)
        total = sum(len(groups[k]) for k in keys)
        if total < min_pairs:
            continue
        perm = torch.randperm(len(keys), generator=gen).tolist()
        n_test_groups = max(1, int(round(test_frac * len(keys))))
        te_keys = {keys[i] for i in perm[:n_test_groups]}
        tr = [d for k in keys if k not in te_keys for d in groups[k]]
        te = [d for k in keys if k in te_keys for d in groups[k]]
        if not tr or not te:
            continue
        out[u] = (torch.stack(tr), torch.stack(te))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--min_pairs", type=int, default=50,
                    help="drop users below this many usable pairs (phase transition is ~50)")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--split_by_pair", action="store_true",
                    help="LEAKY: split train/test by pair, so sibling pairs from the same turn "
                         "(same prompt, same chosen) land on both sides. Only for measuring how "
                         "much that leakage inflates results")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results",
                                                  "community_alignment", "lore_results.csv"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    print(f"{len(pairs)} pairs, {len(emb_lookup)} embeddings")

    users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                             split_by_turn=not args.split_by_pair)
    uids = sorted(users)
    if not uids:
        print("No users met the threshold; nothing to run."); return
    npairs = [users[u][0].shape[0] + users[u][1].shape[0] for u in uids]
    print(f"{len(uids)} users with >= {args.min_pairs} pairs "
          f"(median {int(np.median(npairs))} pairs/user, max {max(npairs)})\n")

    train_feats = [users[u][0] for u in uids]
    test_feats = [users[u][1] for u in uids]

    # ---- reference directions on held-out pairs ----
    head = unit(load_reward_head().reshape(-1))
    global_dir = unit(torch.cat(train_feats, 0).mean(0))
    base_acc, glob_acc, pers_acc, other_acc = [], [], [], []
    for i, u in enumerate(uids):
        te = test_feats[i]
        base_acc.append(acc(te, head))
        glob_acc.append(acc(te, global_dir))
        pers_acc.append(acc(te, unit(train_feats[i].mean(0))))
        j = random.choice([k for k in range(len(uids)) if k != i])
        other_acc.append(acc(te, unit(train_feats[j].mean(0))))
    m = lambda x: float(np.mean(x))
    print("=== held-out reference directions ===")
    print(f"  base_rm (true head)  : {m(base_acc):.4f}   <- the bar LoRe must clear")
    print(f"  global_meandiff      : {m(glob_acc):.4f}")
    print(f"  personal             : {m(pers_acc):.4f}")
    print(f"  other_user (control) : {m(other_acc):.4f}")
    print(f"  personal - global    : {m(pers_acc)-m(glob_acc):+.4f}")
    print(f"  personal - other_user: {m(pers_acc)-m(other_acc):+.4f}   "
          f"(>0 => user-specific signal)\n")

    # ---- LoRe at several ranks ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["rank", "alpha", "train_acc", "test_acc", "min_abs_basis_cos",
                                      "bases_kept", "base_rm", "global", "personal", "other_user",
                                      "n_users", "median_pairs"])
    w.writeheader()

    print(f"{'rank':>5} | {'train':>7} | {'test':>7} | {'vs base':>8} | {'min|cos|':>8} | {'kept':>4}")
    print("-" * 56)
    for K in args.ranks:
        anchor = load_reward_head().reshape(-1, 1)
        model = LoRe_regularized(anchor, args.alpha, len(train_feats), 4096, K, args.iters, args.lr)
        Wk, Vk = model.train(train_feats)
        tr, te = eval_acc(Wk, Vk, train_feats), eval_acc(Wk, Vk, test_feats)
        mc = collapse_metrics(model.V.detach().cpu())
        print(f"{K:>5} | {tr:>7.4f} | {te:>7.4f} | {te-m(base_acc):>+8.4f} | {mc:>8.4f} | "
              f"{Vk.shape[1]:>4}")
        w.writerow({"rank": K, "alpha": args.alpha, "train_acc": round(tr, 4),
                    "test_acc": round(te, 4), "min_abs_basis_cos": round(mc, 4),
                    "bases_kept": Vk.shape[1], "base_rm": round(m(base_acc), 4),
                    "global": round(m(glob_acc), 4), "personal": round(m(pers_acc), 4),
                    "other_user": round(m(other_acc), 4), "n_users": len(uids),
                    "median_pairs": int(np.median(npairs))})
        f.flush()
    f.close()

    print(f"\nSaved {args.out}")
    print("Read-out: LoRe beating base_rm AND personal > other_user => real personalization at this "
          "data volume, so PRISM's null was about pairs-per-user. Both flat => the limitation is the "
          "preference data/representation, not volume.")


if __name__ == "__main__":
    main()
