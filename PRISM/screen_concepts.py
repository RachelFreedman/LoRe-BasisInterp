"""
Screen the extended concept library down to a set of near-independent axes.

Three things, all CPU-only:

1. GENERATOR DRIFT. The six base concepts exist in both concept_vectors.pt (Sonnet 4.5) and
   concept_vectors_v2.pt (Opus 4.6) with identical prompt text, so cos(v1, v2) per concept
   measures how much the generating model moves a concept vector. This is the number the
   limitations section otherwise has to hand-wave.

2. GREEDY SELECTION. Starting from the most independent pair, repeatedly add whichever remaining
   concept maximises effective rank. Effective rank = exp(entropy of the singular-value variance
   spectrum) of the unit-normalised vectors -- how many directions the set really spans.

3. v_pop COVERAGE. mLoRe's shared direction sits at cos 0.034 to the quality axis, so the original
   library probably does not span it. Reports how much of v_pop the selected set explains, which
   says whether the extension actually bought interpretive reach.

Usage:
  python PRISM/screen_concepts.py
  python PRISM/screen_concepts.py --target_rank 7
"""

import argparse
import os

import torch
import torch.nn.functional as F


def unit(v):
    return F.normalize(v.float().reshape(-1), dim=0)


def eff_rank(names, cv):
    A = torch.stack([unit(cv[c]) for c in names], 1)
    s = torch.linalg.svdvals(A)
    p = s ** 2 / (s ** 2).sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def max_abs_cos(names, cv):
    if len(names) < 2:
        return 0.0
    return max(abs(float(unit(cv[a]) @ unit(cv[b])))
               for i, a in enumerate(names) for b in names[i + 1:])


def explained(target, names, cv):
    """Fraction of a unit target vector's norm captured by the span of the concept set."""
    A = torch.stack([unit(cv[c]) for c in names], 1)
    Q, _ = torch.linalg.qr(A)
    return float((Q.T @ unit(target)).pow(2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/prism/concept_vectors_v2.pt")
    ap.add_argument("--v1", default="data/prism/concept_vectors.pt")
    ap.add_argument("--vpop", default="results/community_alignment/ca_v_pop.pt")
    ap.add_argument("--vpop_key", default="CA_K8_seed42_tuned")
    ap.add_argument("--target_rank", type=float, default=6.0,
                    help="stop adding once effective rank reaches this")
    ap.add_argument("--min_gain", type=float, default=0.35,
                    help="stop if the best remaining concept adds less than this to eff. rank")
    args = ap.parse_args()

    cv2 = torch.load(args.v2, weights_only=False)
    names = list(cv2)
    print(f"loaded {len(names)} v2 concepts: {', '.join(names)}\n")

    # ---- 1. generator drift -------------------------------------------------
    if os.path.exists(args.v1):
        cv1 = torch.load(args.v1, weights_only=False)
        shared = [c for c in names if c in cv1]
        if shared:
            print("GENERATOR DRIFT  cos(Sonnet 4.5 vector, Opus 4.6 vector), same prompt text")
            cs = []
            for c in shared:
                x = float(unit(cv1[c]) @ unit(cv2[c]))
                cs.append(x)
                print(f"  {c:<14} {x:+.3f}")
            print(f"  {'mean':<14} {sum(cs) / len(cs):+.3f}")
            print("  (1.0 would mean the generating model is irrelevant; low values mean a "
                  "concept vector\n   is partly an artifact of which model wrote the "
                  "contrastive text)\n")

    # ---- 2. greedy selection ------------------------------------------------
    print("PAIRWISE |cos| -- most entangled pairs")
    pairs = sorted(((abs(float(unit(cv2[a]) @ unit(cv2[b]))), a, b)
                    for i, a in enumerate(names) for b in names[i + 1:]), reverse=True)
    for v, a, b in pairs[:8]:
        print(f"  {a:<14} {b:<14} {v:.3f}")
    print(f"  mean off-diagonal |cos| {sum(p[0] for p in pairs) / len(pairs):.3f}\n")

    seed = min(((abs(float(unit(cv2[a]) @ unit(cv2[b]))), a, b)
                for i, a in enumerate(names) for b in names[i + 1:]))
    sel = [seed[1], seed[2]]
    print("GREEDY SELECTION")
    print(f"  seed: {sel[0]} + {sel[1]}   |cos| {seed[0]:.3f}   eff.rank "
          f"{eff_rank(sel, cv2):.3f}")

    while len(sel) < len(names):
        rest = [c for c in names if c not in sel]
        best_c, best_r = max(((c, eff_rank(sel + [c], cv2)) for c in rest),
                             key=lambda t: t[1])
        gain = best_r - eff_rank(sel, cv2)
        mx = max(abs(float(unit(cv2[best_c]) @ unit(cv2[s]))) for s in sel)
        stop = gain < args.min_gain
        flag = "  <- below min_gain, stopping" if stop else ""
        print(f"  +{best_c:<13} eff.rank {best_r:.3f} (+{gain:.3f})  max|cos| {mx:.3f}{flag}")
        if stop:
            break
        sel.append(best_c)
        if eff_rank(sel, cv2) >= args.target_rank:
            print(f"  reached target rank {args.target_rank}")
            break

    print(f"\nSELECTED ({len(sel)}): {', '.join(sel)}")
    print(f"  effective rank {eff_rank(sel, cv2):.3f} of {len(sel)}"
          f"   max|cos| {max_abs_cos(sel, cv2):.3f}")
    dropped = [c for c in names if c not in sel]
    if dropped:
        print(f"  dropped: {', '.join(dropped)}")

    # ---- 3. v_pop coverage --------------------------------------------------
    if os.path.exists(args.vpop):
        blob = torch.load(args.vpop, weights_only=False)
        if args.vpop_key in blob:
            vpop = blob[args.vpop_key]["v_pop_unit"]
            print("\nv_pop COVERAGE  fraction of mLoRe's shared direction inside the span")
            print(f"  selected v2 set ({len(sel)})   {explained(vpop, sel, cv2):.3f}")
            print(f"  all v2 concepts ({len(names)})   {explained(vpop, names, cv2):.3f}")
            if os.path.exists(args.v1):
                cv1 = torch.load(args.v1, weights_only=False)
                n1 = list(cv1)
                print(f"  original library ({len(n1)})  {explained(vpop, n1, cv1):.3f}")
            # A random k-dim subspace captures k/4096 of any fixed unit vector in expectation.
            print(f"  random {len(sel)}-dim null   {len(sel) / 4096:.4f}")


if __name__ == "__main__":
    main()
