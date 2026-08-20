"""How many independent axes does the concept library actually contain?

Two measurements that bound every result in the planted-direction check, and that
apply to any evaluation built on this library -- not just the SAE:

  1. EFFECTIVE DIMENSIONALITY. Participation ratio of the concept-vector Gram
     eigenvalues. Eleven named concepts is not eleven degrees of freedom if the
     vectors are correlated, and the unit of statistical analysis in a
     concept-library evaluation is the concept, not the item or the seed.

  2. VOCABULARY CEILING. Fingerprint each concept on the surface measures from one
     half of its contrast text, then try to retrieve it from the other half. A
     concept that cannot identify itself from independent text of itself is out of
     reach of any method scored this way, so its failure is not evidence about the
     SAE.

Both are pure geometry over saved artifacts: no model, no seeds, no fitting.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "sae" / "scripts"))
from feature_text_profiles import PROXIES  # noqa: E402


def effective_axes(M):
    """Participation ratio of the Gram eigenvalues: 1 = one shared direction,
    n = fully orthogonal."""
    G = (F.normalize(M, dim=1) @ F.normalize(M, dim=1).T).numpy()
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()), np.sort(ev)[::-1], G


def signature(pairs_c, idx):
    hi = np.array([[PROXIES[k](pairs_c[i]["high_response"]) for k in PROXIES] for i in idx])
    lo = np.array([[PROXIES[k](pairs_c[i]["low_response"]) for k in PROXIES] for i in idx])
    return (hi - lo).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_vectors", default=str(REPO / "data/prism/concept_vectors.pt"))
    ap.add_argument("--pairs", default=str(REPO / "data/prism/contrastive_pairs.json"))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/planted/library_geometry.json"))
    a = ap.parse_args()

    SIX = ["fluency", "repetition", "confidence", "creativity", "formatting", "diversity"]
    cv = torch.load(a.concept_vectors, weights_only=False)
    names = list(cv)
    M = torch.stack([cv[c].float() for c in names])

    pr, ev, G = effective_axes(M)
    cum = np.cumsum(ev) / ev.sum()
    idx6 = [names.index(c) for c in SIX]
    pr6, _, _ = effective_axes(M[idx6])

    print(f"=== 1. effective dimensionality ===")
    print(f"  {len(names)} named concepts -> {pr:.2f} effective independent axes")
    print(f"  eigenvalues: {np.round(ev, 2)}")
    print(f"  axes to reach 90% of variance: {int(np.searchsorted(cum, 0.90)) + 1}")
    print(f"  within the 6 planted concepts: {pr6:.2f} effective axes of 6\n")
    top = sorted(itertools.combinations(range(len(names)), 2),
                 key=lambda p: -abs(G[p]))[:6]
    print("  most collinear pairs:")
    for i, j in top:
        print(f"    {names[i]:<12} ~ {names[j]:<12} {G[i, j]:+.3f}")

    # ---- 2. vocabulary ceiling ----
    pairs = json.load(open(a.pairs))
    rng = np.random.default_rng(a.split_seed)
    A, B = {}, {}
    for c in names:
        n = len(pairs[c]); perm = rng.permutation(n)
        A[c] = signature(pairs[c], perm[:n // 2])
        B[c] = signature(pairs[c], perm[n // 2:])
    P = np.stack([A[c] for c in names]); Q = np.stack([B[c] for c in names])
    P = (P - P.mean(0)) / (P.std(0) + 1e-9); Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    S = Pn @ Qn.T
    hits = {}
    print(f"\n=== 2. vocabulary ceiling (self-retrieval from independent text) ===")
    for i, c in enumerate(names):
        j = int(np.argmax(np.abs(S[i]))); hits[c] = bool(j == i)
        print(f"  {c:<12} -> {names[j]:<12} {'OK  ' if j == i else 'MISS'} "
              f"(self {S[i, i]:+.2f}, best {S[i, j]:+.2f})")
    n_ok = sum(hits.values())
    print(f"\n  {n_ok}/{len(names)} concepts retrieve themselves (chance {1/len(names):.3f})")
    print("  A MISS bounds every method scored on these measures; it is not "
          "evidence about the SAE.")

    out = {"concepts": names, "effective_axes_all": pr, "effective_axes_planted_six": pr6,
           "eigenvalues": ev.tolist(),
           "axes_for_90pct_variance": int(np.searchsorted(cum, 0.90)) + 1,
           "max_abs_cos_all": float(max(abs(G[i, j]) for i, j in
                                        itertools.combinations(range(len(names)), 2))),
           "max_abs_cos_planted_six": float(max(abs(G[i, j]) for i, j in
                                                itertools.combinations(idx6, 2))),
           "most_collinear_pairs": [[names[i], names[j], float(G[i, j])] for i, j in top],
           "self_retrieval": hits, "self_retrieval_n_ok": n_ok,
           "self_retrieval_split_seed": a.split_seed}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
