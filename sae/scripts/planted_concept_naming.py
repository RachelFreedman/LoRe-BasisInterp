#!/usr/bin/env python3
"""
Stage 2 (rebuilt): can the SAE NAME the concept a direction was planted from, when
nothing in the scoring path knows what the concepts are?

The chain, and why each link is clean:

    direction -> features        decoder alignment. Geometry only.
    feature   -> surface meaning top-activating PRISM responses, scored with a priori
                                 text measures (feature_text_profiles.py). PRISM is
                                 the SAE's training distribution and shares nothing
                                 with the LLM-written concept contrast sets.
    concept   -> surface meaning mean high-minus-low delta over the same measures,
                                 from the contrast TEXT. No embeddings, no concept
                                 vectors, no SAE.

The first version of this test failed on exactly this point: it described features
using the concept contrast sets themselves, so identification could be done -- better
-- by plain cosine to the concept vectors (1.000 vs the SAE's 0.778). Nothing in the
chain above passes through a concept vector, so the SAE has to carry the answer.

Baselines reported alongside, because the number is meaningless without them:

  null      random unit directions. Should sit at chance.
  no-SAE    score PRISM responses by v directly and correlate with each surface
            measure. Same texts, same measures, no sparse decomposition -- the
            reward-lens move. If this matches or beats the SAE, the SAE is not
            adding interpretive value, and that is the honest finding to report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "PRISM"))

from PRISM.random_baseline import random_unit_directions  # noqa: E402
from sae.src.io import write_json  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402
from sae.scripts.feature_text_profiles import (PROXIES, choose_device,  # noqa: E402
                                               encode, load_sae, prism_texts)


def match(pred_sig: np.ndarray, concept_sig: np.ndarray) -> tuple[int, int, float]:
    """Nearest (concept, sign) by cosine. Returns (concept idx, sign, margin).

    Sign is part of the answer: a low-preferring group's direction must match the
    NEGATED concept signature, so a model that finds the right concept with the wrong
    polarity is scored wrong.
    """
    p = pred_sig / (np.linalg.norm(pred_sig) + 1e-9)
    c = concept_sig / (np.linalg.norm(concept_sig, axis=1, keepdims=True) + 1e-9)
    sims = c @ p                                              # [n_concepts], signed
    idx = int(np.abs(sims).argmax())
    ordered = np.sort(np.abs(sims))[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
    return idx, (1 if sims[idx] > 0 else -1), margin


def sae_signature(v, dec, profiles, weight, top_k):
    """Surface signature the SAE predicts for direction v."""
    align = dec.T @ v
    score = align.abs().numpy() * weight
    top = np.argsort(score)[::-1][:top_k]
    return align.numpy()[top] @ profiles[top]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    ap.add_argument("--profiles", default="results/planted/feature_text_profiles.pt")
    ap.add_argument("--directions", default="results/planted/planted_directions.pt")
    ap.add_argument("--sae-data", default="sae/data")
    ap.add_argument("--prism-train", default="PRISM/data/prism/train_embeddings.pkl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--results-dir", default="results/planted")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import json
    device = choose_device(args.device)
    model, cfg = load_sae(Path(args.checkpoint), device)
    prof_blob = torch.load(args.profiles, weights_only=False)
    profiles = prof_blob["profiles"]                            # [dict, n_proxies]
    concept_sig = prof_blob["concept_signatures"]               # [n_concepts, n_proxies]
    concepts = prof_blob["concepts"]
    payload = torch.load(args.directions, weights_only=False)
    dec = model.decoder.weight.detach().cpu().float()

    profiled = np.abs(profiles).sum(1) > 0
    weight = prof_blob["activity"].numpy() * profiled           # unprofiled features excluded
    n_groups = 2 * len(concepts)
    chance = 1.0 / n_groups
    print(f"identification over {len(concepts)} concepts x 2 signs = {n_groups} groups "
          f"(chance {chance:.3f})")
    print(f"{int(profiled.sum())} profiled features, top_k={args.top_k}\n")

    # ---- no-SAE baseline: direction scores vs surface measures, over PRISM text ----
    meta = [json.loads(line) for line in open(Path(args.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == args.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    x = torch.load(Path(args.sae_data) / f"sae_{args.split}.pt", map_location="cpu").float()
    texts = prism_texts(Path(args.prism_train), meta, args.split)
    keep = [i for i, t in enumerate(texts) if t]
    X = x[keep]
    P = np.array([[PROXIES[k](texts[i]) for k in PROXIES] for i in keep], dtype=np.float32)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)

    def nosae_signature(v):
        s = (X @ v).numpy()
        s = (s - s.mean()) / (s.std() + 1e-9)
        return (Pz * s[:, None]).mean(0)                       # correlation per measure

    def run(dirs, labels):
        rows, sae_hits, nosae_hits = [], [], []
        for i, v in enumerate(dirs):
            v = F.normalize(v, dim=0)
            ci, si, mg = match(sae_signature(v, dec, profiles, weight, args.top_k), concept_sig)
            nci, nsi, _ = match(nosae_signature(v), concept_sig)
            row = {"sae_concept": concepts[ci], "sae_sign": "high" if si > 0 else "low",
                   "sae_margin": round(mg, 4),
                   "nosae_concept": concepts[nci], "nosae_sign": "high" if nsi > 0 else "low"}
            if labels is not None:
                tc, ts = labels[i]
                row |= {"true_concept": tc, "true_sign": ts,
                        "sae_correct": int(concepts[ci] == tc and row["sae_sign"] == ts),
                        "nosae_correct": int(concepts[nci] == tc and row["nosae_sign"] == ts)}
                sae_hits.append(row["sae_correct"])
                nosae_hits.append(row["nosae_correct"])
            rows.append(row)
        return rows, (np.mean(sae_hits) if sae_hits else None), \
            (np.mean(nosae_hits) if nosae_hits else None)

    # ---- planted directions ----
    all_rows, sae_accs, nosae_accs = [], [], []
    for rec in payload["records"]:
        labels = [(c, s) for c, s in rec["labels"]]
        rows, sa, na = run(rec["group_dirs"], labels)
        for r in rows:
            r["seed"] = rec["seed"]
        all_rows += rows
        sae_accs.append(sa)
        nosae_accs.append(na)

    # ---- null ----
    null_dirs = torch.stack(random_unit_directions(dec.shape[0], args.n_null, seed=0))
    null_rows, _, _ = run(null_dirs, None)
    truth = [(c, s) for c, s in payload["records"][0]["labels"]]
    null_sae = float(np.mean([[int(r["sae_concept"] == tc and r["sae_sign"] == ts)
                               for tc, ts in truth] for r in null_rows]))
    null_nosae = float(np.mean([[int(r["nosae_concept"] == tc and r["nosae_sign"] == ts)
                                 for tc, ts in truth] for r in null_rows]))

    print("=== can it name the planted concept? ===")
    print(f"  chance                {chance:.3f}")
    print(f"  null   SAE {null_sae:.3f}   no-SAE {null_nosae:.3f}   ({args.n_null} draws)")
    print(f"  SAE    {np.mean(sae_accs):.3f} +/- {np.std(sae_accs):.3f}   "
          f"per seed " + ", ".join(f"{a:.3f}" for a in sae_accs))
    print(f"  no-SAE {np.mean(nosae_accs):.3f} +/- {np.std(nosae_accs):.3f}   "
          f"per seed " + ", ".join(f"{a:.3f}" for a in nosae_accs))
    verdict = ("SAE adds interpretive value over the direct correlation"
               if np.mean(sae_accs) > np.mean(nosae_accs) + 0.05 else
               "SAE does NOT beat the direct correlation -- report that, it is the "
               "honest result and it is what a reviewer will check")
    print(f"\n  -> {verdict}")

    print("\n  errors (SAE):")
    for r in all_rows:
        if not r["sae_correct"]:
            print(f"    seed {r['seed']}  {r['true_concept']}/{r['true_sign']:<4} -> "
                  f"{r['sae_concept']}/{r['sae_sign']:<4}  margin {r['sae_margin']:.3f}")

    out = Path(args.results_dir)
    with (out / "planted_naming.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    write_json(out / "planted_naming_summary.json", {
        "chance": chance, "n_groups": n_groups,
        "null_sae": null_sae, "null_nosae": null_nosae,
        "sae_accuracy_mean": float(np.mean(sae_accs)), "sae_per_seed": sae_accs,
        "nosae_accuracy_mean": float(np.mean(nosae_accs)), "nosae_per_seed": nosae_accs,
        "top_k": args.top_k, "profiled_features": int(profiled.sum()),
        "note": ("Feature meanings derived from PRISM text only; concept signatures "
                 "from contrast text only. No concept vectors or embeddings in the "
                 "scoring path."),
    })
    print(f"\nWrote {out}/planted_naming.csv, planted_naming_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
