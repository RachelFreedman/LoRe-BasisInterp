"""Does the naming null come from the readout being too wide?

Pre-registered (Discord, before running):
  Testing  -- whether the Stage-2 null is caused by the readout averaging too many
              features rather than by the SAE lacking the information.
  How      -- the readout window is chosen on FIVE library concepts that are never
              evaluated (helpfulness, factuality, safety, values, sycophancy), then
              frozen. AMENDED before the real run: the tuning objective is the cosine
              between the predicted and true fingerprint, not accuracy. Accuracy on
              the tuning concepts is 0.000 at every window (none of them is
              verbosity-dominated), so it carries no signal to select on. Cosine is
              continuous and still never touches the evaluation concepts. The original six are then planted at 10 seeds and scored at
              that frozen window against the direct-correlation baseline and a null.
  Outcomes -- SAE gains on the baseline => the null was a readout artifact.
              No movement => the original null stands, one mechanism ruled out.

The tune/eval separation is enforced here in code, not by promise: the evaluation
concepts are never scored during the sweep. top_k=256 (the original setting) is
reported alongside so the comparison is like-for-like at the same seed count.

CPU only: utils.py fixes its device at import, so CUDA changes the init stream.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "PRISM"))

from PRISM.synthetic_concept_users import build_users, set_seed, signed_ground_truth, unit  # noqa
from PRISM.planted_directions import group_mean_directions  # noqa
from PRISM.random_baseline import random_unit_directions  # noqa
from sae.scripts.feature_text_profiles import PROXIES, prism_texts, load_sae, encode  # noqa

TUNE = ["helpfulness", "factuality", "safety", "values", "sycophancy"]
EVAL = ["fluency", "repetition", "confidence", "creativity", "formatting", "diversity"]
WINDOWS = [8, 16, 32, 64, 128, 256]


def plant(emb, concepts, cvs, seed, a):
    from utils import LoReV2
    V = torch.stack([cvs[c].float() for c in concepts], dim=1)
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tf, _, gid = build_users(emb, concepts, a.users_per_group, a.test_frac, gen,
                             a.high_frac, a.subset_frac)
    m = LoReV2(len(tf), V.shape[0], 2 * len(concepts), lam_pop=a.lam_pop, lam_d=a.lam_d,
               num_iterations=a.iters, learning_rate=a.lr, verbose=False)
    m.train(tf)
    return group_mean_directions(m.reward_dirs().detach().cpu().T, gid, 2 * len(concepts))


def match(sig, Csig):
    """Nearest (concept, sign) by cosine over the surface-measure fingerprint."""
    p = sig / (np.linalg.norm(sig) + 1e-9)
    Cn = Csig / (np.linalg.norm(Csig, axis=1, keepdims=True) + 1e-9)
    s = Cn @ p
    i = int(np.argmax(np.abs(s)))
    return i, (1 if s[i] > 0 else -1)


def sae_sig(v, dec, P, w, k):
    align = (dec.T @ F.normalize(v, dim=0)).numpy()
    top = np.argsort(np.abs(align) * w)[::-1][:k]
    return align[top] @ P[top]


def tune_fit(dirs, concepts, names, Csig, sigfn):
    """Selection objective: mean cosine between predicted and TRUE signed fingerprint.

    Continuous, unlike argmax accuracy, which is identically zero on the tuning
    concepts at every window and so cannot select anything.
    """
    out = []
    for g in range(len(dirs)):
        true = Csig[names.index(concepts[g // 2])] * (1 if g % 2 == 0 else -1)
        pred = sigfn(dirs[g])
        out.append(float(pred @ true / ((np.linalg.norm(pred) + 1e-9) *
                                        (np.linalg.norm(true) + 1e-9))))
    return float(np.mean(out))


def acc(dirs, concepts, names, Csig, sigfn):
    hit = 0
    for g in range(len(dirs)):
        c, s = match(sigfn(dirs[g]), Csig)
        true_c, true_s = concepts[g // 2], (1 if g % 2 == 0 else -1)
        hit += int(names[c] == true_c and s == true_s)
    return hit / len(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(REPO / "data/prism/contrastive_pair_embeddings.pt"))
    ap.add_argument("--concept_vectors", default=str(REPO / "data/prism/concept_vectors.pt"))
    ap.add_argument("--checkpoint", default=str(REPO / "sae/checkpoints/d3/model.pt"))
    ap.add_argument("--profiles", default=str(REPO / "results/planted/feature_text_profiles.pt"))
    ap.add_argument("--sae-data", default=str(REPO / "sae/data"))
    ap.add_argument("--prism-train", default=str(REPO / "PRISM/data/prism/train_embeddings.pkl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--tune-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--users_per_group", type=int, default=20)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--subset_frac", type=float, default=0.7)
    ap.add_argument("--high_frac", type=float, default=0.5)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default=str(REPO / "results/planted/readout_width.json"))
    a = ap.parse_args()

    device = torch.device("cpu")
    emb = torch.load(a.emb, weights_only=False)
    cvs = torch.load(a.concept_vectors, weights_only=False)
    model, cfg = load_sae(Path(a.checkpoint), device)
    dec = model.decoder.weight.detach().cpu().float()
    blob = torch.load(a.profiles, weights_only=False)
    P, Csig, names = blob["profiles"], blob["concept_signatures"], blob["concepts"]
    w = blob["activity"].numpy() * (np.abs(P).sum(1) > 0)
    n_groups = 2 * len(names)
    print(f"candidates: {len(names)} concepts x 2 signs = {n_groups} (chance {1/n_groups:.3f})\n")

    # ---------- stage 1: choose the window on the TUNING concepts only ----------
    print(f"sweeping window on tuning concepts (never evaluated): {TUNE}")
    tune_dirs = [plant(emb, TUNE, cvs, s, a) for s in a.tune_seeds]
    sweep, sweep_acc = {}, {}
    for k in WINDOWS:
        sweep[k] = float(np.mean([tune_fit(d, TUNE, names, Csig,
                                           lambda v, k=k: sae_sig(v, dec, P, w, k))
                                  for d in tune_dirs]))
        sweep_acc[k] = float(np.mean([acc(d, TUNE, names, Csig,
                                          lambda v, k=k: sae_sig(v, dec, P, w, k))
                                      for d in tune_dirs]))
        print(f"   top{k:<4} fingerprint-cos {sweep[k]:+.4f}   (acc {sweep_acc[k]:.3f})")
    best = max(sweep, key=sweep.get)
    print(f"\nFROZEN window = top{best} (chosen without touching {EVAL})\n")

    # ---------- no-SAE baseline machinery ----------
    meta = [json.loads(l) for l in open(Path(a.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == a.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    x = torch.load(Path(a.sae_data) / f"sae_{a.split}.pt", map_location="cpu").float()
    texts = prism_texts(Path(a.prism_train), meta, a.split)
    keep = [i for i, t in enumerate(texts) if t]
    X = x[keep]
    Pr = np.array([[PROXIES[q](texts[i]) for q in PROXIES] for i in keep], dtype=np.float32)
    Pz = (Pr - Pr.mean(0)) / (Pr.std(0) + 1e-9)

    def nosae_sig(v):
        s = (X @ F.normalize(v, dim=0)).numpy()
        s = (s - s.mean()) / (s.std() + 1e-9)
        return (Pz * s[:, None]).mean(0)

    # ---------- stage 2: evaluate on the held-out concepts ----------
    print(f"evaluating on {EVAL}, {len(a.eval_seeds)} seeds")
    frozen, orig, base = [], [], []
    for s in a.eval_seeds:
        d = plant(emb, EVAL, cvs, s, a)
        frozen.append(acc(d, EVAL, names, Csig, lambda v: sae_sig(v, dec, P, w, best)))
        orig.append(acc(d, EVAL, names, Csig, lambda v: sae_sig(v, dec, P, w, 256)))
        base.append(acc(d, EVAL, names, Csig, nosae_sig))
        print(f"   seed {s}: SAE@top{best} {frozen[-1]:.3f}   SAE@top256 {orig[-1]:.3f}   "
              f"no-SAE {base[-1]:.3f}")

    nulls = torch.stack(random_unit_directions(dec.shape[0], a.n_null, seed=0))
    truth = [(EVAL[g // 2], 1 if g % 2 == 0 else -1) for g in range(2 * len(EVAL))]
    npred = [match(sae_sig(v, dec, P, w, best), Csig) for v in nulls]
    null_sae = float(np.mean([[int(names[c] == tc and s == ts) for tc, ts in truth]
                              for c, s in npred]))

    out = {"frozen_window": best, "tuning_concepts": TUNE, "tuning_sweep_cosine": sweep, "tuning_sweep_accuracy": sweep_acc,
           "tuning_objective": "mean cosine to true signed fingerprint (accuracy is "
                               "identically 0 on tuning concepts, cannot select)",
           "eval_concepts": EVAL, "eval_seeds": a.eval_seeds, "chance": 1 / n_groups,
           "null_sae_at_frozen": null_sae,
           "sae_frozen_mean": float(np.mean(frozen)), "sae_frozen_std": float(np.std(frozen)),
           "sae_top256_mean": float(np.mean(orig)), "sae_top256_std": float(np.std(orig)),
           "nosae_mean": float(np.mean(base)), "nosae_std": float(np.std(base)),
           "sae_frozen_per_seed": frozen, "sae_top256_per_seed": orig,
           "nosae_per_seed": base}
    wins = sum(f > b for f, b in zip(frozen, base))
    out["seeds_sae_beats_baseline"] = f"{wins}/{len(frozen)}"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n=== held-out concepts, {len(a.eval_seeds)} seeds ===")
    print(f"  chance                 {1/n_groups:.3f}")
    print(f"  null (SAE@top{best})      {null_sae:.3f}")
    print(f"  SAE @ top256 (original) {np.mean(orig):.3f} +/- {np.std(orig):.3f}")
    print(f"  SAE @ top{best} (frozen)   {np.mean(frozen):.3f} +/- {np.std(frozen):.3f}")
    print(f"  no-SAE baseline         {np.mean(base):.3f} +/- {np.std(base):.3f}")
    print(f"  seeds where SAE beats baseline: {wins}/{len(frozen)}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
