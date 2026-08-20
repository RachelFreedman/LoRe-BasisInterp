"""Is the naming comparison fair? Match the two routes' information budgets.

Pre-registered before running:
  Testing  -- whether the SAE's loss to the direct-correlation baseline at naming
              survives equalising what the two routes are allowed to see.
  How      -- the original comparison is confounded. The baseline's native output IS
              a 10-measure correlation over all ~20k PRISM texts: lossless. The SAE
              route averaged PRE-COMPUTED per-feature profiles, each built from only
              that feature's top-50 texts, already compressed to 10 numbers, with
              features firing on <50 texts zeroed out -- four losses the baseline
              never pays. Here both routes use the SAME estimator on the SAME texts:
              weight every text, then correlate that weight against each measure.
              The only difference left is how the weight is derived.

                  SAE route   w_t = sum over top-k features of  align_f * z_tf
                  baseline    w_t = v . e_t

  Outcomes -- if the SAE closes the gap or wins, the earlier deficit was an artifact
              of unequal information budgets. If it still loses by a similar margin,
              the deficit is real and the confound was not carrying it.

Window frozen at the value selected in readout_width_check.py. Nothing here is
tuned on the evaluation concepts. CPU-safe; uses CUDA only to encode.
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "PRISM"))
from sae.scripts.feature_text_profiles import PROXIES, prism_texts, load_sae  # noqa: E402


def sign_test(a, b):
    w = sum(x > y for x, y in zip(a, b)); l = sum(x < y for x, y in zip(a, b))
    n = w + l
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, len(a) - n, p


def match(sig, Csig):
    p = sig / (np.linalg.norm(sig) + 1e-9)
    Cn = Csig / (np.linalg.norm(Csig, axis=1, keepdims=True) + 1e-9)
    s = Cn @ p
    i = int(np.argmax(np.abs(s)))
    return i, (1 if s[i] > 0 else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "sae/checkpoints/d3/model.pt"))
    ap.add_argument("--profiles", default=str(REPO / "results/planted/feature_text_profiles.pt"))
    ap.add_argument("--directions", default=str(REPO / "results/planted/planted_directions.pt"))
    ap.add_argument("--sae-data", default=str(REPO / "sae/data"))
    ap.add_argument("--prism-train", default=str(REPO / "PRISM/data/prism/train_embeddings.pkl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--window", type=int, default=256, help="frozen in readout_width_check.py")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", default=str(REPO / "results/planted/matched_naming.json"))
    ap.add_argument("--rows", default=str(REPO / "results/planted/matched_naming_rows.csv"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_sae(Path(a.checkpoint), dev)
    dec = model.decoder.weight.detach().cpu().float()
    blob = torch.load(a.profiles, weights_only=False)
    Csig, names = blob["concept_signatures"], blob["concepts"]
    activity = blob["activity"].numpy() * (np.abs(blob["profiles"]).sum(1) > 0)
    payload = torch.load(a.directions, weights_only=False)
    EVAL = payload["concepts"]

    meta = [json.loads(l) for l in open(Path(a.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == a.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    X = torch.load(Path(a.sae_data) / f"sae_{a.split}.pt", map_location="cpu").float()
    texts = prism_texts(Path(a.prism_train), meta, a.split)
    keep = [i for i, t in enumerate(texts) if t]
    X = X[keep]
    P = np.array([[PROXIES[q](texts[i]) for q in PROXIES] for i in keep], dtype=np.float32)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
    print(f"{len(keep)} PRISM texts, {len(PROXIES)} measures, window top-{a.window}")

    # encode once; TopK gives k nonzeros per row, so store sparse and reuse
    rows, cols, vals = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), a.batch_size):
            z = model.encode(X[i:i + a.batch_size].to(dev)).cpu()
            nz = z.nonzero(as_tuple=False)
            rows.append(nz[:, 0].numpy() + i); cols.append(nz[:, 1].numpy())
            vals.append(z[nz[:, 0], nz[:, 1]].numpy())
    Z = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(X), dec.shape[1]))
    print(f"activation matrix: {Z.nnz} nonzeros ({Z.nnz/len(X):.0f} per text)")

    def fingerprint(w):
        w = (w - w.mean()) / (w.std() + 1e-9)
        return (Pz * w[:, None]).mean(0)

    out_rows, sae_acc, base_acc = [], [], []
    for rec in payload["records"]:
        sh = bh = 0
        for g, (tc, ts) in enumerate(rec["labels"]):
            v = F.normalize(rec["group_dirs"][g], dim=0)
            align = (dec.T @ v).numpy()
            top = np.argsort(np.abs(align) * activity)[::-1][:a.window]
            w_sae = np.asarray(Z[:, top] @ align[top]).ravel()   # feature-mediated weight
            w_base = (X @ v).numpy()                             # direct weight
            si, ss = match(fingerprint(w_sae), Csig)
            bi, bs = match(fingerprint(w_base), Csig)
            sok = int(names[si] == tc and ("high" if ss > 0 else "low") == ts)
            bok = int(names[bi] == tc and ("high" if bs > 0 else "low") == ts)
            sh += sok; bh += bok
            out_rows.append({"seed": rec["seed"], "true_concept": tc, "true_sign": ts,
                             "sae_pred": names[si], "nosae_pred": names[bi],
                             "sae_correct": sok, "nosae_correct": bok})
        n = len(rec["labels"])
        sae_acc.append(sh / n); base_acc.append(bh / n)
        print(f"   seed {rec['seed']}: SAE {sae_acc[-1]:.3f}   no-SAE {base_acc[-1]:.3f}")

    W, L, T, p = sign_test(sae_acc, base_acc)
    per = {}
    for r in out_rows:
        d = per.setdefault(r["true_concept"], [0, 0, 0])
        d[0] += r["sae_correct"]; d[1] += r["nosae_correct"]; d[2] += 1
    with open(a.rows, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); wr.writeheader(); wr.writerows(out_rows)
    json.dump({"window": a.window, "estimator": "identical: correlate text weight with each measure",
               "sae_mean": float(np.mean(sae_acc)), "nosae_mean": float(np.mean(base_acc)),
               "sae_per_seed": sae_acc, "nosae_per_seed": base_acc,
               "sign_test": {"wins": W, "losses": L, "ties": T, "p": p},
               "per_concept": {k: {"sae": v[0], "nosae": v[1], "n": v[2]} for k, v in per.items()}},
              open(a.out, "w"), indent=2)

    print(f"\n=== matched information budget, {len(sae_acc)} seeds ===")
    print(f"  SAE     {np.mean(sae_acc):.3f} +/- {np.std(sae_acc):.3f}")
    print(f"  no-SAE  {np.mean(base_acc):.3f} +/- {np.std(base_acc):.3f}")
    print(f"  sign test: {W}W-{L}L-{T}T   p = {p:.3f}")
    print("\n  per concept (SAE / no-SAE of n):")
    for c in EVAL:
        if c in per: print(f"    {c:<12} {per[c][0]:>3} / {per[c][1]:>3}  of {per[c][2]}")


if __name__ == "__main__":
    main()
