"""Cross-dataset transfer (Sec 5.5, STEP 3): score the tuned PRISM v_pop on the CA
held-out split.

On the exact CA test split exp2 uses (same seed-42 turn-level split, same held-out
pairs, reconstructed via ca_common), report:
  1. held-out acc of the tuned CA v_pop      (sanity: must equal exp2's own output)
  2. held-out acc of the base reward head    (sanity: must equal exp2's own output)
  3. held-out acc of the tuned PRISM v_pop    (new -- the transfer number)
  4. cos(CA v_pop, PRISM v_pop)               (new)
  5. Pearson corr of the two v_pops' pair scores across the pairs (new)

If (1) or (2) disagree with exp2's committed numbers, the split reconstruction is
broken -- we stop and report rather than emit a misleading transfer number.

Both v_pops are used in their natural (data) orientation -- NOT re-oriented to the
head or to each other. Re-orienting a near-orthogonal direction flips its sign on a
coin toss and turns pair accuracy into 1-acc (the exp4 sign-flip bug). Scoring
primitives are the shared ones in sae/experiments/common.py via ca_common.

Writes: results/transfer/summary.json
"""
from __future__ import annotations

import argparse
import json
import os

import ca_common as C
from common import load_basis, shared_direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ca-config", default="CA_K8_seed42_tuned",
                    help="CA fit key in ca_v_pop.pt")
    ap.add_argument("--prism-key", default=None,
                    help="PRISM run_key; default falls back to SAE_RUN_KEY / common default")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = os.path.join(C.RESULTS_DIR, "transfer")
    C.ensure_dirs(out_dir)

    head = C.load_head()
    V_ca, wbar_ca, v_ca, cfg = C.load_ca_basis(args.ca_config)
    _, _, _, test_diffs = C.load_ca_diffs(cfg)

    # tuned PRISM v_pop = unit(V @ wbar), natural (data) orientation (no re-orient).
    V_p, W_p = load_basis(args.prism_key) if args.prism_key else load_basis()
    v_prism = shared_direction(V_p, W_p, head=None)

    acc_ca = C.direction_pair_accuracy(v_ca, test_diffs)
    acc_head = C.direction_pair_accuracy(head, test_diffs)
    acc_prism = C.direction_pair_accuracy(v_prism, test_diffs)

    # --- sanity: reproduce exp2's own CA v_pop and head accuracy exactly ----------
    with open(os.path.join(C.RESULTS_DIR, "exp2", "summary.json")) as f:
        exp2 = json.load(f)
    ref_ca = exp2["directions"]["v_pop"]["overall_pair_acc"]
    ref_head = exp2["directions"]["head"]["overall_pair_acc"]
    d_ca = abs(acc_ca["overall_pair_acc"] - ref_ca)
    d_head = abs(acc_head["overall_pair_acc"] - ref_head)
    passed = d_ca < 1e-6 and d_head < 1e-6

    summary = {
        "ca_config": args.ca_config,
        "prism_key": args.prism_key or os.environ.get("SAE_RUN_KEY", "PART2_K10_seed42_v2"),
        "n_pairs": acc_ca["n_pairs"],
        "n_users": acc_ca["n_users"],
        "acc_ca_vpop": acc_ca["overall_pair_acc"],
        "acc_base_head": acc_head["overall_pair_acc"],
        "acc_prism_vpop_transfer": acc_prism["overall_pair_acc"],
        "cos_ca_vpop_prism_vpop": float(v_ca @ v_prism),
        "score_corr_ca_prism_vpop": C.score_correlation(v_ca, v_prism, test_diffs),
        "sanity": {
            "exp2_acc_ca_vpop": ref_ca, "exp2_acc_head": ref_head,
            "delta_ca_vpop": d_ca, "delta_head": d_head, "passed": passed,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[transfer] wrote {out_dir}/summary.json")
    if not passed:
        raise SystemExit("[transfer] SANITY FAILED: CA v_pop/head acc disagree with "
                         "exp2 -- split reconstruction is broken, do not trust the "
                         "transfer number.")


if __name__ == "__main__":
    main()
