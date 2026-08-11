# Red-team: LoReV2 on Community Alignment

Ifesi — branch `prism-dataset-analysis`

**Target claim (Mohamed's).** LoReV2 beats the base Skywork RM by ~+0.03 on Community Alignment, and that gain is a single shared population direction (`wbar`) — the per-user personalization deltas add nothing. "Individuals don't matter much here."

All runs: LoReV2, rank 8, 5 seeds, leak-free turn-level split, 200 users (median 180 pairs/user). Producing script: `PRISM/community_alignment_lore.py`. Full run log: `results/community_alignment/redteam_sweep.log`.

## What was done

- **Experiment 1 — Reproduction.** Re-ran his exact config (lr 1e-4, lam_pop 0.01, lam_d 10) over ranks 1/5/8/10/20.
- **Experiment 2 — lam_d sweep.** Swept the delta ridge penalty across 0, 1, 2, 4, 6, 8, 10, 14, 16 at rank 8. Tests whether his lam_d=10 was cherry-picked to crush the deltas (making the "zeroing deltas costs nothing" ablation circular).
- **Experiment 3 — High-volume-users cut.** Restricted to users with ≥200 and ≥250 pairs each. Tests the one remaining escape hatch: maybe individuals matter but 180 pairs/user is too little to learn them.

## Results

- **Reproduction matched.** test_acc − base_rm = **+0.0333 ± 0.0020** at rank 8 (his +0.030). `personal − global = −0.0003`; zeroing deltas (`wbar_only`) ≈ test_acc. Faithful.
- **lam_d doesn't matter.** test_acc flat at ~0.670 across all 9 values. Delta contribution `test_acc − wbar_only` stays within noise everywhere — **including lam_d=0** (ridge fully off): −0.0012 ± 0.0024. → My "cherry-picked lam_d" hypothesis is **disproved**; his choice is on a flat surface but harmless.
- **Data volume doesn't matter.** `personal − global`: base −0.0003, min_pairs=200 **−0.0007 ± 0.0042**, min_pairs=250 **+0.0023 ± 0.0061** (std > mean). Even data-richest users (median 279 pairs) show no personalization gain over the shared direction. → "Not enough data" escape hatch **closed**.
- **Loose end (not a rebuttal).** `personal − other_user` stays positive (+0.013 to +0.030) throughout — user identity isn't literally random, but that signal never converts into beating the shared direction.

**Verdict:** on Community Alignment, the target claim survived reproduction, a lam_d sweep, and a data-volume cut. No within-dataset experiment left that plausibly overturns it. Next challenge would be an independent dataset (MultiPref).

## Artifacts

| Experiment | Result files | Script |
|---|---|---|
| 1. Reproduction | `results/community_alignment/repro_lamd10.csv` | `PRISM/community_alignment_lore.py` |
| 2. lam_d sweep | `results/community_alignment/lamd_{0,1,2,4,6,8,10,14,16}.csv` | `PRISM/community_alignment_lore.py` (`--lam_d`) |
| 3. High-volume cut | `results/community_alignment/highvol_min{200,250}.csv` | `PRISM/community_alignment_lore.py` (`--min_pairs`) |
| Full run log (all 3) | `results/community_alignment/redteam_sweep.log` | — |

Key CSV columns: `test_acc`, `wbar_only` (deltas zeroed), `base_rm`/`global`/`personal`/`other_user` (reference directions), `min_abs_basis_cos` (delta collapse).
