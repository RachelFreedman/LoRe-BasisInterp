# Checkpoint — basis collapse investigation

Date: 2026-07-22
Repo: LoRe-BasisInterp (this repo)

## TL;DR

The SAE "smearing / can't decompose bases" blocker is **not an SAE problem**. It
traces upstream to a **degenerate basis fit**: our PRISM basis matrix `V` is
effectively **rank 1** — every basis points in the same direction and every user
loads onto the same single basis. The SAE reconstruction itself is fine
(explained variance 0.99). The classmate's 6% "smear" is just the downstream
symptom of there being only ~1 real reward direction to attribute features to.

## What we measured (this repo)

File: `PRISM/basis_matrices.pt`, run key `PART2_K10_seed42` (V is 4096x10, W is 1029x10).

- `V` singular values: `[4490, 14, 0.9, 0.9, ...noise...]` → **rank ~1** (s2/s1 = 0.003).
- Basis-basis |cos| = **1.00** (all 10 basis columns are the *same direction*).
- Only **3 of 10** bases have any user weight; `bases_kept = 3`.
- **1025 of 1029 users** have basis #1 as their dominant basis.
- **99.2%** of user pairs have |cos| > 0.5 (everyone is basically identical).
- Weights `W` are ~**0/1** (unit-like), not small signed values.
- Collapse holds across **every run/seed** in the file (K=5, 10, 20); bases_kept = 3–6.

Reproduce with: SVD on `matrices[run_key]['V']` + per-basis max|W| + user-user cosine.

## The healthy reference (mentor's APA repo — do NOT switch yet)

Files: `/Users/ifesionubogu/rlhf/APA/apa/experiments/checkpoints/`
  - `V_K8.pt` (4096 x 8), `W_seen_K8.pt` (182 x 8)

Same diagnostics on hers show a **healthy, non-collapsed** fit:
- V singular values all ~0.63–0.67 → all 8 bases equal energy (s2/s1 = 0.99).
- Basis-basis |cos| ≈ 0.01 (near-orthogonal, distinct directions).
- All **8 of 8** bases used; users spread across all bases (dominant counts ~16–28).
- **16.8%** of user pairs |cos| > 0.5 (diverse population) — matches her reported ~17%.
- Weights small and **signed** (~±0.02), not unit vectors.

Mentor's note: our weights are "unit vectors, not signed like you said on the call
— possibly a much earlier LoRe implementation." Her weights are NOT necessarily her
final-analysis weights; she said to reproduce similarly-performing weights ourselves
before doing much analysis, but the patterns are informative.

## Open questions to investigate

1. **Did the original authors see this collapse too**, or is it specific to *my* setup?
   - Check whether the collapse is baked into the committed `basis_matrices.pt`
     (i.e. whoever generated it) vs introduced by how I ran/regenerated things.
   - Compare the basis-training code path here (`PRISM/train_basis.py`) against the
     APA implementation to find the algorithmic difference.
2. **Why 0/1 unit-like weights instead of small signed weights?** Likely a
   normalization / regularization / init / parametrization choice in
   `PRISM/train_basis.py` driving all bases onto the mean reward direction.
3. Once we understand it: **reproduce healthy (non-collapsed) weights** like the
   mentor's, then re-run SAE attribution against *those*. Do NOT change the SAE
   architecture (Matryoshka etc. is overkill for a rank-1 target).

## Next step

Switch (together) into the freshly-cloned **APA** repo to compare its LoRe/basis
training against `PRISM/train_basis.py` and determine whether the collapse is a
setup issue on our side or something the original reporting also exhibited.
