# Planted-direction check

**Owner:** Hassan **Red-teamer:** Harsh
**Workstream:** "Inject a known preference direction; confirm the pipeline recovers it."

Plant a concept we choose, run it through mLoRe, and test whether the SAE can say
which concept it was. Unlike real directions, here the answer is known, so a
confident wrong label is visibly wrong.

```
users who disagree on a known axis -> mLoRe fit -> recovered direction -> SAE readout -> a name
        (we plant the concept here)                                      (we check it here)
```

## Run it

```bash
bash sae/experiments/planted_direction/run.sh all       # everything, ~45 min on CPU
bash sae/experiments/planted_direction/run.sh geometry  # a single stage
```

**CPU is not optional.** `utils.py:14` resolves the device at import and creates
parameters with `device=device`, so `manual_seed` draws a different stream under
CUDA and every fitted number shifts — subspace alignment moved 0.755 → 0.795 on one
GPU run with identical seeds. `run.sh` exports `CUDA_VISIBLE_DEVICES=""` for you.
Do not bypass it.

## Inputs (all pre-existing, none generated here)

| Path | What |
|---|---|
| `data/prism/contrastive_pair_embeddings.pt` | 11 concepts x 50 high/low pairs, Skywork 4096-d |
| `data/prism/contrastive_pairs.json` | the same pairs as raw text |
| `data/prism/concept_vectors.pt` | mean high-minus-low per concept |
| `sae/checkpoints/d3/model.pt` | frozen D3 SAE, dict 16384, k=256 |
| `sae/data/`, `../phase1_artifacts/train_embeddings.pkl` | PRISM responses + text, for feature profiling |

## Stages

| Stage | Script | Produces | What it establishes |
|---|---|---|---|
| `geometry` | `concept_library_geometry.py` | `library_geometry.json` | effective axes in the library; which concepts are reachable at all |
| `plant` | `PRISM/planted_directions.py` | `planted_directions.pt` | labelled group directions — the answer key |
| `readout` | `planted_concept_readout.py` | `planted_readout*.{csv,json}` | Stage 1: identify the concept from the direction |
| `profiles` | `feature_text_profiles.py` | `feature_text_profiles.pt` | per-feature surface meaning from PRISM text |
| `naming` | `planted_concept_naming.py` | `planted_naming*.{csv,json}` | Stage 2: name it from text, vs a no-SAE baseline |
| `circularity` | `planted_circularity_controls.py` | `circularity_controls.json` | disjoint-text control; confidence when the answer is absent |
| `width` | `readout_width_check.py` | `readout_width.json` | readout window, tuned on 5 held-out concepts and frozen |
| `seeds` | `resampled_seeds_check.py` | `resampled_seeds*`, `fixed_split_seeds*` | resampled vs fixed items, at half data |

All outputs land in `results/planted/` (gitignored).

## Two implementation points that will bite a reimplementation

**Never encode a difference vector.** The encoder computes `enc(x - b_pre)` with
`b_pre` the mean training embedding, `||b_pre|| ~ 144`. A pair difference is already
centred near zero, so it reads as roughly `-b_pre` and the top-k selection describes
that offset instead of the preference. Encode each response separately and subtract
the codes. Same caution applies to any unit-norm direction — see
`planted_concept_readout.py:24-27`.

**The readout window is frozen at top-32**, selected in `readout_width_check.py` on
five concepts (helpfulness, factuality, safety, values, sycophancy) that are never
evaluated. Do not re-tune it on the six evaluation concepts.

## Reading the results

The aggregate accuracy is not a stable quantity and should not be quoted on its own.
The library holds ~2.15 effective independent axes across 11 named concepts, so the
unit of analysis is the concept and n is small. Use the per-concept tables in
`resampled_seeds_rows.csv` / `fixed_split_seeds_rows.csv`.
