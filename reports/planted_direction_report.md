# Planted-Direction Check: can the SAE name a concept we injected?

**Owner:** Hassan **Red-teamer:** Harsh

## 1. Pre-registration

**Testing:** whether the SAE can name a preference direction whose true concept we planted ourselves, using synthetic users who disagree along six known concept axes over real Skywork embeddings.

**How:** plant a concept → mLoRe fit → recovered group direction → SAE readout, scored as forced choice over (concept, sign) against a random-direction null and a ground-truth-vector ceiling. Design committed in Discord on 14/08 before running: drop `test_acc`, a random baseline beside every alignment number, the ceiling control, and a guard against circularity. It was not posted in the 2–4 sentence format — that is a process failure on my side, noted here rather than backfilled.

**Outcomes:** above null with a healthy ceiling ⇒ SAE decomposition reads concept content off reward directions. At chance with a healthy ceiling ⇒ the SAE is the broken link. Poor ceiling ⇒ the fit did not preserve enough to test the SAE.

*Added after seeing Stage 1 (declared exploratory):* a no-SAE baseline — correlate the direction's response scores against the same surface measures, the reward-lens move — and a calibration check that removes the correct answer from the candidate set. Both were later re-run under pre-registered protocols (§2.3, §2.4).

## 2. Result

**Headline:** the SAE recovers a planted concept far above chance, but naming it from text is dominated by a shared length component, and no comparison against a no-SAE baseline can be settled on this library because it holds only ~2.15 effective independent axes.

### 2.1 Identification works

| Direction given to the readout | Accuracy |
|---|---:|
| recovered (planted concept) | **0.778 ± 0.079** |
| true concept vector (ceiling) | 0.667 |
| random directions (null) | 0.083 |
| chance (1/12) | 0.083 |

Nine times chance, null exactly on chance. Ceiling sits marginally below the recovered condition — one to two items at n=12, within noise.

### 2.2 Naming from text: complementary, not ranked

Both routes produce a fingerprint over ten countable text measures, then match it to the nearest concept fingerprint. The only difference is whether the direction is decomposed into sparse features first.

Per-concept, out of 20 (10 seeds × 2 signs), at the frozen top-32 window:

| concept | SAE | no-SAE | resampled items | fixed items |
|---|---:|---:|---:|---:|
| repetition | **18 / 20** | 12 / 7 | 18 | 20 |
| formatting | **5 / 8** | 0 / 0 | 5 | 8 |
| confidence | 0 / 0 | **15 / 18** | 0 | 0 |
| diversity | 14–18 | **20 / 20** | 18 | 14 |
| fluency | 0 | 0 | 0 | 0 |
| creativity | 0 | 0 | 0 | 0 |

The SAE is the only route that ever surfaces formatting; the direct correlation is the only route that surfaces confidence. **Replicated in three independent runs.** Aggregates hide this: they differ by ~0.05 while the per-concept behaviour is opposite.

### 2.3 The aggregate cannot be settled

| Setting | SAE | no-SAE | paired sign test |
|---|---:|---:|---|
| 3 seeds, top-256, full data | 0.389 | 0.417 | — |
| 10 seeds, top-32, full data | 0.483 | 0.358 | 9W-1L, p = 0.021 |
| 10 seeds, top-32, half data, fixed items | 0.350 | 0.375 | 2W-4L-4T, p = 0.688 |
| 10 seeds, top-32, half data, resampled items | 0.342 | 0.392 | 2W-5L-3T, p = 0.453 |

Fixed-half and resampled-half agree, so the collapse is **data volume, not item resampling**. With 50 contrast pairs per concept, validating item-robustness requires splitting, which halves volume — the two cannot be separated on this dataset.

The p = 0.021 is not valid regardless: seeds resample the model, not the data. With **concept** as the unit — the correct one — the result is **2W-2L-2T, p = 1.000**. Accuracy spread across concepts is 0.00–0.90; across seeds, 0.17–0.50.

### 2.4 The readout cannot tell when it is wrong

Removing the true concept from the candidate set, against removing a random other concept so both score over equal columns:

| Condition | Mean margin |
|---|---:|
| a random other concept removed (answer available) | 0.406 |
| the true concept removed (answer unavailable) | 0.392 |

Confidence falls 3.4%. It does not abstain and does not signal anything. **On real directions a confident SAE label is not evidence the label is correct.**

### 2.5 Mechanism, and the ceiling that bounds it

Every predicted fingerprint returns dominated by verbosity — `avg_word_len`, `sent_count`, `length`, `ttr` — regardless of input. Formatting's true fingerprint is markdown (+4.24) and confidence's is hedging (−4.68), and neither ever surfaces.

Why: the 11 concept vectors hold **2.15 effective independent axes**, one eigenvalue carrying 7.32 of 11. Each concept vector is mostly one shared "generic good response" direction plus a small distinctive remainder — and the remainder is exactly the discriminating marker. Within the six planted concepts, 3.01 effective axes of 6.

The measurement vocabulary is *not* the bottleneck for the concepts that fail. Fingerprinting each concept from one half of its contrast text and retrieving it from the other half succeeds for **7 of 11**, including formatting (+0.95) and confidence (+0.99). Genuinely unreachable: creativity (+0.30), factuality (+0.33), and safety↔values (collinear at 0.983).

Circularity is ruled out: planting on one half of the text and scoring on the other gives 0.722, against 0.694 with shared items.

## 3. Testing info

**Branch:** `hassan/sae-planted-concept` **Runbook:** `sae/experiments/planted_direction/README.md`

```bash
bash sae/experiments/planted_direction/run.sh all     # ~45 min, CPU
```

| Stage | Script | Result files (under `results/planted/`) |
|---|---|---|
| library geometry + ceiling | `sae/scripts/concept_library_geometry.py` | `library_geometry.json` |
| plant directions | `PRISM/planted_directions.py` | `planted_directions.pt` |
| Stage 1 identification | `sae/scripts/planted_concept_readout.py` | `planted_readout.csv`, `planted_readout_summary.json` |
| feature text profiles | `sae/scripts/feature_text_profiles.py` | `feature_text_profiles.pt` |
| Stage 2 naming | `sae/scripts/planted_concept_naming.py` | `planted_naming.csv`, `planted_naming_summary.json` |
| circularity + calibration | `sae/scripts/planted_circularity_controls.py` | `circularity_controls.json` |
| readout width | `sae/scripts/readout_width_check.py` | `readout_width.json` |
| seed/item resampling | `sae/scripts/resampled_seeds_check.py` | `resampled_seeds*`, `fixed_split_seeds*` |

**Data:** `data/prism/{contrastive_pair_embeddings.pt, contrastive_pairs.json, concept_vectors.pt}`, frozen SAE `sae/checkpoints/d3/model.pt` (16384 / k=256), PRISM responses via `sae/data/` + `phase1_artifacts/train_embeddings.pkl`.

**CPU only.** `utils.py:14` fixes the device at import, so CUDA draws a different init stream — subspace alignment moved 0.755 → 0.795 on one GPU run at identical seeds. `run.sh` enforces this.

**Frozen parameter:** readout window top-32, selected in `readout_width_check.py` on five concepts never evaluated (helpfulness, factuality, safety, values, sycophancy), by fingerprint cosine to ground truth. Accuracy on those five is 0.000 at every window — they are largely the collinear cluster — so accuracy could not serve as the selection objective; this amendment was made before the evaluation run.

## 4. Alternative explanations, ruled out

**"The hit is circular — planting and scoring share the same text."** Disjoint halves give 0.722 vs 0.694 shared. Ruled out.

**"The scoring path leaks."** Random directions land at 0.083 against a chance of 0.083, and 0.055–0.061 in the naming runs. Ruled out.

**"The concepts are just unmeasurable."** No: 7 of 11 retrieve themselves from independent text, and both concepts that fail most cleanly — formatting, confidence — are among the strongest. Ruled out for those; genuinely true for creativity, factuality, safety/values.

**"More seeds would settle the aggregate."** No: concept identity explains 0.00–0.90 of the spread, seeds 0.17–0.50. Seeds are pseudo-replication of six concepts.

**"Extend to all 11 concepts for power."** Not available: every held-out concept sits at |cos| ≥ 0.870 to something already in the set.

**Still open — the naming null is confounded by the length component.** Both arms are dominated by the same shared direction, so the SAE-vs-baseline comparison had little power to detect a difference either way. Untested fix: project the dominant shared component out of both fingerprints. Prediction, with collateral damage stated: formatting and confidence become separable, diversity and creativity degrade, aggregate may not move.

## 5. What could prove this wrong

- **Stage 1** would fall if a reimplementation encoded pair differences directly rather than differencing in activation space — that reads `-b_pre` and would collapse the result. Anyone reproducing should confirm the null sits at chance first.
- **The complementarity** rests on formatting at 5/20 and 8/20 against 0. It replicates across three runs but is thin; a fourth run where the SAE fails formatting would weaken it materially. Confidence (0 vs 15–18) is the sturdier half.
- **The calibration claim** removes a concept from the candidate set rather than planting a genuinely novel one. A removed concept still correlates with what remains, so 0.966 is a lower bound on the confidence a truly unrepresentable axis would draw. A generated out-of-library axis (Bedrock + Skywork embedding pass) would test it properly.
- **The effective-dimensionality result** is the load-bearing claim, and it is pure geometry over `concept_vectors.pt` — no seeds, no fitting. It fails only if those vectors are wrong, e.g. built from too few contrast pairs. Worth checking against the CA concept vectors.
- **All numbers assume the D3 checkpoint matches current embeddings.** D3 trained on the 12,999-record pickles; the fixed set has 19,638. If the fix changed vectors rather than coverage, every SAE number here needs a rerun.
