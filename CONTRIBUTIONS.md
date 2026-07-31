# Contributions — Mohamed Eldagla (LoRe / PRISM interpretability)

- Defined a concept library of 11 human-interpretable traits (7 from PRISM, 4 from Reward-Lens). Used Claude Sonnet 4 to generate 50 contrastive pairs per concept (50 × 11 = 550 total), then extracted 4096-dim concept vectors from the Skywork reward model on Modal.

- Tested cosine similarity between learned bases and concept vectors against a 95th-percentile null distribution.

- Wrote a causal ablation script and discovered all checkpoints collapsed to Rank-1 (inter-basis cosine sim ≥ 0.9996). Traced it to the original alpha=10000 penalty in Meta's training script.

- Also built cross-dataset transfer and causal-edit probes to characterize the direction, which pointed to a generic-quality axis — later explained by the formatting artifact.

- Following Ifesi's finding that the V_sft anchor was never the real reward head (AutoModel silently drops Skywork's score.weight, so the original "last nn.Linear" extraction grabbed an arbitrary internal MLP column, layers.31.mlp.down_proj), verified it on our pipeline and implemented the fix to load the true score.weight, which corrected the base-RM baseline and the anchor.

- Ran a multi-seed collapse sweep with the corrected anchor and showed the alpha penalty is a co-driver, not the whole cause: at alpha=0 the bases were still heavily correlated (min |cos| ≈ 0.83), so prompt-uniqueness/data was pushing toward one direction independently of the regularizer.

- Verified the true root cause on our shared pipeline end-to-end: prepare.py stored chosen responses as a string but rejected as a list, so the chat template rendered rejected as a Python list-repr (`['answer one', ...]<|eot_id|>`) vs. clean prose for chosen. The reward model was separating pairs on that formatting fingerprint — the actual source of both the Rank-1 collapse and the inflated accuracy. Fixed prepare.py and generate-prism-embeddings.py (single-string formatting + handling of empty-rejected turns) and built a Modal pipeline to regenerate corrected embeddings.

- On the corrected embeddings, confirmed the collapse is gone at alpha=0 (inter-basis cos ≈ 0.06) but LoRe now only ties the base RM (~0.59) at every alpha and rank (best +0.0045) — i.e. the original ~0.98 was the artifact.

- Wrote a per-user diagnostic: each user's own preference direction (0.566) scores worse on their held-out pairs than a single global direction (0.591), and only +0.013 over a random other user's — no learnable per-user signal above the general quality axis.

- Wrote a cluster-level diagnostic to rule out sparsity: clustered users and fit one direction per cluster on pooled pairs; own-cluster stayed below the global direction at every K from 2 to 50. So the null isn't a per-user-sparsity artifact — there is no separable per-user or per-group personalization signal in these reward-tuned embeddings.

- Built a synthetic positive control to test whether LoRe can recover known axes at all: planted K orthonormal directions, gave each user one, and swept pairs-per-user. Found a sharp phase transition at ~50 pairs/user — above 100 LoRe recovers the axes near-perfectly (subspace alignment 0.998, every user matched to their axis), below ~25 it is at chance even with clean orthogonal axes. PRISM's ~15 pairs/user sits in the failure regime and reproduces its exact signature there (train 1.000 / test 0.52).

- Built the text-based version of that control, since the abstract one bypasses the embedding space entirely: took 6 mutually-distinct concepts, constructed users who genuinely disagree (for each concept, some prefer high-C and some low-C, each seeing their own random subset), and embedded through Skywork using the same last-token representation as PRISM. LoRe recovers the planted axes — subspace alignment 0.694 vs a 0.050 random-basis floor, per-axis match 0.604 vs 0.030, user→axis assignment 0.458 vs 0.083 chance, no basis collapse. So the reward-tuned embedding space *does* linearly encode interpretable preference axes; the representation was never the bottleneck. This also positively controls the per-user diagnostic above: it reports personal ≫ global (0.995 vs 0.500) when personalization exists, so the PRISM null was a real null rather than a dead instrument.

- Built the full Community Alignment pipeline (English filter, data-rich user selection, a batched + de-duplicated embedder, LoRe training and diagnostics) to test the data-volume hypothesis on a dataset with a median of 201 pairs/user versus PRISM's ~15. Fixed the `preferred_response` parsing (values are `response_a`..`response_d`, not a bare letter or the response text).

- Found and fixed a train/test leakage bug of my own in that pipeline: each turn yields 3 pairs sharing the same prompt and chosen response, so splitting by pair put siblings on both sides and let LoRe memorise "for this prompt, response_d wins" while the untrained base RM gained nothing. Splitting by turn removed it, and the apparent result reversed — LoRe 0.815 (+0.177 vs base) became 0.618 (−0.019).

- Swept pairs-per-user on Community Alignment (200 users, budgets {51, 102, 201, 225}, 3 seeds, leak-free split) — the real-data analogue of the synthetic phase-transition curve. LoRe stays ~0.03 *below* the base RM at every budget with no upward trend, and a user's own direction never beats a single shared direction (personal − global is zero within noise throughout). This falsifies the data-volume hypothesis, including my own earlier conclusion that PRISM's null was a volume problem: 15× the data per user changes nothing.

- Also swept alpha there, reproducing PRISM's bind on a second dataset: at alpha=0 the bases stay distinct (min |cos| ≈ 0.0003) but LoRe is significantly worse than the base RM (−0.026 ± 0.004 over 5 seeds); by alpha=200 it only ties the base RM (+0.0003 ± 0.0010) with the bases collapsed to 0.94 — i.e. it has become the reward head. Distinct or useful, never both.

- Noted that `personal − other_user` stays positive but *shrinks* as data grows (+0.032 → +0.014). Genuine per-user taste should sharpen with more data rather than fade, so this is more consistent with topic overlap or small-sample idiosyncrasy than with preference personalization. Remaining unresolved confound: Community Alignment does not record which model produced each of responses a–d, so part of the measured "preference" is model quality rather than user taste.
