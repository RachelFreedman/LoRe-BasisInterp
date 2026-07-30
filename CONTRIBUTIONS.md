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
