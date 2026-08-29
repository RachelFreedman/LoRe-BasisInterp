# Reward Basis Decomposition (RBD)

**RBD** is a method for isolating and interpreting the preference structure a multi-user
reward model has learned. It fits a basis shared across users and reads two objects off the
fit: a single **population direction**, describing what users agree on, and one **user
direction** per user, describing how that user departs from it. Both are vectors in the
reward model's embedding space, so shared and individual preference structure can be
examined separately, and the analysis applies to a handful of directions rather than to a
whole network.

This repository is a fork of the reference implementation of
**[LoRe](https://arxiv.org/abs/2504.14439)** (Bose et al., 2025). RBD is built on that
family of methods, and LoRe serves as the baseline throughout. LoRe's own documentation is
retained below, and its usage instructions still apply.

---

## What this fork adds

- **The RBD estimator** — signed user weights factored as a population term plus a per-user
  deviation, `w_u = wbar + delta_u`; a ridge penalty in reward-function space in place of a
  cosine anchor to a reference direction; no basis-column dropping; joint optimisation with
  early stopping on a validation split. Implemented as `LoReV2` in `utils.py` alongside the
  untouched LoRe classes, so the two run head to head.
- **Identifiable axes** — signed weights leave the raw basis columns unidentified, so
  `canonical_variation_axes(...)` recovers interpretable axes from a fit by SVD of the
  centred per-user reward deviations followed by a varimax rotation.
- **A synthetic control with planted personalization** — personas with known reward
  directions judging one shared response pool, so the recovery target is known in closed
  form. This establishes what the estimator finds when per-user structure is known to be
  present, which is what makes a null result on real data interpretable.
- **Interpretability analyses** — alignment of the recovered directions against a named
  concept library, and decomposition over sparse-autoencoder features.
- **Corrections to the released code** — the reward-head extraction, and a preprocessing
  artifact that formatted the two sides of a preference pair differently.

Datasets: PRISM, Community Alignment, and the synthetic control, alongside LoRe's original
Reddit TLDR and PersonalLLM support.

---

## 📦 Installation

LoRe requires Python 3.8+ and PyTorch. To install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂 Project Structure

```
LoRe/
├── utils.py                    # Core training, optimization, and evaluation helpers
├── RedditTLDR/                 # TLDR dataset scripts
│   ├── prepare.py              # Preprocess the dataset
│   ├── train_basis.py          # Train shared reward model and user weights
│   └── vary_fewshot.py         # Evaluate few-shot personalization
├── PRISM/                      # PRISM dataset scripts
│   ├── prepare.py
│   ├── train_basis.py
│   └── vary_fewshot.py
├── PersonalLLM/                # PersonalLLM dataset scripts
│   ├── prepare.py
│   ├── train_basis.py
```

---

## 🔧 Core Components (`utils.py`)

The following functions are central to training and evaluating personalized reward models. You may want to modify these if you’re extending LoRe:

### Model 
- `LoRe(...)`: Class modeling shared reward model `V` (linear transformation on fixed embeddings) and user-specific weights `W`
- `LoRe_regularized(...)`: Class modeling shared reward model `V` (linear transformation on fixed embeddings), cosine similarity regularization to base model, and user-specific weights `W`
- `PersonalizeBatch(...)`: Class to model weights for new users
- `LoReV2(...)`: Class modeling a shared basis `V` with unconstrained (signed) user weights factored as a population term plus a per-user deviation, `w_u = wbar + delta_u`, regularized by a ridge penalty in reward-function space rather than a cosine anchor to a reference direction
- `PersonalizeDelta(...)`: Class to model weights for new users under `LoReV2`, freezing `V` and `wbar` and learning only `delta_u`
- `solve_lore_v2(...)`: Convenience wrapper that fits `LoReV2` and returns the learned basis and weights
- `canonical_variation_axes(...)`: Recovers identifiable axes from a fitted `LoReV2` model by SVD of the centred per-user reward deviations, followed by a varimax rotation

> **A note on naming.** The class is called `LoReV2` in the source for historical
> reasons. It is the estimator referred to as **RBD (Reward Basis Decomposition)**
> in the accompanying paper; the two names denote the same model. The source name
> is retained so that existing scripts, checkpoints and result files continue to
> resolve.

## Training and Evaluation
- `run(...)`: Runs the entire pipeline with 1. Learning the basis rewards, 2. Evaluation on seen users, 3. Fewshot learning on new users, 4. Evaluation on new users. The input K_list can be modified to specify the number of basis. 0 is the reference model, and 1 is the BT model.
- `run_regularized(...)`: Runs the entire pipeline as run(...) but with regularization on the final layer


---

## 🧪 Usage Instructions

Datasets used in this work, each following the same workflow: prepare, train the basis,
evaluate.

### 🔸 PRISM

Inside the `PRISM/` directory:
- `prepare.py`: prepares PRISM dialogue in chat format
- `generate-prepare-embeddings.py`: prepares PRISM embeddings
- `train_basis.py`: runs reward model training and evaluation
- `eval_rm2.py`: runs learn reward basis models on the reward bench 2 dataset

Example usage:
```bash
cd LoRe/PRISM
python prepare.py          # only needed once
python generate-prism-embeddings.py # only needed once
python train_basis.py # train the model for a list of ranks, default regularization is specified
python eval_rb2.py --rm_head "path to saved final layer (V) weights" # evaluate learnt reward basis on RewardBench2 to avoid overfitting to PRISM
```

---

### 🟢 Community Alignment

The second real dataset used in this work. Preparation, embedding and RBD training are in
`PRISM/`:

```bash
python PRISM/community_alignment_prep.py --min_pairs 50 --out data/community_alignment/pairs.json
python PRISM/embed_community_alignment.py
python PRISM/community_alignment_lore.py --pairs data/community_alignment/pairs.json \
    --emb data/community_alignment/embeddings.pt
```

Splits are taken at the level of conversation turns, never individual pairs: each turn yields
three pairs sharing a prompt and a preferred response, so a pair-level split leaks siblings
across the boundary.

---

### 🧬 Synthetic control

Personas with known reward directions judging one shared response pool, used to establish
what the estimator recovers when per-user structure is known to be present.

```bash
python PRISM/build_synthetic_personas.py personas --n_users 60 --max_cos 0.6
python PRISM/build_synthetic_personas.py generate --n_prompts 60 --workers 4
modal run modal_embed_pool.py                     # embeds the shared response pool
python PRISM/build_synthetic_personas.py pairs --pairs_per_user 240 --margin 0.25
python PRISM/synthetic_persona_lore.py --ranks 1 4 8 16
```

Splits are taken by whole prompt: every user draws pairs from the same prompt set, so a
pair-level split would put identical response texts on both sides.

---

### 🔍 SAE interpretability of the shared population direction

Interpretability analysis of the LoRe v2 shared population direction `v_pop = unit(V @ wbar)` — is it a distinct learned preference axis, or just the base Skywork reward head? It is read through an 11-concept library and a 16,384-feature TopK SAE, and run on both PRISM and Community Alignment.

- **PRISM:** writeup [`PRISM/sae_analysis.md`](PRISM/sae_analysis.md); code and results in [`sae/experiments/`](sae/experiments/) (`exp1`–`exp5`, results under `results/exp*/`, saved directions under `artifacts/`).
- **Community Alignment:** writeup [`PRISM/ca_sae_analysis/ca_sae_analysis.md`](PRISM/ca_sae_analysis/ca_sae_analysis.md); code, results, and reproduction steps in [`PRISM/ca_sae_analysis/`](PRISM/ca_sae_analysis/) (see its `README.md`). The ~512 MB D3 SAE checkpoint is gitignored — download and drop it in locally to run exp3.

---

## Inherited from LoRe

The fork also carries LoRe's original dataset scripts. They are not used in this work, and
are kept so the upstream pipelines continue to run; both have had the reward-head extraction
fix applied.

### 🔹 Reddit TLDR

Inside the `RedditTLDR/` directory:
- `prepare.py`: preprocesses the TLDR dataset
- `train_basis.py`: trains the shared reward model and user weights
- `vary_fewshot.py`: evaluates few-shot personalization performance

Example usage:
```bash
cd LoRe/RedditTLDR
python prepare.py          # only needed once
python train_basis.py
python vary_fewshot.py
```

### 🟣 PersonalLLM

Inside the `PersonalLLM/` directory:
- `prepare.py`: prepares model response data and user splits
- `train_basis.py`: learns reward basis across users
- `vary_fewshot.py`: evaluates few-shot generalization

Example usage:
```bash
cd LoRe/PersonalLLM
python prepare.py          # only needed once
python train_basis.py
python vary_fewshot.py
```

---

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
CC-BY-NC 4.0 licensed, as found in the LICENSE file.

## Cite Us
If you use this codebase, please cite us:
```bibtex
@misc{bose2025lorepersonalizingllmslowrank,
      title={LoRe: Personalizing LLMs via Low-Rank Reward Modeling}, 
      author={Avinandan Bose and Zhihan Xiong and Yuejie Chi and Simon Shaolei Du and Lin Xiao and Maryam Fazel},
      year={2025},
      eprint={2504.14439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14439}, 
}
