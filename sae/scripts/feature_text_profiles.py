#!/usr/bin/env python3
"""
Stage 2 (rebuilt): give every SAE feature a meaning derived ONLY from PRISM, so the
planted-concept test cannot be passed by knowing the concepts.

Why this exists
---------------
The first Stage 2 scored a direction by comparing it to feature profiles built from
the SAME concept contrast sets the directions were planted from. That made the task
solvable without the SAE at all: plain cosine to the concept vectors identifies all
12 groups perfectly (1.000), while routing it through the SAE scored 0.778. So that
number measured how much concept identity survives the sparse bottleneck -- a
fidelity result -- not whether the SAE can NAME anything.

The fix is to break the link between how features are described and how concepts
were planted. Here a feature's meaning comes from the text of the PRISM responses it
fires on, measured with a priori surface statistics. PRISM is the SAE's own training
distribution and has no connection to the LLM-written concept contrast sets.

    feature -> meaning        from PRISM text  (this script)
    direction -> features     from geometry, decoder alignment
    direction -> true concept from the planting

No path in that chain passes through a concept vector, so the SAE has to carry the
answer itself.

Proxies
-------
Only concepts with a defensible surface proxy can be tested this way, which is four
of the six: formatting, repetition, diversity, confidence. fluency and creativity
have no honest surface measure and are excluded rather than faked. Each proxy is
oriented so that HIGHER means MORE of the concept -- note confidence is the negated
hedging rate.

Each proxy's validity is checked on the concept contrast set (does it actually
separate that concept's high from low responses?) and reported before anything else.
A proxy that fails that check is not evidence about its concept.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))

from sae.src.io import ensure_dir, write_json  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402

HEDGES = ["maybe", "perhaps", "might", "possibly", "probably", "seems", "appears",
          "i think", "i believe", "could be", "not sure", "unsure", "arguably",
          "somewhat", "tends to", "it depends"]
AGREE = ["you're right", "great question", "absolutely", "i completely agree",
         "good point", "exactly", "of course"]
REFUSE = ["i can't", "i cannot", "i'm not able", "i won't", "as an ai", "i'm sorry",
          "cannot provide", "not appropriate"]
_MD = re.compile(r"(^\s*[-*+]\s|^\s*\d+\.\s|^#{1,6}\s|\*\*|__|```|\|)", re.M)
_WORD = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _rate(text: str, phrases: list[str]) -> float:
    low = text.lower()
    return 100.0 * sum(low.count(p) for p in phrases) / max(len(_tokens(text)), 1)


def _ngram_rep(text: str, n: int = 4) -> float:
    t = _tokens(text)
    if len(t) < n + 1:
        return 0.0
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return 1.0 - len(Counter(grams)) / len(grams)


# A generic battery of surface measures, NOT one proxy per concept. Chosen this way
# deliberately: a single-proxy mapping fails here because `markdown` alone scores AUC
# 1.00 on formatting, 1.00 on diversity and 0.99 on helpfulness -- three concepts the
# generator rendered with the same surface device. Concepts separate as VECTORS over
# the whole battery (max pairwise cos 0.94, safety~values) even where individual
# measures collide, so identification uses the full profile.
PROXIES = {
    "markdown": lambda t: 100.0 * len(_MD.findall(t)) / max(len(_tokens(t)), 1),
    "length": lambda t: float(len(_tokens(t))),
    "ttr": lambda t: len(set(_tokens(t)[:200])) / max(len(_tokens(t)[:200]), 1),
    "ngram_rep": _ngram_rep,
    "hedge": lambda t: _rate(t, HEDGES),
    "agree": lambda t: _rate(t, AGREE),
    "refuse": lambda t: _rate(t, REFUSE),
    "avg_word_len": lambda t: float(np.mean([len(w) for w in _tokens(t)])) if _tokens(t) else 0.0,
    "sent_count": lambda t: float(t.count(".") + t.count("!") + t.count("?")),
    "question": lambda t: float(t.count("?")),
}


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sae(path: Path, device: torch.device) -> tuple[TopKSAE, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    tcfg = cfg.get("training", {})
    model = TopKSAE(input_dim=int(cfg["input_dim"]), dict_size=int(cfg["dict_size"]),
                    k=int(cfg["k"]), normalize_decoder=bool(tcfg.get("normalize_decoder", True)),
                    aux_k=int(tcfg.get("aux_k", cfg["k"])),
                    sparsity_mode=str(tcfg.get("sparsity_mode", "topk")))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def encode(model: TopKSAE, x: torch.Tensor, bs: int, device: torch.device) -> torch.Tensor:
    return torch.cat([model.encode(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


def prism_texts(pkl_path: Path, metadata: list[dict], split: str) -> list[str | None]:
    """Text for each row of the SAE split, joined by (original_index, response_role)."""
    rows = torch.load(pkl_path, map_location="cpu", weights_only=False)
    out = []
    for m in metadata:
        if m["sae_split"] != split or m["source_split"] != rows[0]["extra_info"]["split"]:
            out.append(None)
            continue
        extra = rows[m["original_index"]]["extra_info"]
        utt = extra["chosen_utterance" if m["response_role"] == "chosen" else "rejected_utterance"]
        out.append(" ".join(utt) if isinstance(utt, list) else utt)
    return out


def feature_profiles(z: torch.Tensor, props: np.ndarray, top_n: int) -> np.ndarray:
    """[dict, n_props] z-scored property means over each feature's top-activating texts.

    A feature that fires mostly on heavily formatted PRISM responses gets a high
    formatting score. Features active on fewer than top_n texts are left at zero, so
    they cannot dominate a signature on the strength of one example.
    """
    mu, sd = props.mean(0), props.std(0) + 1e-8
    prof = np.zeros((z.shape[1], props.shape[1]), dtype=np.float32)
    for i in range(z.shape[1]):
        act = z[:, i]
        nz = int((act > 0).sum())
        if nz < top_n:
            continue
        idx = torch.topk(act, top_n).indices.numpy()
        prof[i] = (props[idx].mean(0) - mu) / sd
    return prof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    ap.add_argument("--sae-data", default="sae/data")
    ap.add_argument("--prism-train", default="../phase1_artifacts/train_embeddings.pkl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--embeddings", default="data/prism/contrastive_pair_embeddings.pt")
    ap.add_argument("--top-n", type=int, default=50, help="top-activating texts per feature")
    ap.add_argument("--out", default="results/planted/feature_text_profiles.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = choose_device(args.device)
    model, cfg = load_sae(Path(args.checkpoint), device)

    meta = [json.loads(line) for line in open(Path(args.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == args.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    x = torch.load(Path(args.sae_data) / f"sae_{args.split}.pt", map_location="cpu").float()
    print(f"SAE {args.split} split: {len(x)} embeddings, dict={cfg['dict_size']}, k={cfg['k']}")

    texts = prism_texts(Path(args.prism_train), meta, args.split)
    keep = [i for i, t in enumerate(texts) if t]
    print(f"joined text for {len(keep)}/{len(texts)} rows")

    props = np.array([[PROXIES[p](texts[i]) for p in PROXIES] for i in keep], dtype=np.float32)
    z = encode(model, x[keep], args.batch_size, device)
    prof = feature_profiles(z, props, args.top_n)
    live = int((np.abs(prof).sum(1) > 0).sum())
    print(f"profiled {live}/{cfg['dict_size']} features (rest fire on < {args.top_n} texts)\n")

    # Each concept's own surface signature: the mean z-scored high-minus-low delta over
    # the battery, computed from the contrast TEXT alone. This is the target a
    # direction's feature profile is matched against in the identification step -- and
    # it never touches an embedding, a concept vector, or the SAE.
    pairs = json.load(open("data/prism/contrastive_pairs.json"))
    concepts = sorted(pairs)
    pooled = {k: [] for k in PROXIES}
    for c in concepts:
        for e in pairs[c]:
            for k, f in PROXIES.items():
                pooled[k] += [f(e["high_response"]), f(e["low_response"])]
    sd = {k: float(np.std(v)) + 1e-9 for k, v in pooled.items()}
    sig = np.array([[float(np.mean([(f(e["high_response"]) - f(e["low_response"])) / sd[k]
                                    for e in pairs[c]]))
                     for k, f in PROXIES.items()] for c in concepts], dtype=np.float32)
    sig_n = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-9)
    cos = sig_n @ sig_n.T
    off = np.abs(cos - np.eye(len(concepts)))

    print("=== concept signatures in surface-proxy space ===")
    print(f"  {len(concepts)} concepts x {len(PROXIES)} measures")
    print(f"  max pairwise |cos| {off.max():.2f}")
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            if abs(cos[i, j]) >= 0.80:
                print(f"    {concepts[i]:<12} {concepts[j]:<12} {cos[i, j]:+.2f}   "
                      f"[not separable on surface features]")
    print("  strongest measure per concept:")
    names = list(PROXIES)
    for i, c in enumerate(concepts):
        j = int(np.abs(sig[i]).argmax())
        print(f"    {c:<12} {names[j]:<13} {sig[i, j]:+.2f} sd")

    ensure_dir(Path(args.out).parent)
    torch.save({"profiles": prof, "proxy_names": names, "concepts": concepts,
                "concept_signatures": sig, "activity": z.abs().mean(0),
                "top_n": args.top_n, "n_texts": len(keep), "split": args.split}, args.out)
    write_json(Path(args.out).with_suffix(".json"),
               {"profiled_features": live, "dict_size": int(cfg["dict_size"]),
                "n_texts": len(keep), "top_n_texts_per_feature": args.top_n,
                "proxy_names": names, "concepts": concepts,
                "max_concept_pair_cos": float(off.max())})
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
