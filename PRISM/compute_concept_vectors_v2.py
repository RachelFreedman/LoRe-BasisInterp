"""
Compute concept vectors for the extended library from contrastive_pairs_v2.json.

Differences from compute_concept_vectors.py, which is left untouched:
  - input/output are arguments, defaulting to the _v2 paths, so concept_vectors.pt is never
    overwritten (committed results depend on it)
  - batched with left padding instead of one forward pass per response (~1200 passes here)
  - de-duplicates identical (prompt, response) texts before embedding

Left padding matters: with padding on the left the final position is the true last token for
every row, so last_hidden_state[:, -1] is correct without gathering per-row lengths.

Usage (on a GPU box):
  python PRISM/compute_concept_vectors_v2.py
  python PRISM/compute_concept_vectors_v2.py --limit 3   # smoke test, 3 pairs per concept
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def embed_texts(convs, model, tokenizer, device, batch_size, desc):
    """Embed a list of chat conversations, returning [N, 4096] last-token hidden states."""
    out = []
    for i in tqdm(range(0, len(convs), batch_size), desc=desc):
        chunk = convs[i:i + batch_size]
        texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096, add_special_tokens=False).to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, -1]
        out.append(h.cpu().to(torch.float32))
    return torch.cat(out) if out else torch.empty(0, 4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/prism/contrastive_pairs_v2.json")
    ap.add_argument("--output", default="data/prism/concept_vectors_v2.pt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="pairs per concept (smoke test)")
    ap.add_argument("--save_embeddings", default=None,
                    help="also save per-pair high/low embeddings, for split-half reliability")
    args = ap.parse_args()

    if args.output == "data/prism/concept_vectors.pt":
        raise SystemExit("refusing to overwrite concept_vectors.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    with open(args.input) as f:
        pairs_by_concept = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", num_labels=1,
    )
    model.eval()

    # Collect every distinct conversation once across all concepts, then embed in one pass.
    cache_keys, convs = {}, []

    def register(prompt, response):
        key = (prompt, response)
        if key not in cache_keys:
            cache_keys[key] = len(convs)
            convs.append([{"content": prompt, "role": "user"},
                          {"content": response, "role": "assistant"}])
        return cache_keys[key]

    index = {}
    for concept, pairs in pairs_by_concept.items():
        if not pairs:
            continue
        use = pairs[:args.limit] if args.limit else pairs
        index[concept] = [(register(p["prompt"], p["high_response"]),
                           register(p["prompt"], p["low_response"])) for p in use]
        print(f"{concept}: {len(use)} pairs", flush=True)

    print(f"\n{len(convs)} unique conversations to embed", flush=True)
    E = embed_texts(convs, model, tokenizer, device, args.batch_size, "embedding")

    vectors = {}
    for concept, idxs in index.items():
        hi = E[[a for a, _ in idxs]]
        lo = E[[b for _, b in idxs]]
        v = hi.mean(0) - lo.mean(0)
        vectors[concept] = v
        print(f"{concept:<14} n={len(idxs):<4} norm={v.norm():.4f}", flush=True)

    if args.save_embeddings:
        # Per-pair embeddings let us split the pairs in half and build two independent estimates
        # of the same concept from the same generator. cos between those halves is the noise
        # floor that any cross-generator comparison has to be judged against.
        per = {c: {"hi": E[[a for a, _ in idxs]], "lo": E[[b for _, b in idxs]]}
               for c, idxs in index.items()}
        torch.save(per, args.save_embeddings)
        print(f"saved per-pair embeddings to {args.save_embeddings}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(vectors, args.output)
    print(f"\nsaved {len(vectors)} vectors to {args.output}")


if __name__ == "__main__":
    main()
