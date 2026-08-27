"""
Embed the synthetic response pool with the frozen Skywork backbone.

Produces {pool_key: [4096] tensor}, the format build_synthetic_personas.py's `pairs` stage
expects. Same extraction as everywhere else in this project: last-token hidden state of the
chat-templated (prompt, response) conversation, so the synthetic pool lives in the same space as
PRISM, Community Alignment and the concept vectors.

Usage (on a GPU box):
  python PRISM/embed_response_pool.py
  python PRISM/embed_response_pool.py --limit 20   # smoke test
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic_personas/response_pool.json")
    ap.add_argument("--prompts", default="data/synthetic_personas/prompts.json")
    ap.add_argument("--output", default="data/synthetic_personas/pool_embeddings.pt")
    ap.add_argument("--batch_size", type=int, default=16, help="hard cap on batch size")
    ap.add_argument("--token_budget", type=int, default=12000,
                    help="max batch_size * longest_sequence_in_batch")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    with open(args.pool) as f:
        pool = json.load(f)
    with open(args.prompts) as f:
        prompts = json.load(f)

    keys = sorted(pool)
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} responses to embed", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # sdpa rather than eager: eager materialises the full [batch, heads, seq, seq] attention
    # matrix, which OOMs a 22 GB A10G on this pool because verbosity_high responses run to ~5k
    # tokens. sdpa computes the same attention without that allocation. The numerical difference
    # is far below the bf16 noise already present, and well below the effect sizes measured here.
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", num_labels=1,
    )
    model.eval()

    # Sort by length so each batch pads to a similar width. The pool spans 530 to 17k characters
    # because verbosity is one of the planted axes; batching mixed lengths would pad every short
    # response out to the longest in its batch.
    keys = sorted(keys, key=lambda k: len(pool[k]["response"]))

    # Fixed-size batches do not work when lengths span 530 to 17k characters: a batch of 4 long
    # responses allocates ~16x what a batch of 4 short ones does. Build batches under a token
    # budget instead, so long sequences run in small batches and short ones in large batches.
    tok_len = {k: len(tokenizer(pool[k]["response"] + prompts[pool[k]["prompt_idx"]],
                                add_special_tokens=False)["input_ids"]) for k in keys}
    batches, cur = [], []
    for k in keys:
        trial = cur + [k]
        width = max(tok_len[t] for t in trial)
        if cur and (len(trial) * width > args.token_budget or len(trial) > args.batch_size):
            batches.append(cur)
            cur = [k]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches, max size {max(len(b) for b in batches)}, "
          f"longest sequence {max(tok_len.values())} tokens", flush=True)

    out = {}
    for chunk in tqdm(batches, desc="embedding"):
        texts = [tokenizer.apply_chat_template(
            [{"content": prompts[pool[k]["prompt_idx"]], "role": "user"},
             {"content": pool[k]["response"], "role": "assistant"}], tokenize=False)
            for k in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=8192, add_special_tokens=False).to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, -1].cpu().to(torch.float32)
        for k, v in zip(chunk, h):
            out[k] = v

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(out, args.output)
    print(f"saved {len(out)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
