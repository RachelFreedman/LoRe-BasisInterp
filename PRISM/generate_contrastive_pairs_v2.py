"""
Generate contrastive pairs for the extended concept library (concept_library_v2).

Writes to data/prism/contrastive_pairs_v2.json. Leaves the original contrastive_pairs.json and
concept_vectors.pt untouched, so no committed result changes.

Auth: AWS_BEARER_TOKEN_BEDROCK, AWS_DEFAULT_REGION, BEDROCK_MODEL_ID.
The inference-profile prefix must match the region (eu.anthropic.* for an eu-* key).

Resumable: re-running tops up whichever concepts are short of --target and skips the rest.

Usage:
  python PRISM/generate_contrastive_pairs_v2.py --smoke
  python PRISM/generate_contrastive_pairs_v2.py --target 50
  python PRISM/generate_contrastive_pairs_v2.py --concepts verbosity formality --target 50
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_library_v2 import CONCEPT_LIBRARY_V2

REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")
# Judging is a YES/NO classification and does not need the generation model. Pointing it at a
# smaller model roughly halves the load on the generation model's quota, which is the binding
# constraint here -- sustained 12-way generation throttles hard.
JUDGE_MODEL_ID = os.environ.get("BEDROCK_JUDGE_MODEL_ID") or MODEL_ID
OUTPUT_FILE = "data/prism/contrastive_pairs_v2.json"

# Bedrock throttles hard on sustained load. Retry in the SDK as well as in our own loop.
_cfg = Config(region_name=REGION, retries={"max_attempts": 8, "mode": "adaptive"},
              read_timeout=300, connect_timeout=15)
_client = boto3.client("bedrock-runtime", config=_cfg)

THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException",
                  "ServiceUnavailableException", "ModelTimeoutException"}


def invoke(prompt, max_tokens=8000, temperature=0.7, max_retries=6, model_id=None):
    """One Converse call with exponential backoff on throttling. Returns text or None."""
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    for attempt in range(max_retries):
        try:
            r = _client.converse(
                modelId=model_id or MODEL_ID,
                messages=messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            return r["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_CODES and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                continue
            print(f"\n  [bedrock] {code or type(e).__name__}: {e}", flush=True)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                continue
            print(f"\n  [bedrock] {type(e).__name__}: {e}", flush=True)
            return None
    return None


def parse_json_list(text):
    """Models sometimes wrap JSON in a fence despite instructions. Strip and parse."""
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    # Fall back to the outermost bracket pair if there is stray prose around the array.
    if not t.startswith("["):
        i, j = t.find("["), t.rfind("]")
        if i == -1 or j <= i:
            return []
        t = t[i:j + 1]
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def generate_batch(name, info, batch_size, seen_prompts):
    avoid = ""
    if seen_prompts:
        sample = random.sample(sorted(seen_prompts), min(12, len(seen_prompts)))
        avoid = ("\nThese user prompts have already been used. Write about clearly different "
                 "topics:\n" + "\n".join(f"- {p}" for p in sample) + "\n")

    prompt = f"""We are building a dataset of contrastive response pairs that isolate a single stylistic dimension: '{name}'.
Dimension: {info['description']}

Write {batch_size} diverse, realistic user prompts (questions or requests), varied in topic across coding, writing, science, everyday advice, reasoning and general knowledge.

For EACH prompt write two responses that differ ONLY along this dimension:
1. high_response: {info['high']}
2. low_response: {info['low']}

Critical constraints:
- Both responses must be equally correct, equally accurate, and equally on-topic. Neither may be the better answer overall; they differ only in {name}.
- Hold length roughly matched between the two responses UNLESS the dimension is itself about length.
- Do not mention the dimension, and do not explain your choices.
{avoid}
Return ONLY a JSON array, no markdown fence, no commentary:
[{{"prompt": "...", "high_response": "...", "low_response": "..."}}]
"""
    return parse_json_list(invoke(prompt, temperature=0.9))


def judge(name, info, pair):
    """Ask whether the contrast is clean in the intended direction."""
    prompt = f"""You are judging whether two responses cleanly isolate one stylistic dimension: '{name}'.
Dimension: {info['description']}
- High pole: {info['high']}
- Low pole: {info['low']}

User prompt: {pair['prompt']}

Response 1:
{pair['high_response']}

Response 2:
{pair['low_response']}

Answer YES only if BOTH hold:
(a) Response 1 exhibits '{name}' clearly more than Response 2.
(b) The two are otherwise comparable in correctness and relevance, so neither is simply the better answer.

Reply with one word: YES or NO."""
    r = invoke(prompt, max_tokens=8, temperature=0.0, model_id=JUDGE_MODEL_ID)
    return bool(r) and r.strip().upper().startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="accepted pairs per concept")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--concepts", nargs="*", default=None, help="subset of concept names")
    ap.add_argument("--judge_workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="one small batch for one concept; writes nothing")
    ap.add_argument("--out", default=OUTPUT_FILE)
    args = ap.parse_args()

    if not MODEL_ID:
        sys.exit("BEDROCK_MODEL_ID is not set. source ~/.bedrock_env first.")
    print(f"[bedrock] region={REGION} model={MODEL_ID} judge={JUDGE_MODEL_ID}", flush=True)

    library = CONCEPT_LIBRARY_V2
    if args.concepts:
        missing = [c for c in args.concepts if c not in library]
        if missing:
            sys.exit(f"unknown concepts: {missing}")
        library = {c: library[c] for c in args.concepts}

    if args.smoke:
        name = next(iter(library))
        info = library[name]
        print(f"\nSMOKE TEST: '{name}', batch of 2\n" + "-" * 60, flush=True)
        t = time.time()
        batch = generate_batch(name, info, 2, set())
        print(f"generated {len(batch)} pairs in {time.time() - t:.1f}s", flush=True)
        if not batch:
            sys.exit("generation returned nothing -- check auth and model id")
        for p in batch:
            ok = all(k in p for k in ("prompt", "high_response", "low_response"))
            print(f"\n  keys ok: {ok}")
            if not ok:
                print(f"  got keys: {list(p)}")
                continue
            print(f"  prompt: {p['prompt'][:100]}")
            print(f"  high  ({len(p['high_response']):>4} ch): {p['high_response'][:110]!r}")
            print(f"  low   ({len(p['low_response']):>4} ch): {p['low_response'][:110]!r}")
            print(f"  judge: {'PASS' if judge(name, info, p) else 'REJECT'}", flush=True)
        print("\nsmoke test complete, nothing written")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_pairs = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            all_pairs = json.load(f)

    for name, info in library.items():
        kept = all_pairs.setdefault(name, [])
        if len(kept) >= args.target:
            print(f"{name}: {len(kept)}/{args.target} already, skipping", flush=True)
            continue

        print(f"\n{name}: {len(kept)}/{args.target}", flush=True)
        seen = {p["prompt"] for p in kept}
        fails = 0
        while len(kept) < args.target and fails <= 4:
            batch = generate_batch(name, info, args.batch_size, seen)
            batch = [p for p in batch
                     if all(k in p for k in ("prompt", "high_response", "low_response"))
                     and p["prompt"] not in seen]
            if not batch:
                fails += 1
                print(f"  empty batch ({fails}/5)", flush=True)
                time.sleep(2 * fails)
                continue
            fails = 0

            with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
                verdicts = list(ex.map(lambda p: judge(name, info, p), batch))

            for p, ok in zip(batch, verdicts):
                if ok and len(kept) < args.target:
                    kept.append(p)
                    seen.add(p["prompt"])
            with open(args.out, "w") as f:
                json.dump(all_pairs, f, indent=2)
            print(f"  +{sum(verdicts)}/{len(batch)} accepted -> {len(kept)}/{args.target}",
                  flush=True)

        if len(kept) < args.target:
            print(f"  WARNING: stopped at {len(kept)}/{args.target}", flush=True)

    print("\nfinal counts:")
    for name in library:
        print(f"  {name:<14} {len(all_pairs.get(name, []))}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
