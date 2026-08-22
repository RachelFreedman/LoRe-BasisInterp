"""
Build the synthetic preference dataset: personas, response pool, and preference pairs.

Design goal (from the spec): maximise the diversity of user preferences, so that LoRe-v2 has the
best possible chance of recovering per-user personalization. Realism is explicitly not the goal.

Why a SHARED response pool. The obvious construction -- generate each persona its own responses
and let it prefer its own -- does not test personalization. Users would never judge the same
items, so "user u prefers its own response" is separable by a single global direction that just
detects which generator wrote the text. Instead every prompt gets one pool of responses spanning
all persona styles, and every user judges pairs drawn from that shared pool. Users then genuinely
disagree about the same items, which is the only configuration where a shared direction cannot
explain the labels.

Stages:
  personas  design persona weight vectors, enforcing low pairwise similarity   (no API calls)
  generate  write the response pool via Bedrock                                (API calls)
  pairs     score responses, emit per-user preference pairs                    (needs embeddings)

Usage:
  python PRISM/build_synthetic_personas.py personas --n_users 60 --dry_run
  python PRISM/build_synthetic_personas.py generate --n_prompts 40
  python PRISM/build_synthetic_personas.py pairs --pairs_per_user 120
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_library_v2 import CONCEPT_LIBRARY_V2

OUT_DIR = "data/synthetic_personas"
PERSONA_FILE = f"{OUT_DIR}/personas.json"
POOL_FILE = f"{OUT_DIR}/response_pool.json"
PAIRS_FILE = f"{OUT_DIR}/pairs.json"


# --------------------------------------------------------------------------- personas

def design_personas(concepts, n_users, n_pref, n_dis, max_cos, seed, tries=200000,
                    weight_mode="continuous", jitter_sparsity=True, concept_matrix=None):
    """Sparse signed weight vectors over `concepts`, rejecting any too close to one already
    accepted.

    weight_mode='binary' gives every preferred concept +1 and every dispreferred one -1. That
    quantises the achievable cosines to multiples of 1/(n_pref+n_dis), so only a handful of
    personas exist below a given threshold -- 9 at 6 concepts, 27 at 8 -- and thresholds inside
    a quantisation gap (0.5 vs 0.6) behave identically. 'continuous' keeps the sign structure but
    draws magnitudes from U(0.5, 1.5), which fills the gaps and lets the persona count be chosen
    rather than dictated by the concept count.

    jitter_sparsity also varies how many concepts each persona cares about, so the set is not all
    one shape.

    concept_matrix ([4096, k]) makes the distinctness test run in REWARD space rather than weight
    space. This is not cosmetic: the concepts are themselves correlated (up to |cos| 0.83), so two
    personas with near-orthogonal weight vectors can still induce almost the same reward direction.

    IT MUST BE THE SIGMA-SCALED BASIS, C / sd, not the unit-normalised C. Labels come from
    STANDARDISED projections, so a persona's true reward direction is C @ (w / sd). The per-concept
    sd spans 5.2 to 27.5 on this pool, so testing separation on C @ w tests the wrong vector: with
    unit columns every requested threshold (0.60, 0.45, 0.35) still admitted persona pairs sitting
    at |cos| ~0.95 in the space the labels actually live in. Those users are indistinguishable by
    construction, which caps user->axis recovery at a level that has nothing to do with the method.
    """
    rng = random.Random(seed)
    k = len(concepts)
    if n_pref + n_dis > k:
        raise SystemExit(f"n_pref+n_dis={n_pref + n_dis} exceeds {k} concepts")

    accepted, vecs = [], []
    for _ in range(tries):
        if len(accepted) >= n_users:
            break
        np_, nd_ = n_pref, n_dis
        if jitter_sparsity:
            np_ = max(1, n_pref + rng.choice([-1, 0, 0, 1]))
            nd_ = max(1, n_dis + rng.choice([-1, 0, 0, 1]))
            if np_ + nd_ > k:
                continue
        idx = rng.sample(range(k), np_ + nd_)
        pref, dis = idx[:np_], idx[np_:]
        w = torch.zeros(k)
        if weight_mode == "binary":
            w[pref], w[dis] = 1.0, -1.0
        else:
            for i in pref:
                w[i] = rng.uniform(0.5, 1.5)
            for i in dis:
                w[i] = -rng.uniform(0.5, 1.5)
        w = w / w.norm()
        probe = w if concept_matrix is None else F.normalize(concept_matrix @ w, dim=0)
        if vecs and max(abs(float(probe @ v)) for v in vecs) > max_cos:
            continue
        vecs.append(probe)
        accepted.append({
            "user_id": f"synth_{len(accepted):03d}",
            "preferred": [concepts[i] for i in pref],
            "dispreferred": [concepts[i] for i in dis],
            "weights": {concepts[i]: float(w[i]) for i in range(k) if w[i] != 0},
        })

    if len(accepted) < n_users:
        print(f"WARNING: only {len(accepted)}/{n_users} personas met max_cos={max_cos}. "
              f"Raise --max_cos or widen the concept set.")
    W = torch.stack(vecs)
    off = (W @ W.T).abs()
    off.fill_diagonal_(0)
    space = "reward" if concept_matrix is not None else "weight"
    print(f"{len(accepted)} personas over {k} concepts ({space}-space separation)")
    print(f"  pairwise |cos|: max {off.max():.3f}  mean {off.sum() / (len(vecs) ** 2 - len(vecs)):.3f}")
    return accepted


def pool_styles(concepts, library):
    """The response pool is built per CONCEPT POLE, not per persona.

    Generating one response per persona would cost n_users x n_prompts calls (2400 at 60 users x
    40 prompts) and buys nothing: what the pool has to do is span the concept space so that
    different personas rank it differently. 2k responses per prompt -- each concept at its high
    and low pole -- spans the same space for 20 calls per prompt regardless of how many users
    there are, and decouples pool cost from population size entirely.

    It is also the better control: no response is "user u's own response", so no global direction
    can separate the labels by detecting who a response was written for.
    """
    styles = []
    for c in concepts:
        styles.append({"style_id": f"{c}_high", "concept": c, "pole": "high",
                       "instruction": library[c]["high"]})
        styles.append({"style_id": f"{c}_low", "concept": c, "pole": "low",
                       "instruction": library[c]["low"]})
    return styles


# --------------------------------------------------------------------------- prompts

def load_prompts(n_prompts, seed, existing_file=None):
    """Fixed question set, sampled from Community Alignment if available so the prompts look like
    the real data. Falls back to PRISM.

    EXTENSION IS NESTED. If a prompts.json already exists, its prompts are kept in their existing
    order and only the shortfall is drawn fresh. random.sample(pop, 60) is not a superset of
    random.sample(pop, 30) from the same seed, so re-sampling would silently orphan every response
    already generated against the old prompt_idx values -- paying twice for the same pool.
    """
    rng = random.Random(seed)
    keep = []
    if existing_file and os.path.exists(existing_file):
        with open(existing_file) as f:
            keep = json.load(f)
        if len(keep) >= n_prompts:
            print(f"reusing first {n_prompts} of {len(keep)} existing prompts")
            return keep[:n_prompts]
        print(f"keeping {len(keep)} existing prompts, drawing {n_prompts - len(keep)} more")
    for path, key in ((f"data/community_alignment/pairs.json", "prompt"),
                      (f"data/prism/pairs.json", "prompt")):
        if os.path.exists(path):
            with open(path) as f:
                rows = json.load(f)
            seen, out = set(), []
            for r in rows:
                p = r.get(key)
                if p and p not in seen and 40 < len(p) < 600:
                    seen.add(p)
                    out.append(p)
            have = set(keep)
            fresh = [p for p in out if p not in have]
            need = n_prompts - len(keep)
            if len(fresh) >= need:
                sel = keep + rng.sample(fresh, need)
                print(f"sampled {need} new prompts from {path} ({len(sel)} total)")
                return sel
    raise SystemExit("no prompt source found (data/community_alignment/pairs.json or "
                     "data/prism/pairs.json)")


# --------------------------------------------------------------------------- generation

def make_client():
    """Return (provider, client). Anthropic's API and Bedrock serve the same underlying models,
    so switching provider does not change the generator and introduces no drift -- only the
    request format and the model-id string differ.

    ANTHROPIC_API_KEY takes precedence when both are configured, since it is the one we fall
    back to when the Bedrock daily token cap is exhausted.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return "anthropic", anthropic.Anthropic(max_retries=0)   # we do our own backoff
    import boto3
    from botocore.config import Config
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    cfg = Config(region_name=region, retries={"max_attempts": 8, "mode": "adaptive"},
                 read_timeout=300, connect_timeout=15)
    return "bedrock", boto3.client("bedrock-runtime", config=cfg)


def model_id_for(provider):
    """Model id differs by provider: Bedrock needs a region-scoped inference profile
    (eu.anthropic.claude-opus-4-6-v1), Anthropic needs the bare name (claude-opus-4-6)."""
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL_ID") or "claude-opus-4-6"
    return os.environ.get("BEDROCK_MODEL_ID")


def invoke(client, model_id, prompt, max_tokens=4000, temperature=0.8, max_retries=10):
    provider, handle = client
    if provider == "anthropic":
        return _invoke_anthropic(handle, model_id, prompt, max_tokens, temperature, max_retries)
    return _invoke_bedrock(handle, model_id, prompt, max_tokens, temperature, max_retries)


def _invoke_anthropic(client, model_id, prompt, max_tokens, temperature, max_retries):
    import anthropic
    for attempt in range(max_retries):
        try:
            r = client.messages.create(
                model=model_id, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            if r.stop_reason == "max_tokens":
                print(f"  [truncated at max_tokens, discarding] {prompt[-60:]!r}", flush=True)
                return None
            return "".join(b.text for b in r.content if b.type == "text")
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            status = getattr(e, "status_code", None)
            if (isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError))
                    or status in (408, 409, 429, 500, 502, 503, 529)):
                if attempt < max_retries - 1:
                    time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                    continue
            print(f"  [anthropic] {type(e).__name__}: {str(e)[:160]}", flush=True)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                continue
            print(f"  [anthropic] {type(e).__name__}: {str(e)[:160]}", flush=True)
            return None
    return None


def _invoke_bedrock(client, model_id, prompt, max_tokens, temperature, max_retries):
    from botocore.exceptions import ClientError
    throttle = {"ThrottlingException", "TooManyRequestsException",
                "ServiceUnavailableException", "ModelTimeoutException"}
    # 6 retries of 2**attempt backoff covers only ~63s, which is short of what sustained
    # Opus throttling needs; exhausting it dropped ~21% of the pool silently on the first run.
    for attempt in range(max_retries):
        try:
            r = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature})
            text = r["output"]["message"]["content"][0]["text"]
            if r.get("stopReason") == "max_tokens":
                # Truncation is not random: it hits the verbosity_high pole almost exclusively,
                # so a clipped mid-sentence tail would correlate with one concept axis and plant
                # an artifact in the very dimension the control tests. Drop rather than keep.
                print(f"  [truncated at max_tokens, discarding] {prompt[-60:]!r}", flush=True)
                return None
            return text
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in throttle and attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                continue
            # Silent failures are indistinguishable from refusals or bad configuration; a run
            # that quietly drops a fifth of its output looks healthy until the counts are read.
            print(f"  [bedrock] {code or type(e).__name__}: {str(e)[:160]}", flush=True)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(60, 2 ** attempt) * (1 + random.random()))
                continue
            print(f"  [bedrock] {type(e).__name__}: {str(e)[:160]}", flush=True)
            return None
    return None


def generate_pool(styles, prompts, model_id, workers, out_file, max_tokens=4000):
    """One response per (prompt, style). Resumable: existing entries are kept."""
    client = make_client()
    pool = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            pool = json.load(f)

    jobs = [(pi, p, s) for pi, p in enumerate(prompts) for s in styles
            if f"{pi}|{s['style_id']}" not in pool]
    print(f"{len(jobs)} responses to generate ({len(pool)} already present)")

    def one(job):
        pi, p, s = job
        text = invoke(client, model_id, max_tokens=max_tokens,
                      prompt=f"""Answer the user's question below.

Write the answer in this style:
- {s['instruction']}

Keep the answer factually correct and genuinely responsive to the question. Do not mention these
style instructions, and do not comment on your own writing style.

Question: {p}""")
        return f"{pi}|{s['style_id']}", pi, s["style_id"], text

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, pi, sid, text in ex.map(one, jobs):
            done += 1
            if text:
                pool[key] = {"prompt_idx": pi, "style_id": sid, "response": text}
            else:
                failed += 1
            if done % 25 == 0:
                with open(out_file, "w") as f:
                    json.dump(pool, f, indent=2)
                print(f"  {done}/{len(jobs)}  (failed {failed})", flush=True)

    with open(out_file, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"wrote {len(pool)} responses to {out_file} ({failed} failed)")


# --------------------------------------------------------------------------- pairs

def build_pairs(personas, prompts, pool, emb, concepts, cv, pairs_per_user, margin, seed):
    """Score every response under every persona and emit labelled pairs.

    utility_u(r) = sum_c w_u[c] * z_c(r), where z_c(r) is the response's projection onto the unit
    concept direction, standardised across the whole pool so concepts contribute comparably.
    Pairs are drawn within a prompt, and only kept when the utility gap exceeds `margin`, which
    keeps ambiguous pairs out rather than labelling them by numerical noise.
    """
    rng = random.Random(seed)
    keys = sorted(pool)
    E = torch.stack([emb[k] for k in keys]).float()

    C = F.normalize(torch.stack([cv[c].float().reshape(-1) for c in concepts], 1), dim=0)
    Z = E @ C                                   # [N, k] raw projections
    Z = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)

    by_prompt = {}
    for i, k in enumerate(keys):
        by_prompt.setdefault(pool[k]["prompt_idx"], []).append(i)

    out, kept_frac = [], []
    for s in personas:
        w = torch.tensor([s["weights"].get(c, 0.0) for c in concepts])
        u = Z @ w
        cands = [pi for pi, idxs in by_prompt.items() if len(idxs) >= 2]
        made, tried = 0, 0
        while made < pairs_per_user and tried < pairs_per_user * 40:
            tried += 1
            pi = rng.choice(cands)
            a, b = rng.sample(by_prompt[pi], 2)
            gap = float(u[a] - u[b])
            if abs(gap) < margin:
                continue
            hi, lo = (a, b) if gap > 0 else (b, a)
            out.append({
                "user_id": s["user_id"],
                "prompt_idx": pi,
                # Pool keys let the runner look embeddings up directly, and prompt_idx is the
                # split unit: with only 30 prompts, splitting by PAIR would put the same prompt
                # and often the same response texts on both sides, letting a trained model
                # memorise "for this prompt, response X wins" and score its test sibling for
                # free. That is the Community Alignment leak in a smaller pool.
                # Keys only, never the response text: 7,200 pairs drawn from a 600-response pool
                # duplicate that text ~12x each, turning a 1.7 MB pool into a 41 MB pairs file.
                # Resolve through response_pool.json instead.
                "chosen_key": keys[hi],
                "rejected_key": keys[lo],
                "chosen_style": pool[keys[hi]]["style_id"],
                "rejected_style": pool[keys[lo]]["style_id"],
                "utility_gap": abs(gap),
            })
            made += 1
        kept_frac.append(made / max(1, tried))
        if made < pairs_per_user:
            print(f"  {s['user_id']}: only {made}/{pairs_per_user} pairs cleared margin")

    print(f"{len(out)} pairs, mean accept rate {sum(kept_frac) / len(kept_frac):.2f}")

    # Disagreement check: the point of the shared pool. If users never disagree about an ordered
    # pair, a single global direction explains everything and the dataset tests nothing.
    seen = {}
    for r in out:
        # Keyed on pool keys, not response text: the records no longer carry the text, and keys
        # identify a response exactly rather than by an 80-character prefix.
        kp = (r["chosen_key"], r["rejected_key"])
        rp = (r["rejected_key"], r["chosen_key"])
        if rp in seen:
            seen[rp] = "both"
        else:
            seen.setdefault(kp, "one")
    both = sum(1 for v in seen.values() if v == "both")
    print(f"contested orderings (some user prefers A>B, another B>A): {both}/{len(seen)} "
          f"= {both / max(1, len(seen)):.2f}")
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["personas", "generate", "pairs"])
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="screened concept set; defaults to all of concept_library_v2")
    ap.add_argument("--n_users", type=int, default=60)
    ap.add_argument("--n_pref", type=int, default=2)
    ap.add_argument("--n_dis", type=int, default=2)
    ap.add_argument("--max_cos", type=float, default=0.5)
    ap.add_argument("--weight_mode", choices=["continuous", "binary"], default="continuous")
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--pairs_per_user", type=int, default=120)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=4000,
                    help="raise on a resume pass to fill cells lost to truncation; the "
                         "verbosity_high pole is the one that hits the cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vectors", default="data/prism/concept_vectors_v2.pt")
    ap.add_argument("--embeddings", default=f"{OUT_DIR}/pool_embeddings.pt")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    concepts = args.concepts or list(CONCEPT_LIBRARY_V2)

    if args.stage == "personas":
        cm = None
        if os.path.exists(args.vectors):
            cv = torch.load(args.vectors, weights_only=False)
            missing = [c for c in concepts if c not in cv]
            if missing:
                sys.exit(f"concept vectors missing for {missing}")
            cm = F.normalize(torch.stack([cv[c].float().reshape(-1) for c in concepts], 1), dim=0)
            if os.path.exists(args.embeddings):
                # Scale columns by 1/sd so separation is measured on the direction the labels are
                # actually generated from, C @ (w / sd). Without the pool we cannot know sd.
                emb = torch.load(args.embeddings, weights_only=False)
                P = torch.stack([emb[k] for k in sorted(emb)]).float() @ cm
                cm = cm / P.std(0).clamp_min(1e-6)
                print("separation measured on the sigma-scaled (true planted) basis")
            else:
                print(f"WARNING: {args.embeddings} not found; separation will be measured on the "
                      f"unit concept basis, which is NOT the direction labels come from")
        else:
            print(f"WARNING: {args.vectors} not found, falling back to weight-space separation")
        personas = design_personas(concepts, args.n_users, args.n_pref, args.n_dis,
                                   args.max_cos, args.seed, weight_mode=args.weight_mode,
                                   concept_matrix=cm)
        for s in personas[:5]:
            print(f"  {s['user_id']}  +{'/'.join(s['preferred'])}  "
                  f"-{'/'.join(s['dispreferred'])}")
        if args.dry_run:
            print("\ndry run, nothing written")
            return
        with open(PERSONA_FILE, "w") as f:
            json.dump({"concepts": concepts, "personas": personas}, f, indent=2)
        print(f"wrote {PERSONA_FILE}")

    elif args.stage == "generate":
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "bedrock"
        model_id = model_id_for(provider)
        if not model_id:
            sys.exit("no model id: set ANTHROPIC_API_KEY (+ANTHROPIC_MODEL_ID) or "
                     "BEDROCK_MODEL_ID")
        print(f"[{provider}] model={model_id}", flush=True)
        with open(PERSONA_FILE) as f:
            blob = json.load(f)
        prompts = load_prompts(args.n_prompts, args.seed, f"{OUT_DIR}/prompts.json")
        styles = pool_styles(blob["concepts"], CONCEPT_LIBRARY_V2)
        n = len(prompts) * len(styles)
        print(f"{len(prompts)} prompts x {len(styles)} styles = {n} responses "
              f"(independent of the {len(blob['personas'])} personas)")
        if args.dry_run:
            print("dry run, no API calls, nothing written")
            return
        # Written only once we are actually committing to generate against these prompts:
        # prompt_idx is the key the response pool is stored under, so a prompts.json that
        # disagrees with the pool silently invalidates it.
        with open(f"{OUT_DIR}/prompts.json", "w") as f:
            json.dump(prompts, f, indent=2)
        generate_pool(styles, prompts, model_id, args.workers, POOL_FILE,
                      max_tokens=args.max_tokens)

    else:
        with open(PERSONA_FILE) as f:
            blob = json.load(f)
        with open(f"{OUT_DIR}/prompts.json") as f:
            prompts = json.load(f)
        with open(POOL_FILE) as f:
            pool = json.load(f)
        if not os.path.exists(args.embeddings):
            sys.exit(f"{args.embeddings} not found -- embed the response pool on GPU first")
        emb = torch.load(args.embeddings, weights_only=False)
        cv = torch.load(args.vectors, weights_only=False)
        rows = build_pairs(blob["personas"], prompts, pool, emb, blob["concepts"], cv,
                           args.pairs_per_user, args.margin, args.seed)
        with open(PAIRS_FILE, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {PAIRS_FILE}")


if __name__ == "__main__":
    main()
