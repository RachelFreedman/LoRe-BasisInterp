"""Does the SAE support naming when the readout is NOT bottlenecked through ten counters?

Pre-registered before running:
  Testing  -- whether the SAE's loss to the direct-correlation baseline survives
              replacing the ten-surface-measure readout with an LLM judge, which is
              the readout the method actually proposes.
  How      -- for each planted direction, both routes select a preferred and a
              dispreferred set of PRISM responses and show them to the same judge
              with the same prompt and the same concept library. Only the selection
              differs:
                  SAE route   text weight = sum over top-k features of align_f * z_tf
                  baseline    text weight = v . e_t
              The judge returns (concept, sign) as JSON. Identical budget: same number
              of texts per side, same truncation, same prompt.
  Outcomes -- SAE at or above the baseline means the earlier deficit was an artifact
              of the counter readout. SAE still below means the deficit is real and
              survives removing the bottleneck.

  Controls, on the SAME evidence budget as the routes (declared before running:
  16 texts per side, 600 characters each, applied uniformly to all four conditions
  so nothing is tuned on accuracy):
    ceiling -- show the judge the REAL contrast pairs for each concept. If it cannot
               name those, the prompt or the judge is the bottleneck and no route
               number below means anything. This bounds every other row.
    null    -- random unit directions through the SAE route. Should sit at chance;
               if it does not, the selection or the prompt is leaking.

The judge is deliberately NOT a Claude model: the concept contrast sets were written
by Claude Sonnet 4.5, so a Claude judge naming them would not be independent.

Credentials come from FOUNDRY_KEY / FOUNDRY_BASE in the environment; nothing is
written to disk. Window frozen at the value from readout_width_check.py.
"""
import argparse, csv, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "PRISM"))
from sae.scripts.feature_text_profiles import prism_texts, load_sae  # noqa: E402
from PRISM.concept_library import CONCEPT_LIBRARY  # noqa: E402

PROMPT = """You are analysing a preference direction learned from human feedback.

Below are two sets of AI assistant responses. Set A is PREFERRED by this direction;
Set B is DISPREFERRED. The difference between the sets reflects exactly one concept
from the library below.

CONCEPT LIBRARY
{library}

SET A -- preferred
{set_a}

SET B -- dispreferred
{set_b}

Which single concept best explains the difference, and does the direction prefer
HIGH or LOW of that concept? Reply with JSON only, no other text:
{{"concept": "<one name from the library>", "sign": "high" or "low"}}"""


FAILURES = []


def sign_test(a, b):
    w = sum(x > y for x, y in zip(a, b)); l = sum(x < y for x, y in zip(a, b))
    n = w + l
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, len(a) - n, p


def render(texts, idx, chars):
    return "\n\n".join(f"[{i+1}] {texts[j][:chars]}" for i, j in enumerate(idx))


def ask(client, model, prompt, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=200, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            raw = r.choices[0].message.content.strip()
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
            d = json.loads(raw)
            return str(d["concept"]).strip().lower(), str(d["sign"]).strip().lower(), r.usage
        except Exception as e:
            if attempt == retries - 1:
                print(f"      judge failed: {type(e).__name__} {str(e)[:100]}")
                FAILURES.append(1)
                return None, None, None
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "sae/checkpoints/d3/model.pt"))
    ap.add_argument("--profiles", default=str(REPO / "results/planted/feature_text_profiles.pt"))
    ap.add_argument("--directions", default=str(REPO / "results/planted/planted_directions.pt"))
    ap.add_argument("--sae-data", default=str(REPO / "sae/data"))
    ap.add_argument("--prism-train", default=str(REPO / "PRISM/data/prism/train_embeddings.pkl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-texts", type=int, default=16, help="texts per side (all conditions)")
    ap.add_argument("--chars", type=int, default=600, help="truncation per text (all conditions)")
    ap.add_argument("--contrast-pairs", default=str(REPO / "data/prism/contrastive_pairs.json"))
    ap.add_argument("--n-null", type=int, default=12, help="random directions for the null")
    ap.add_argument("--model", default="DeepSeek-V4-Pro")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", default=str(REPO / "results/planted/llm_judge.json"))
    ap.add_argument("--rows", default=str(REPO / "results/planted/llm_judge_rows.csv"))
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=os.environ["FOUNDRY_BASE"], api_key=os.environ["FOUNDRY_KEY"])
    library = "\n".join(f"- {k}: {v['description']}" for k, v in CONCEPT_LIBRARY.items())
    valid = set(CONCEPT_LIBRARY)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_sae(Path(a.checkpoint), dev)
    dec = model.decoder.weight.detach().cpu().float()
    blob = torch.load(a.profiles, weights_only=False)
    activity = blob["activity"].numpy() * (np.abs(blob["profiles"]).sum(1) > 0)
    payload = torch.load(a.directions, weights_only=False)

    meta = [json.loads(l) for l in open(Path(a.sae_data) / "metadata.jsonl")]
    meta = [m for m in meta if m["sae_split"] == a.split]
    meta.sort(key=lambda m: m["sae_split_index"])
    X = torch.load(Path(a.sae_data) / f"sae_{a.split}.pt", map_location="cpu").float()
    texts = prism_texts(Path(a.prism_train), meta, a.split)
    keep = [i for i, t in enumerate(texts) if t]
    X = X[keep]; texts = [texts[i] for i in keep]
    print(f"{len(texts)} PRISM texts | judge={a.model} | window=top{a.window} | "
          f"{a.n_texts} texts/side | seeds={a.seeds}")

    rows_, cols_, vals_ = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), a.batch_size):
            z = model.encode(X[i:i + a.batch_size].to(dev)).cpu()
            nz = z.nonzero(as_tuple=False)
            rows_.append(nz[:, 0].numpy() + i); cols_.append(nz[:, 1].numpy())
            vals_.append(z[nz[:, 0], nz[:, 1]].numpy())
    Z = sparse.csr_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))),
                          shape=(len(X), dec.shape[1]))

    tok_in = tok_out = 0
    EVAL = payload["concepts"]

    # ---- ceiling: the judge sees the REAL contrast text for each concept ----
    cpairs = json.load(open(a.contrast_pairs))
    ceil_hits, ceil_rows = 0, []
    for c in EVAL:
        hi = [p["high_response"] for p in cpairs[c]][:a.n_texts]
        lo = [p["low_response"] for p in cpairs[c]][:a.n_texts]
        for sgn, (A, B) in (("high", (hi, lo)), ("low", (lo, hi))):
            prompt = PROMPT.format(library=library,
                                   set_a=render(A, range(len(A)), a.chars),
                                   set_b=render(B, range(len(B)), a.chars))
            pc, ps, u = ask(client, a.model, prompt)
            if u: tok_in += u.prompt_tokens; tok_out += u.completion_tokens
            ok = int(pc == c and ps == sgn)
            ceil_hits += ok
            ceil_rows.append({"condition": "ceiling", "true_concept": c, "true_sign": sgn,
                              "pred": pc, "pred_sign": ps, "correct": ok})
    ceiling = ceil_hits / len(ceil_rows)
    print(f"   CEILING (real contrast text): {ceiling:.3f}  ({ceil_hits}/{len(ceil_rows)})")

    out_rows, sae_acc, base_acc = list(ceil_rows), [], []
    recs = [r for r in payload["records"] if r["seed"] in a.seeds]
    for rec in recs:
        sh = bh = n = 0
        for g, (tc, ts) in enumerate(rec["labels"]):
            v = F.normalize(rec["group_dirs"][g], dim=0)
            align = (dec.T @ v).numpy()
            top = np.argsort(np.abs(align) * activity)[::-1][:a.window]
            weights = {"sae": np.asarray(Z[:, top] @ align[top]).ravel(),
                       "nosae": (X @ v).numpy()}
            preds = {}
            for route, w in weights.items():
                order = np.argsort(w)
                prompt = PROMPT.format(library=library,
                                       set_a=render(texts, order[::-1][:a.n_texts], a.chars),
                                       set_b=render(texts, order[:a.n_texts], a.chars))
                c, s, u = ask(client, a.model, prompt)
                if u: tok_in += u.prompt_tokens; tok_out += u.completion_tokens
                preds[route] = (c if c in valid else None, s)
            sok = int(preds["sae"] == (tc, ts)); bok = int(preds["nosae"] == (tc, ts))
            sh += sok; bh += bok; n += 1
            out_rows.append({"condition": "route", "seed": rec["seed"],
                             "true_concept": tc, "true_sign": ts,
                             "sae_pred": preds["sae"][0], "sae_sign": preds["sae"][1],
                             "nosae_pred": preds["nosae"][0], "nosae_sign": preds["nosae"][1],
                             "sae_correct": sok, "nosae_correct": bok})
        sae_acc.append(sh / n); base_acc.append(bh / n)
        print(f"   seed {rec['seed']}: SAE {sae_acc[-1]:.3f}   no-SAE {base_acc[-1]:.3f}")

    # ---- null: random unit directions through the SAE route ----
    rng = np.random.default_rng(0)
    truth = [(c, s_) for c in EVAL for s_ in ("high", "low")]
    null_hits = 0
    for i in range(a.n_null):
        v = torch.tensor(rng.normal(size=dec.shape[0]), dtype=torch.float32)
        v = F.normalize(v, dim=0)
        align = (dec.T @ v).numpy()
        top = np.argsort(np.abs(align) * activity)[::-1][:a.window]
        w = np.asarray(Z[:, top] @ align[top]).ravel()
        order = np.argsort(w)
        prompt = PROMPT.format(library=library,
                               set_a=render(texts, order[::-1][:a.n_texts], a.chars),
                               set_b=render(texts, order[:a.n_texts], a.chars))
        pc, ps, u = ask(client, a.model, prompt)
        if u: tok_in += u.prompt_tokens; tok_out += u.completion_tokens
        null_hits += sum(int(pc == tc and ps == ts) for tc, ts in truth) / len(truth)
        out_rows.append({"condition": "null", "true_concept": "-", "true_sign": "-",
                         "pred": pc, "pred_sign": ps, "correct": ""})
    null_acc = null_hits / a.n_null
    print(f"   NULL (random directions): {null_acc:.3f}")

    W, L, T, p = sign_test(sae_acc, base_acc)
    per = {}
    for r in (x for x in out_rows if x.get("condition") == "route"):
        d = per.setdefault(r["true_concept"], [0, 0, 0])
        d[0] += r["sae_correct"]; d[1] += r["nosae_correct"]; d[2] += 1
    fields = sorted({k for r in out_rows for k in r})
    with open(a.rows, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval=""); wr.writeheader(); wr.writerows(out_rows)
    json.dump({"judge": a.model, "ceiling": ceiling, "null": null_acc,
               "window": a.window, "n_texts_per_side": a.n_texts,
               "chars": a.chars, "seeds": a.seeds, "chance": 1 / (2 * len(CONCEPT_LIBRARY)),
               "sae_mean": float(np.mean(sae_acc)), "nosae_mean": float(np.mean(base_acc)),
               "sae_per_seed": sae_acc, "nosae_per_seed": base_acc,
               "sign_test": {"wins": W, "losses": L, "ties": T, "p": p},
               "per_concept": {k: {"sae": v[0], "nosae": v[1], "n": v[2]} for k, v in per.items()},
               "tokens": {"input": tok_in, "output": tok_out},
               "judge_call_failures": len(FAILURES)}, open(a.out, "w"), indent=2)

    print(f"\n=== LLM judge ({a.model}), {len(sae_acc)} seeds ===")
    print(f"  chance  {1/(2*len(CONCEPT_LIBRARY)):.3f}")
    print(f"  null    {null_acc:.3f}   (random directions)")
    print(f"  CEILING {ceiling:.3f}   (real contrast text -- bounds everything below)")
    print(f"  SAE     {np.mean(sae_acc):.3f} +/- {np.std(sae_acc):.3f}")
    print(f"  no-SAE  {np.mean(base_acc):.3f} +/- {np.std(base_acc):.3f}")
    print(f"  sign test: {W}W-{L}L-{T}T   p = {p:.3f}")
    print("\n  per concept (SAE / no-SAE of n):")
    for c, v in per.items(): print(f"    {c:<12} {v[0]:>3} / {v[1]:>3}  of {v[2]}")
    print(f"\n  tokens: {tok_in} in, {tok_out} out | failed judge calls: {len(FAILURES)}")


if __name__ == "__main__":
    main()
