# Synthetic preference dataset — design decisions

Owner and red-teamers: anonymized for review.

The draft spec builds the dataset on the existing library of 11 concepts:
`{helpfulness, fluency, factuality, safety, diversity, creativity, values, confidence,
formatting, sycophancy, repetition}`. It does not, and this records why, with the measurements
behind each choice. Everything else follows the draft.

## 1. The concept set

Personas are built over a screened set of 6-8 axes drawn from a 12-concept candidate library,
rather than over all 11 existing concepts.

The 11 concepts do not span 11 directions. Measured on `data/prism/concept_vectors.pt`:

| set | n | eff. rank | top-1 var | max abs cos |
|---|---|---|---|---|
| all 11 | 11 | **3.48** | 67% | 0.98 |
| quality cluster only | 5 | **1.54** | 90% | 0.98 |
| stylistic survivors | 6 | **4.00** | 52% | 0.68 |

Effective rank = `exp(entropy of the singular-value variance spectrum)` over unit-normalised
concept vectors.

The five quality concepts — helpfulness, factuality, safety, values, sycophancy — have an
effective rank of 1.54 between them. `safety` and `values` sit at cos **0.98**: the same vector
under two names. `values`↔`sycophancy` is −0.92, `helpfulness`↔`factuality` +0.87.

This matters because the spec asks for users with "a strong preference towards one or
more concepts, and a strong dispreference for other concepts." Over that library, a
persona preferring `safety` and dispreferring `sycophancy` is not expressing two preferences —
those directions are 0.90 apart with opposite sign, so the persona is approximately 1.9× a single
axis. Two personas built from different members of the quality cluster would be near-identical by
construction. LoRe-v2 could then recover the planted structure **correctly** and still score near
zero on per-user recovery, because the planted users were never distinguishable.

A positive control must not be able to fail for reasons unrelated to what it tests. That is the reason for the change.

**Consequence:** the existing library is a strictly worse instrument than a subset of itself.
Dropping the five quality concepts *raises* effective rank from 3.48 to 4.00.

## 2. Quality is represented by `diversity`, not `helpfulness`

Dropping quality entirely would make the control unrepresentative, so one quality axis is kept.
It is not one of the five quality concepts. Adding any of them to the stylistic survivors makes
the set worse on both measures:

| set | eff. rank | max abs cos |
|---|---|---|
| style 5 + **diversity** | **3.996** | **0.682** |
| style 5 + values | 3.959 | 0.761 |
| style 5 + factuality | 3.830 | 0.776 |
| style 5 + helpfulness | 3.702 | 0.805 |

`diversity` already carries the quality direction — it sits at cos 0.84–0.89 from every member of
the quality cluster, and is the nearest kept concept to all five of them — while being less
entangled than any explicit quality concept. It is the quality axis in this library.

**Caveat carried forward:** the name `diversity` understates what the vector measures. Anywhere it
appears as a persona axis it should be read as the quality/goodness direction, not as
"presents multiple viewpoints".

## 3. Six new stylistic candidates

Added and screened: `verbosity`, `formality`, `concreteness`, `technicality`, `warmth`, `humor`.

Design constraint, learned from why the original library lost its rank: **neither pole may be the
worse response**. Every concept whose low pole reads as "bad" collapses onto the quality axis.
Both poles must be defensible stylistic choices real readers could differ on.

Two candidates from the initial plan were cut before generation, on inspection rather than
measurement: `hedging` is `confidence` inverted, and `directness` overlaps both `confidence` and
`verbosity`.

Screening is greedy on effective rank (`PRISM/screen_concepts.py`); the surviving set is whatever
the measurement returns, not a set chosen in advance. Target 6–8 real axes from 12 candidates. If
the additions correlate as badly as the quality cluster did, the fallback is the existing 6.

## 4. Regenerated with a different model, to a new file

All 12 concepts are generated with **Claude Opus 4.6** (`eu.anthropic.claude-opus-4-6-v1`). The
original 11 were generated with Claude Sonnet 4.5.

The reason is not preference for the newer model. The synthetic dataset's *responses* will be
Opus 4.6 text. If the concept axes were estimated from Sonnet 4.5 text while the data exhibits
Opus 4.6 style, the planted ground-truth direction and the actual variation in the data come from
different generators — another way for the control to fail for a reason unrelated to the method.
Same generator on both sides removes it.

Outputs are written to `data/prism/contrastive_pairs_v2.json` and
`data/prism/concept_vectors_v2.pt`. **`contrastive_pairs.json` and `concept_vectors.pt` are not
touched**, so no committed result changes.

The six base concepts therefore exist in both files with byte-identical prompt text, differing
only in generator. `cos(v_Sonnet, v_Opus)` per concept measures how much a concept vector is an
artifact of which model wrote the contrastive text.

**Measured, and it is large.** Split-half reliability within a single generator (same model, same
text, half the pairs in each estimate) is the noise floor any cross-generator number must be read
against:

| concept | cross-generator | split-half (Opus) | split-half (Sonnet) |
|---|---|---|---|
| repetition | **-0.248** | 0.850 | 0.945 |
| confidence | +0.758 | 0.972 | 0.952 |
| fluency | +0.775 | 0.971 | 0.942 |
| diversity | +0.831 | 0.956 | 0.986 |
| formatting | +0.930 | 0.990 | 0.981 |
| creativity | +0.953 | 0.961 | 0.981 |
| **mean** | **+0.667** | ~0.95 | ~0.96 |

Every concept is well determined within its own generator (0.85-0.99), so the gap is not
estimation noise -- an explanation considered and falsified. Generator identity genuinely moves
the vector. For `repetition` the two generators produce directions pointing opposite ways from
byte-identical definitions, with both sets of contrastive text correct on inspection.

**Implication beyond this dataset.** Concept-vector analyses -- including our own
`reports/wbar_concept_alignment.md` and reward-lens-style alignment work generally -- treat one
generator's vectors as *the* concept direction. Any such alignment number carries an unreported
generator-dependence term of roughly this size, which for weakly determined concepts can exceed
the signal being measured.

## Known limitation

Ground-truth preference labels are generated by projecting response embeddings onto the concept
vectors and scoring with the persona's weights. This places the planted reward exactly inside
LoRe's model class — a linear function of the embedding. It is the best case by construction.

**A pass here shows the instrument can recover per-user structure when that structure is present
and linearly encoded. It does not show LoRe-v2 recovers personalization in real data.** The
spec is explicit that realism is not the goal, so this is in keeping with it, but the claim it
licenses is correspondingly narrow and should be stated that way in the write-up.

An LLM-judge labelling variant, where a model role-plays each persona and picks a winner, would be
less tautological and noisier. Worth running on a subset as a check that projection labels and
judged labels agree.

## Files

| file | role |
|---|---|
| `PRISM/concept_library_v2.py` | 6 base + 6 candidate definitions |
| `PRISM/generate_contrastive_pairs_v2.py` | Bedrock generation, resumable, per-concept output |
| `PRISM/compute_concept_vectors_v2.py` | batched embedding, refuses to write `concept_vectors.pt` |
| `modal_compute_vectors_v2.py` | GPU wrapper (the v1 wrapper hardcodes the v1 output path) |
| `PRISM/screen_concepts.py` | generator drift, greedy selection, v_pop coverage |
