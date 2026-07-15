"""Judge prompt construction. Pure functions only -- no hidden state -- so the
prompt is unit-testable and the cache key over its inputs is meaningful.

The runner owns position swapping; this module always labels the two answers
"Answer A" (slot 1) and "Answer B" (slot 2) and never knows a swap happened.
"""

from __future__ import annotations

from dataclasses import dataclass

from .concepts import Concept

SYSTEM = (
    'You are an impartial evaluator comparing two AI assistant responses -- '
    '"Answer A" and "Answer B" -- to the same user prompt.\n\n'
    "Your job: for each concept you are given, judge WHICH answer exhibits MORE of that "
    "concept, on a 0-to-1 scale. This is a directional comparison between the two answers, "
    "not a quality rating of either answer on its own.\n\n"
    "Score every concept INDEPENDENTLY. Do not let an overall preference for one answer pull "
    "its scores on unrelated concepts in the same direction. Answer A may exhibit more of one "
    "concept while Answer B exhibits more of another.\n\n"
    "Respond with a single JSON object and nothing else: no prose, no explanation, and no "
    "markdown code fences."
)


@dataclass(frozen=True)
class JudgePrompt:
    system: str
    user: str
    concept_keys: tuple[str, ...]


def _render_concepts(concepts: list[Concept]) -> str:
    return "\n".join(f"- {c.key} ({c.label}): {c.definition}" for c in concepts)


def build_prompt(prompt: str, answer_a: str, answer_b: str, concepts: list[Concept]) -> JudgePrompt:
    """Build the judge prompt for one item over the given concepts.

    Pass all concepts for a joint call, or a single-element list for --per-concept.
    """
    keys = tuple(c.key for c in concepts)
    example = "{" + ", ".join(f'"{k}": 0.5' for k in keys) + "}"

    user = (
        "<user_prompt>\n"
        f"{prompt}\n"
        "</user_prompt>\n\n"
        "<answer_a>\n"
        f"{answer_a}\n"
        "</answer_a>\n\n"
        "<answer_b>\n"
        f"{answer_b}\n"
        "</answer_b>\n\n"
        "<concepts>\n"
        f"{_render_concepts(concepts)}\n"
        "</concepts>\n\n"
        "<scale>\n"
        "For each concept, output a single number in [0, 1] indicating which answer exhibits "
        "MORE of that concept:\n"
        "  0.00 = Answer A exhibits it clearly more\n"
        "  0.25 = Answer A exhibits it somewhat more\n"
        "  0.50 = the two answers are equivalent on this concept\n"
        "  0.75 = Answer B exhibits it somewhat more\n"
        "  1.00 = Answer B exhibits it clearly more\n"
        "Any value in [0, 1] is allowed; the anchors are guidance. This is about WHICH answer, "
        "A or B, shows more of the concept -- not about how good either answer is.\n"
        "</scale>\n\n"
        "<output_format>\n"
        "Return a single JSON object and nothing else -- no prose, no markdown code fences, no "
        "explanation. It must contain every one of these keys, each mapped to a number in "
        "[0, 1]:\n"
        f"{', '.join(keys)}\n"
        "Example of the required shape (values are illustrative only):\n"
        f"{example}\n"
        "</output_format>"
    )
    return JudgePrompt(system=SYSTEM, user=user, concept_keys=keys)
