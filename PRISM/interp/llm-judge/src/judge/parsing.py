"""Defensive parsing of a judge model's raw text into per-concept scores.

Models are told to emit a bare JSON object, but they misbehave: markdown fences,
leading prose, a refusal, a trailing comma. We try hard to recover a valid object
without ever *guessing* a score -- if a concept is missing or out of range we fail
the parse rather than invent a number. No clamping: an out-of-range value is a bug
in the model's output, and silently squashing it to [0, 1] would hide that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    OK = "ok"
    PARSE_ERROR = "parse_error"          # could not recover a JSON object at all
    VALIDATION_ERROR = "validation_error"  # got JSON, but keys/values are wrong
    REFUSAL = "refusal"                  # model declined to answer
    API_ERROR = "api_error"              # the API call itself failed (set by the runner)


@dataclass(frozen=True)
class ParseResult:
    status: Status
    scores: dict[str, float] | None  # populated iff status is OK
    detail: str                      # human-readable reason when not OK; "" when OK


# Substrings that, in the ABSENCE of any recoverable JSON, signal a refusal rather
# than a mangled answer. Kept deliberately small and specific to avoid false hits.
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
)


def _iter_json_spans(text: str):
    """Yield every balanced ``{...}`` span in ``text``, left to right.

    Brace-counting that respects JSON string literals and escapes, so a ``}`` inside
    a quoted value does not close the object early. Yielding all top-level spans (not
    just the first) lets the caller skip an earlier ``{...}`` that is not valid JSON.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escaped = False
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : j + 1]
                    break
            j += 1
        i = j + 1


def _load_object(raw_text: str) -> dict | None:
    """Best-effort recovery of a JSON object from possibly-noisy model text."""
    stripped = raw_text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for span in _iter_json_spans(stripped):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_scores(raw_text: str, concept_keys: tuple[str, ...]) -> ParseResult:
    """Parse ``raw_text`` into a score for every key in ``concept_keys``.

    Success requires exactly the expected keys to be present with numeric values in
    [0, 1]. Extra keys are ignored (with a note); missing keys, non-numeric values,
    booleans, or out-of-range numbers are validation errors. If no JSON object can be
    recovered, an empty/refusal-looking response maps to REFUSAL, otherwise PARSE_ERROR.
    """
    obj = _load_object(raw_text)
    if obj is None:
        lowered = raw_text.lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return ParseResult(Status.REFUSAL, None, "model refused to answer")
        return ParseResult(Status.PARSE_ERROR, None, "no JSON object found in response")

    missing = [k for k in concept_keys if k not in obj]
    if missing:
        return ParseResult(
            Status.VALIDATION_ERROR, None, f"missing keys: {', '.join(missing)}"
        )

    scores: dict[str, float] = {}
    for k in concept_keys:
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return ParseResult(
                Status.VALIDATION_ERROR, None, f"value for '{k}' is not a number: {v!r}"
            )
        f = float(v)
        if not (0.0 <= f <= 1.0):
            return ParseResult(
                Status.VALIDATION_ERROR, None, f"value for '{k}' out of range [0, 1]: {f}"
            )
        scores[k] = f

    extra = [k for k in obj if k not in concept_keys]
    detail = f"ignored extra keys: {', '.join(extra)}" if extra else ""
    return ParseResult(Status.OK, scores, detail)
