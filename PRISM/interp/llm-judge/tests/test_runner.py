"""Runner tests: swap math, NaN handling, caching, and input-order guarantee."""

from __future__ import annotations

import json
import math

from judge.cache import Cache
from judge.concepts import Concept
from judge.parsing import Status
from judge.providers.fake import FakeProvider
from judge.runner import run_model
from judge.schema import InputItem

CONCEPTS = [
    Concept(key="helpfulness", label="Helpfulness", definition="how useful"),
    Concept(key="fluency", label="Fluency", definition="how smooth"),
]


def _is_forward(user: str) -> bool:
    """Forward pass puts the original answer_a ('A') in the <answer_a> slot."""
    return "<answer_a>\nA\n</answer_a>" in user


def _run(provider, items, **over):
    kw = dict(temperature=0.0, concept_set_version="v1", cache=Cache("", enabled=False))
    kw.update(over)
    return run_model(provider, items, CONCEPTS, **kw)


def test_swap_math_final_and_disagreement():
    # forward: slot-B (answer_b) scored 0.8; reverse: slot-B (answer_a) scored 0.1
    def responder(system, user, temperature):
        if _is_forward(user):
            return json.dumps({"helpfulness": 0.8, "fluency": 0.5})
        return json.dumps({"helpfulness": 0.1, "fluency": 0.5})

    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    r = _run(FakeProvider(responder=responder), items)[0]

    # fwd_b=0.8, rev_b=1-0.1=0.9 -> final=0.85, disagreement=0.1
    assert math.isclose(r.final["helpfulness"], 0.85)
    assert math.isclose(r.disagreement["helpfulness"], 0.1, abs_tol=1e-9)
    # fluency symmetric: fwd_b=0.5, rev_b=0.5 -> final 0.5, no disagreement
    assert math.isclose(r.final["fluency"], 0.5)
    assert math.isclose(r.disagreement["fluency"], 0.0, abs_tol=1e-9)


def test_perfectly_consistent_judge_has_zero_disagreement():
    def responder(system, user, temperature):
        # answer_b consistently exhibits more: forward 0.9, reverse (b in slot A) 0.1
        val = 0.9 if _is_forward(user) else 0.1
        return json.dumps({"helpfulness": val, "fluency": val})

    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    r = _run(FakeProvider(responder=responder), items)[0]
    assert math.isclose(r.final["helpfulness"], 0.9)
    assert math.isclose(r.disagreement["helpfulness"], 0.0, abs_tol=1e-9)


def test_parse_failure_yields_nan_for_all_concepts():
    def responder(system, user, temperature):
        if _is_forward(user):
            return json.dumps({"helpfulness": 0.5, "fluency": 0.5})
        return "I won't answer."

    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    r = _run(FakeProvider(responder=responder), items)[0]
    assert r.reverse.status is Status.REFUSAL
    assert all(math.isnan(v) for v in r.final.values())
    assert all(math.isnan(v) for v in r.disagreement.values())


def test_api_error_yields_nan_and_api_error_status():
    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    r = _run(FakeProvider(error="boom"), items)[0]
    assert r.forward.status is Status.API_ERROR
    assert all(math.isnan(v) for v in r.final.values())


def test_results_are_input_ordered_despite_out_of_order_completion():
    # Later items return faster, so completion order is reversed vs input order.
    n = 6

    class ReverseDelayProvider(FakeProvider):
        def complete(self, system, user, temperature):
            # slower for earlier items: index encoded in the prompt text
            import time

            for i in range(n):
                if f"prompt {i}\n" in user:
                    time.sleep((n - i) * 0.01)
                    break
            return super().complete(system, user, temperature)

    items = [InputItem(id=f"id{i}", prompt=f"prompt {i}", answer_a="A", answer_b="B") for i in range(n)]
    results = _run(ReverseDelayProvider(), items, max_workers=n * 2)
    assert [r.item.id for r in results] == [f"id{i}" for i in range(n)]


def test_cache_hit_skips_second_provider_call(tmp_path):
    calls = {"n": 0}

    def responder(system, user, temperature):
        calls["n"] += 1
        return json.dumps({"helpfulness": 0.5, "fluency": 0.5})

    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    cache = Cache(tmp_path)
    _run(FakeProvider(responder=responder), items, cache=cache)
    first = calls["n"]
    assert first == 2  # forward + reverse

    _run(FakeProvider(responder=responder), items, cache=cache)
    assert calls["n"] == first  # both passes served from cache, no new calls


def test_cache_hit_marks_cached_flag(tmp_path):
    items = [InputItem(id="x", prompt="q", answer_a="A", answer_b="B")]
    cache = Cache(tmp_path)
    _run(FakeProvider(), items, cache=cache)
    r = _run(FakeProvider(), items, cache=cache)[0]
    assert r.forward.cached and r.reverse.cached
