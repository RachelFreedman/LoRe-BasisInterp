"""Defensive-parsing tests. All offline, no API calls."""

from __future__ import annotations

import json

import pytest

from judge.parsing import Status, parse_scores

KEYS = ("helpfulness", "fluency")


def test_clean_object():
    r = parse_scores('{"helpfulness": 0.25, "fluency": 0.75}', KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.25, "fluency": 0.75}
    assert r.detail == ""


def test_markdown_fenced_object_recovered():
    raw = '```json\n{"helpfulness": 0.5, "fluency": 0.5}\n```'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 0.5, "fluency": 0.5}


def test_leading_prose_then_object_recovered():
    raw = 'Here are my scores:\n{"helpfulness": 1, "fluency": 0}'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert r.scores == {"helpfulness": 1.0, "fluency": 0.0}


def test_integer_values_coerced_to_float():
    r = parse_scores('{"helpfulness": 1, "fluency": 0}', KEYS)
    assert r.status is Status.OK
    assert isinstance(r.scores["helpfulness"], float)


def test_brace_in_string_value_does_not_break_extraction():
    raw = 'note {weird}\n{"helpfulness": 0.5, "fluency": 0.5, "junk": "a } b"}'
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert "ignored extra keys: junk" in r.detail


def test_missing_key_is_validation_error():
    r = parse_scores('{"helpfulness": 0.5}', KEYS)
    assert r.status is Status.VALIDATION_ERROR
    assert "fluency" in r.detail


def test_out_of_range_high_is_validation_error_not_clamped():
    r = parse_scores('{"helpfulness": 1.5, "fluency": 0.5}', KEYS)
    assert r.status is Status.VALIDATION_ERROR
    assert r.scores is None


def test_out_of_range_negative_is_validation_error():
    r = parse_scores('{"helpfulness": -0.1, "fluency": 0.5}', KEYS)
    assert r.status is Status.VALIDATION_ERROR


def test_non_numeric_value_is_validation_error():
    r = parse_scores('{"helpfulness": "high", "fluency": 0.5}', KEYS)
    assert r.status is Status.VALIDATION_ERROR


def test_boolean_value_rejected():
    r = parse_scores('{"helpfulness": true, "fluency": 0.5}', KEYS)
    assert r.status is Status.VALIDATION_ERROR


def test_no_json_at_all_is_parse_error():
    r = parse_scores("I scored them in my head, trust me.", KEYS)
    assert r.status is Status.PARSE_ERROR
    assert r.scores is None


def test_empty_string_is_parse_error():
    r = parse_scores("", KEYS)
    assert r.status is Status.PARSE_ERROR


def test_refusal_detected():
    r = parse_scores("I'm unable to compare these responses.", KEYS)
    assert r.status is Status.REFUSAL


def test_refusal_marker_ignored_when_valid_json_present():
    raw = "I won't editorialize. {\"helpfulness\": 0.5, \"fluency\": 0.5}"
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK


def test_extra_keys_ignored_with_note():
    raw = json.dumps({"helpfulness": 0.5, "fluency": 0.5, "safety": 0.9})
    r = parse_scores(raw, KEYS)
    assert r.status is Status.OK
    assert "safety" in r.detail
