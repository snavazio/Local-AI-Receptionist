"""Unit tests for eval/analyze.py — classification + flakiness."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from analyze import (  # noqa: E402
    _classify_one,
    classify_failures,
    summarize_failure_modes,
    flakiness_score,
    top_flaky,
)


# ---------- classification ----------

def _row(**kw):
    base = {"id": "x", "category": "happy_path", "passed": False,
            "failures": [], "assistant_texts": []}
    base.update(kw)
    return base


class TestClassifyOne:
    def test_passed_row_skipped(self):
        # Wouldn't reach _classify_one in practice but should not crash
        assert _classify_one(_row(passed=True)) == "passed"

    def test_leaked_tokens(self):
        # leaked_template_tokens beats no_tool_call when both apply
        r = _row(failures=["got: no tool calls"],
                 assistant_texts=["<|im_start|>helper", "Got it"])
        assert _classify_one(r) == "leaked_template_tokens"

    def test_language_drift_chinese(self):
        r = _row(failures=["got: no tool calls"],
                 assistant_texts=["请告诉我您的电话号码"])
        assert _classify_one(r) == "language_drift"

    def test_forbidden_text(self):
        r = _row(failures=["Assistant said forbidden phrase: 'I have a slot at'"],
                 assistant_texts=["I have a slot at 2 PM"])
        assert _classify_one(r) == "forbidden_text"

    def test_false_positive_tool(self):
        r = _row(failures=["Tool 'book_appointment_callback' was called successfully but must not be"])
        assert _classify_one(r) == "false_positive_tool"

    def test_talks_without_acting(self):
        r = _row(failures=["Expected successful tool call to 'take_message', got: no tool calls"],
                 assistant_texts=["Got it, your message is saved."])
        assert _classify_one(r) == "talks_without_acting"

    def test_hallucinated_value(self):
        r = _row(failures=["Tool book_appointment_callback.callback_number should contain '4155550182'; actual values: ['2013882149']"])
        assert _classify_one(r) == "hallucinated_value"

    def test_no_tool_call_pure(self):
        r = _row(failures=["Expected successful tool call to 'book_appointment_callback', got: no tool calls"],
                 assistant_texts=["Sure, let me know what time"])
        assert _classify_one(r) == "no_tool_call"

    def test_missed_required_phrase(self):
        r = _row(failures=["Assistant must say one of ['five five five']; none found in transcript"])
        assert _classify_one(r) == "missed_required_phrase"

    def test_stacked_farewells(self):
        r = _row(failures=["Too many farewells (3 > 1 allowed)"])
        assert _classify_one(r) == "stacked_farewells"

    def test_uncategorized_fallback(self):
        r = _row(failures=["Some weird new failure shape"])
        assert _classify_one(r) == "uncategorized"


class TestSummarizeFailureModes:
    def test_groups_correctly(self):
        rows = [
            _row(id="a", failures=["got: no tool calls"], assistant_texts=["saved"]),
            _row(id="b", failures=["got: no tool calls"], assistant_texts=["saved"]),
            _row(id="c", failures=["should contain"]),
            _row(id="d", failures=["Too many farewells"]),
        ]
        ann = classify_failures(rows)
        modes = summarize_failure_modes(ann)
        assert modes["talks_without_acting"]["count"] == 2
        assert modes["hallucinated_value"]["count"] == 1
        assert modes["stacked_farewells"]["count"] == 1
        assert "a" in modes["talks_without_acting"]["examples"]


# ---------- flakiness ----------

class TestFlakiness:
    def test_stable_case_zero_flakiness(self):
        history = [{"by_case": {"x": True}} for _ in range(5)]
        scores = flakiness_score(history)
        assert scores["x"]["flakiness"] == 0.0
        assert scores["x"]["flips"] == 0

    def test_alternating_case_one_flakiness(self):
        history = [{"by_case": {"x": i % 2 == 0}} for i in range(5)]
        scores = flakiness_score(history)
        # 5 results: T F T F T -> 4 flips out of 4 transitions = 1.0
        assert scores["x"]["flakiness"] == 1.0
        assert scores["x"]["flips"] == 4

    def test_partial_flakiness(self):
        # T T T F F -> 1 flip out of 4 = 0.25
        history = [
            {"by_case": {"x": True}}, {"by_case": {"x": True}},
            {"by_case": {"x": True}}, {"by_case": {"x": False}},
            {"by_case": {"x": False}},
        ]
        scores = flakiness_score(history)
        assert scores["x"]["flakiness"] == 0.25
        assert scores["x"]["flips"] == 1

    def test_top_flaky_excludes_stable(self):
        history = [
            {"by_case": {"flaky": i % 2 == 0, "stable": True}} for i in range(5)
        ]
        scores = flakiness_score(history)
        flaky = top_flaky(scores, n=10)
        ids = [cid for cid, _ in flaky]
        assert "flaky" in ids
        assert "stable" not in ids

    def test_top_flaky_min_runs_filter(self):
        history = [{"by_case": {"x": False}}, {"by_case": {"x": True}}]
        scores = flakiness_score(history)
        # Only 2 runs — below default min_runs=3
        assert top_flaky(scores) == []

    def test_empty_history(self):
        assert flakiness_score([]) == {}
