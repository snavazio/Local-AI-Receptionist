"""Unit tests for the eval harness helpers and the watcher diff logic.

These are pure-Python tests with no Ollama / Pipecat dependency, so they
run in a couple seconds and can be wired into a pre-commit hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval.harness import (  # noqa: E402
    _extract_phone_digits,
    _FAREWELL_RE,
    _missing,
    _strip_malformed_tool_call,
    execute_tool,
)
from eval.run_eval import (  # noqa: E402
    LEGACY_TOOL_KIND,
    _matches_expected,
    _percentile,
    evaluate,
)
from eval.watch import diff_report, summarize  # noqa: E402

# Synthetic tool-call records — match the dataclass shape lightly via
# duck-typing. The evaluator only reads .name, .args, .result.
class FakeToolCall:
    def __init__(self, name: str, args: dict, result: dict):
        self.name = name
        self.args = args
        self.result = result


class FakeTurn:
    def __init__(self, role: str, text: str = ""):
        self.role = role
        self.text = text


class FakeCaseResult:
    def __init__(self, turns, tool_calls):
        self.turns = turns
        self.tool_calls = tool_calls


# ---------- harness helpers ----------

class TestStripMalformedToolCall:
    def test_indEx_pattern_removed(self):
        s = ' iNdEx_icall_{"name":"escalate_emergency","arguments":{"reason":"pain"}} iNdEx_icism!'
        assert _strip_malformed_tool_call(s) == ""

    def test_xml_tool_call_removed(self):
        s = '<tool_call>{"name": "foo", "arguments": {}}</tool_call>'
        assert _strip_malformed_tool_call(s) == ""

    def test_unterminated_tool_call_removed(self):
        s = '<tool_call>{"name": "foo"\nstill text after'
        assert _strip_malformed_tool_call(s) == ""

    def test_legitimate_text_preserved(self):
        s = "Got it. Your callback is saved."
        assert _strip_malformed_tool_call(s) == s

    def test_mixed_keeps_clean_part(self):
        s = "Hello there. <tool_call>{}</tool_call> rest of message"
        cleaned = _strip_malformed_tool_call(s)
        assert "Hello there" in cleaned
        assert "rest of message" in cleaned
        assert "tool_call" not in cleaned


class TestFarewellRegex:
    def test_take_care_matched(self):
        assert _FAREWELL_RE.search("Take care!")
        assert _FAREWELL_RE.search("Have a great day")
        assert _FAREWELL_RE.search("Goodbye")
        assert _FAREWELL_RE.search("good bye")

    def test_thanks_for_calling_NOT_matched(self):
        # Greeting phrase, not farewell — must not trigger the deduper
        assert not _FAREWELL_RE.search("Thanks for calling Smith Family Dental")

    def test_clean_text_NOT_matched(self):
        assert not _FAREWELL_RE.search("What time would work for you?")
        assert not _FAREWELL_RE.search("Got it, your callback is saved.")


# ---------- harness execute_tool gating ----------

class TestExecuteTool:
    def test_save_appointment_happy(self):
        r = execute_tool("save_request", {
            "kind": "appointment",
            "caller_name": "Steve",
            "callback_number": "201-388-2149",
            "preferred_window": "Tuesday 2pm",
        })
        assert r["ok"] and r["kind"] == "appointment"

    def test_save_appointment_missing_window(self):
        r = execute_tool("save_request", {
            "kind": "appointment",
            "caller_name": "Steve",
            "callback_number": "201-388-2149",
        })
        assert not r["ok"]
        assert "preferred_window" in r["error"]

    def test_save_message_happy(self):
        r = execute_tool("save_request", {
            "kind": "message",
            "caller_name": "Steve",
            "callback_number": "201-388-2149",
            "message": "Crown is loose",
        })
        assert r["ok"] and r["kind"] == "message"

    def test_save_request_word_form_phone(self):
        # Critical: word-form phone must work end-to-end.
        r = execute_tool("save_request", {
            "kind": "appointment",
            "caller_name": "Steve",
            "callback_number": "two zero one three eight eight two one four nine",
            "preferred_window": "Wednesday 1pm",
        })
        assert r["ok"]

    def test_save_request_bad_kind(self):
        r = execute_tool("save_request", {"kind": "bogus"})
        assert not r["ok"]
        assert r["error"] == "Bad kind"

    def test_emergency(self):
        r = execute_tool("escalate_emergency", {"reason": "pain"})
        assert r["ok"]
        assert "five five five" in r["spoken_response"]

    def test_unknown_tool(self):
        r = execute_tool("nope", {})
        assert not r["ok"]


# ---------- run_eval evaluation logic ----------

class TestLegacyToolMapping:
    def test_legacy_appointment_maps_to_save_request(self):
        tc = FakeToolCall(
            "save_request",
            {"kind": "appointment", "caller_name": "Steve"},
            {"ok": True},
        )
        assert _matches_expected(tc, "book_appointment_callback")
        assert _matches_expected(tc, "save_request")
        assert not _matches_expected(tc, "take_message")

    def test_legacy_message_maps_to_save_request(self):
        tc = FakeToolCall(
            "save_request",
            {"kind": "message"},
            {"ok": True},
        )
        assert _matches_expected(tc, "take_message")
        assert not _matches_expected(tc, "book_appointment_callback")


class TestEvaluate:
    def _make_result(self, tool_calls=None, assistant_texts=None):
        turns = [FakeTurn("assistant", t) for t in (assistant_texts or [])]
        return FakeCaseResult(turns=turns, tool_calls=tool_calls or [])

    def test_passes_when_tool_called(self):
        case = {"expect": {"tool_called": "book_appointment_callback"}}
        result = self._make_result(tool_calls=[
            FakeToolCall("save_request", {"kind": "appointment"}, {"ok": True})
        ])
        passed, fails = evaluate(case, result)
        assert passed, fails

    def test_fails_when_tool_missing(self):
        case = {"expect": {"tool_called": "book_appointment_callback"}}
        result = self._make_result(tool_calls=[])
        passed, fails = evaluate(case, result)
        assert not passed
        assert any("Expected" in f for f in fails)

    def test_forbidden_tool_caught(self):
        case = {"expect": {"tool_must_not_be_called": ["book_appointment_callback"]}}
        result = self._make_result(tool_calls=[
            FakeToolCall("save_request", {"kind": "appointment"}, {"ok": True})
        ])
        passed, fails = evaluate(case, result)
        assert not passed

    def test_must_say_any_satisfied(self):
        case = {"expect": {"assistant_must_say_any": ["five five five", "emergency"]}}
        result = self._make_result(assistant_texts=["For dental emergencies call five five five"])
        passed, fails = evaluate(case, result)
        assert passed

    def test_must_not_say_caught(self):
        case = {"expect": {"assistant_must_not_say": ["I have a slot at"]}}
        result = self._make_result(assistant_texts=["I have a slot at 2 PM available."])
        passed, fails = evaluate(case, result)
        assert not passed

    def test_phone_args_normalized_for_match(self):
        case = {"expect": {
            "tool_called": "book_appointment_callback",
            "tool_args_contain": {"callback_number": "2013882149"},
        }}
        # Tool called with word-form phone — assertion should still match
        result = self._make_result(tool_calls=[
            FakeToolCall("save_request", {
                "kind": "appointment",
                "callback_number": "two zero one three eight eight two one four nine",
            }, {"ok": True})
        ])
        passed, _ = evaluate(case, result)
        assert passed

    def test_too_many_farewells(self):
        case = {"expect": {"max_assistant_farewells": 1}}
        result = self._make_result(assistant_texts=[
            "Take care!", "Goodbye!", "Have a great day!"
        ])
        passed, fails = evaluate(case, result)
        assert not passed
        assert any("farewell" in f.lower() for f in fails)


class TestPercentile:
    def test_50th(self):
        assert _percentile([10, 20, 30, 40, 50], 50) == 30

    def test_95th(self):
        assert _percentile(list(range(1, 101)), 95) == 95

    def test_empty(self):
        assert _percentile([], 50) == 0


# ---------- watch.py diff logic ----------

class TestSummarize:
    def test_basic_shape(self):
        result = {
            "rows": [
                {"id": "a", "category": "happy_path", "passed": True},
                {"id": "b", "category": "happy_path", "passed": False},
                {"id": "c", "category": "message", "passed": True},
            ],
            "llm_call_ms": [100, 200, 300, 400, 500],
            "wall_s": 12.5,
        }
        s = summarize(result)
        assert s["total"] == 3
        assert s["passed"] == 2
        assert s["by_case"] == {"a": True, "b": False, "c": True}
        assert s["by_category"]["happy_path"] == {"pass": 1, "total": 2}
        assert s["by_category"]["message"] == {"pass": 1, "total": 1}
        assert s["latency"]["p50"] == 300
        assert s["wall_s"] == 12.5


class TestDiffReport:
    def _summary(self, passed, by_case, by_cat, latency=None):
        return {
            "ts": "2026-05-05T10:00:00",
            "total": sum(c["total"] for c in by_cat.values()),
            "passed": passed,
            "by_case": by_case,
            "by_category": by_cat,
            "wall_s": 100.0,
            "latency": latency or {"n": 100, "p50": 1000, "p95": 3000, "p99": 4000, "max": 5000},
        }

    def test_no_baseline_no_regression_marker(self):
        curr = self._summary(80, {"a": True}, {"happy_path": {"pass": 1, "total": 1}})
        report, has_reg = diff_report(curr, prev=None)
        assert not has_reg
        assert "80/" in report

    def test_regression_detected(self):
        prev = self._summary(80, {"a": True, "b": True}, {"happy_path": {"pass": 2, "total": 2}})
        curr = self._summary(79, {"a": True, "b": False}, {"happy_path": {"pass": 1, "total": 2}})
        report, has_reg = diff_report(curr, prev)
        assert has_reg
        assert "Regressions (1)" in report
        assert "`b`" in report

    def test_recovery_no_regression(self):
        prev = self._summary(78, {"a": False, "b": True}, {"happy_path": {"pass": 1, "total": 2}})
        curr = self._summary(80, {"a": True, "b": True}, {"happy_path": {"pass": 2, "total": 2}})
        report, has_reg = diff_report(curr, prev)
        assert not has_reg
        assert "Recoveries (1)" in report

    def test_p95_latency_regression(self):
        prev = self._summary(80, {}, {}, latency={"p50": 1000, "p95": 3000, "p99": 4000, "max": 5000, "n": 100})
        curr = self._summary(80, {}, {}, latency={"p50": 1100, "p95": 4500, "p99": 5500, "max": 6000, "n": 100})
        # +1500 ms p95 should trigger regression even though pass count is unchanged
        report, has_reg = diff_report(curr, prev)
        assert has_reg
