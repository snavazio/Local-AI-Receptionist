"""Analysis helpers for the qa report.

Two pure-Python utilities that the qa.py renderer calls into:

  flakiness_score(history_jsonl) -> dict[case_id, dict]
    For each case, computes how often it has flipped pass/fail across the
    most recent N runs in history.jsonl. Cases that flip often are noise;
    cases that pass-then-fail consistently are real regressions.

  classify_failures(rows) -> list[dict]
    For each failing row in an eval result, classifies the failure mode
    using rule-based pattern matching against the failure messages and
    assistant transcripts. Returns annotated rows so qa.py can summarize
    "10 talks-without-acting failures, 5 hallucinated-value, ..." instead
    of just "31 failures."

Both are pure Python with no external deps. Unit tests cover the rule logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================================
# Flakiness
# ============================================================================

def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def flakiness_score(history: list[dict], last_n: int = 10) -> dict[str, dict]:
    """For each case_id seen in the last `last_n` runs, compute:
      - runs_seen: how many of the last N runs include this case
      - pass_count / fail_count
      - flips: number of pass↔fail transitions in chronological order
      - flakiness: flips / max(runs_seen-1, 1) — 0 means stable, 1 means
        flips on every run
    Returns dict keyed by case_id.
    """
    if not history:
        return {}
    runs = history[-last_n:]
    by_case: dict[str, list[bool]] = {}
    for run in runs:
        bcase = run.get("by_case", {}) or {}
        for cid, passed in bcase.items():
            by_case.setdefault(cid, []).append(bool(passed))

    out: dict[str, dict] = {}
    for cid, results in by_case.items():
        if len(results) < 2:
            flakiness = 0.0
            flips = 0
        else:
            flips = sum(1 for a, b in zip(results, results[1:]) if a != b)
            flakiness = flips / (len(results) - 1)
        out[cid] = {
            "runs_seen": len(results),
            "pass_count": sum(results),
            "fail_count": len(results) - sum(results),
            "flips": flips,
            "flakiness": round(flakiness, 3),
            "current": results[-1],
        }
    return out


def top_flaky(scores: dict[str, dict], n: int = 10, min_runs: int = 3) -> list[tuple[str, dict]]:
    """Return the top-N flakiest cases (highest flakiness, breaking ties by flips desc)."""
    items = [
        (cid, info) for cid, info in scores.items()
        if info["runs_seen"] >= min_runs and info["flakiness"] > 0
    ]
    items.sort(key=lambda x: (-x[1]["flakiness"], -x[1]["flips"]))
    return items[:n]


# ============================================================================
# Failure-mode classification
# ============================================================================
# Each rule is a tuple of (mode_name, predicate). Predicate takes the row dict
# and returns True if the row matches that mode. Rules are checked in order;
# first match wins. Falls back to "uncategorized" if no rule matches.

# Common failure-message patterns from run_eval.py's evaluate():
#   "Expected successful tool call to 'X', got: ..."
#   "Tool X.field should contain '...'; actual values: [...]"
#   "Tool 'X' was called successfully but must not be"
#   "Assistant must say one of [...]; none found"
#   "Assistant said forbidden phrase: ..."
#   "Too many farewells (N > M allowed)"

def _has_failure(row: dict, *needles: str) -> bool:
    fails = " | ".join(row.get("failures", []) or [])
    return any(n.lower() in fails.lower() for n in needles)


def _has_text(row: dict, *needles: str) -> bool:
    text = " | ".join(row.get("assistant_texts", []) or [])
    return any(n.lower() in text.lower() for n in needles)


def _no_tool_called(row: dict) -> bool:
    return _has_failure(row, "got: no tool calls", "got: ['save_request']")  # last covers wrong-kind


def _classify_one(row: dict) -> str:
    """Return a mode label for this row. Order matters."""
    if not row.get("failures"):
        return "passed"  # shouldn't reach classifier but safe

    # Deeper causes first — these often produce a downstream "no tool called"
    # too, but we'd rather surface the underlying issue than the surface
    # symptom.

    # Output-formatting issues (leaked chat-template tokens). Loud signal.
    if _has_text(row, "<|im_start|>", "<|endoftext|>", "<tool_call>", "TokenName:", "_icall_"):
        return "leaked_template_tokens"

    # Language drift — assistant said something obviously non-English
    if _has_text(row, "请", "您", "你", "我", "是", "的", "了"):  # CJK common chars
        return "language_drift"

    # Forbidden text in assistant output (assertion failure on must_not_say)
    if _has_failure(row, "said forbidden phrase"):
        return "forbidden_text"

    # Tool was called when it shouldn't have been (false-positive tool fire)
    if _has_failure(row, "was called successfully but must not be"):
        return "false_positive_tool"

    # Talks-without-acting: tool was expected, none called, but the bot
    # produced reassuring text like "saved" / "got it" / "confirmed"
    if _no_tool_called(row) and _has_text(row, "saved", "got it", "confirmed", "recorded", "noted"):
        return "talks_without_acting"

    # Tool called but with wrong values — hallucinated argument
    if _has_failure(row, "should contain"):
        return "hallucinated_value"

    # No tool called at all (and bot didn't say it saved anything)
    if _has_failure(row, "got: no tool calls"):
        return "no_tool_call"

    # Assistant didn't satisfy a "must_say_any" assertion
    if _has_failure(row, "must say one of"):
        return "missed_required_phrase"

    # Multiple farewells stacked
    if _has_failure(row, "Too many farewells"):
        return "stacked_farewells"

    return "uncategorized"


def classify_failures(rows: list[dict]) -> list[dict]:
    """Annotate each FAILING row with a `failure_mode` label. Returns a new
    list of dicts (the original rows passed through, with the extra field)."""
    annotated = []
    for row in rows:
        if row.get("passed"):
            continue
        mode = _classify_one(row)
        annotated.append({**row, "failure_mode": mode})
    return annotated


def summarize_failure_modes(annotated_failures: list[dict]) -> dict[str, dict]:
    """Build a summary table: mode -> {count, examples (top 3 case ids), categories}."""
    out: dict[str, dict] = {}
    for r in annotated_failures:
        mode = r.get("failure_mode", "uncategorized")
        bucket = out.setdefault(mode, {"count": 0, "examples": [], "categories": set()})
        bucket["count"] += 1
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(r.get("id"))
        bucket["categories"].add(r.get("category", "?"))
    # convert sets to sorted lists for json/markdown rendering
    for mode in out:
        out[mode]["categories"] = sorted(out[mode]["categories"])
    return out
