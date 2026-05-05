"""Run all cases in cases.yaml, evaluate assertions, write a report.

Usage:
    python eval/run_eval.py            # run everything, write eval/report.md
    python eval/run_eval.py --case ID  # run a single case, print to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from harness import CaseResult, run_case, _extract_phone_digits


FAREWELL_RE = re.compile(
    r"\b(take care|have a (great|good|nice|wonderful|lovely) day"
    r"|good ?bye|bye now|see you|we look forward|talk to you (soon|later))\b",
    re.IGNORECASE,
)


def _assistant_texts(result: CaseResult) -> list[str]:
    return [t.text for t in result.turns if t.role == "assistant" and t.text]


def _assistant_blob(result: CaseResult) -> str:
    return "\n".join(_assistant_texts(result)).lower()


def evaluate(case: dict, result: CaseResult) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failure_reasons)."""
    expect = case.get("expect", {}) or {}
    failures: list[str] = []

    expected_tool = expect.get("tool_called")
    called_names = [tc.name for tc in result.tool_calls]

    if expected_tool:
        successful_calls = [
            tc for tc in result.tool_calls
            if tc.name == expected_tool and tc.result.get("ok")
        ]
        if not successful_calls:
            failures.append(
                f"Expected successful tool call to {expected_tool!r}, "
                f"got: {called_names or 'no tool calls'}"
            )
        else:
            args_contain = expect.get("tool_args_contain") or {}
            for key, needle in args_contain.items():
                found = False
                for tc in successful_calls:
                    val = str(tc.args.get(key, "")).lower()
                    needle_str = str(needle).lower()
                    # For phone numbers, compare by normalized digits.
                    if key == "callback_number":
                        actual_digits = _extract_phone_digits(val) or ""
                        needle_digits = _extract_phone_digits(needle_str) or needle_str
                        if needle_digits in actual_digits:
                            found = True; break
                    elif needle_str in val:
                        found = True; break
                if not found:
                    failures.append(
                        f"Tool {expected_tool}.{key} should contain {needle!r}; "
                        f"actual values: {[tc.args.get(key) for tc in successful_calls]}"
                    )

    forbidden = expect.get("tool_must_not_be_called") or []
    for name in forbidden:
        if any(tc.name == name and tc.result.get("ok") for tc in result.tool_calls):
            failures.append(f"Tool {name!r} was called successfully but must not be")

    blob = _assistant_blob(result)

    must_say_any = expect.get("assistant_must_say_any") or []
    if must_say_any:
        if not any(s.lower() in blob for s in must_say_any):
            failures.append(
                f"Assistant must say one of {must_say_any!r}; none found in transcript"
            )

    must_not_say = expect.get("assistant_must_not_say") or []
    for s in must_not_say:
        if s.lower() in blob:
            failures.append(f"Assistant said forbidden phrase: {s!r}")

    max_farewells = expect.get("max_assistant_farewells", 1)
    farewell_count = sum(
        1 for t in _assistant_texts(result) if FAREWELL_RE.search(t)
    )
    if farewell_count > max_farewells:
        failures.append(
            f"Too many farewells ({farewell_count} > {max_farewells} allowed)"
        )

    return (len(failures) == 0, failures)


def render_report(rows: list[dict]) -> str:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    lines = ["# Eval report", ""]
    lines.append(f"**{passed}/{total}** cases passed.")
    lines.append("")
    lines.append("| Category | Pass | Total |")
    lines.append("| --- | --- | --- |")
    for cat, items in sorted(by_cat.items()):
        cp = sum(1 for r in items if r["passed"])
        lines.append(f"| {cat} | {cp} | {len(items)} |")
    lines.append("")

    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"## {status} — `{r['id']}`  ({r['category']})")
        lines.append(f"_{r['description']}_")
        lines.append("")
        if r["failures"]:
            lines.append("**Failures:**")
            for f in r["failures"]:
                lines.append(f"- {f}")
            lines.append("")
            lines.append("**Assistant transcript:**")
            for t in r["assistant_texts"]:
                lines.append(f"> {t}")
            if r["tool_calls"]:
                lines.append("")
                lines.append("**Tool calls:**")
                for tc in r["tool_calls"]:
                    ok = "ok" if tc["result"].get("ok") else "FAIL"
                    lines.append(f"- `{tc['name']}` [{ok}] args=`{tc['args']}`")
            lines.append("")
        else:
            lines.append("(passed)")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument(
        "--cases-file",
        default=str(Path(__file__).parent / "cases.yaml"),
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).parent / "report.md"),
    )
    args = parser.parse_args()

    cases = yaml.safe_load(Path(args.cases_file).read_text())
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}", file=sys.stderr)
            return 2

    rows: list[dict] = []
    for case in cases:
        print(f"running {case['id']}...", flush=True)
        try:
            result = run_case(case["id"], case["user_turns"])
        except Exception as e:
            rows.append({
                "id": case["id"],
                "category": case.get("category", ""),
                "description": case.get("description", ""),
                "passed": False,
                "failures": [f"harness exception: {e!r}"],
                "assistant_texts": [],
                "tool_calls": [],
            })
            continue

        passed, failures = evaluate(case, result)
        rows.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "description": case.get("description", ""),
            "passed": passed,
            "failures": failures,
            "assistant_texts": [t.text for t in result.turns if t.role == "assistant" and t.text],
            "tool_calls": [
                {"name": tc.name, "args": tc.args, "result": tc.result}
                for tc in result.tool_calls
            ],
        })

    report = render_report(rows)
    Path(args.report).write_text(report)
    print(report)
    print(f"\nReport written to {args.report}")
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
