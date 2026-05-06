"""Run all cases in cases.yaml, evaluate assertions, write a report.

Usage:
    python eval/run_eval.py            # run everything, write eval/report.md
    python eval/run_eval.py --case ID  # run a single case, print to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import asyncio
from harness import CaseResult, run_case, _extract_phone_digits, run_cases_concurrent


FAREWELL_RE = re.compile(
    r"\b(take care|have a (great|good|nice|wonderful|lovely) day"
    r"|good ?bye|bye now|see you|we look forward|talk to you (soon|later))\b",
    re.IGNORECASE,
)


def _assistant_texts(result: CaseResult) -> list[str]:
    return [t.text for t in result.turns if t.role == "assistant" and t.text]


def _assistant_blob(result: CaseResult) -> str:
    return "\n".join(_assistant_texts(result)).lower()


# Map legacy expectation names (from when there were two separate tools) onto
# the merged save_request tool: `book_appointment_callback` -> save_request with
# kind=appointment, `take_message` -> save_request with kind=message.
LEGACY_TOOL_KIND = {
    "book_appointment_callback": "appointment",
    "take_message": "message",
}


def _matches_expected(tc, expected_tool: str) -> bool:
    if tc.name == expected_tool:
        return True
    expected_kind = LEGACY_TOOL_KIND.get(expected_tool)
    if expected_kind and tc.name == "save_request":
        return (tc.args.get("kind") or "").lower() == expected_kind
    return False


def evaluate(case: dict, result: CaseResult) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_failure_reasons)."""
    expect = case.get("expect", {}) or {}
    failures: list[str] = []

    expected_tool = expect.get("tool_called")
    called_names = [tc.name for tc in result.tool_calls]

    if expected_tool:
        successful_calls = [
            tc for tc in result.tool_calls
            if _matches_expected(tc, expected_tool) and tc.result.get("ok")
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
        if any(_matches_expected(tc, name) and tc.result.get("ok") for tc in result.tool_calls):
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


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def render_report(rows: list[dict], latency_ms: list[int] | None = None,
                  concurrency: int = 1, wall_s: float | None = None) -> str:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    lines = ["# Eval report", ""]
    lines.append(f"**{passed}/{total}** cases passed.")
    lines.append("")
    if latency_ms:
        lines.append("## LLM-call latency")
        lines.append(f"- concurrency: **{concurrency}**")
        lines.append(f"- total LLM calls: **{len(latency_ms)}**")
        if wall_s is not None:
            lines.append(f"- wall time: **{wall_s:.1f}s**")
            lines.append(f"- aggregate throughput: **{len(latency_ms) / wall_s:.2f} calls/s**")
        lines.append(f"- p50: **{_percentile(latency_ms, 50)} ms**")
        lines.append(f"- p95: **{_percentile(latency_ms, 95)} ms**")
        lines.append(f"- p99: **{_percentile(latency_ms, 99)} ms**")
        lines.append(f"- max: **{max(latency_ms)} ms**")
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
        "--cases",
        help="Run multiple cases by id (comma-separated). Example: --cases happy_path_basic,emergency_bleeding,correction_mid_flow",
    )
    parser.add_argument("--category", help="Run only cases in this category")
    parser.add_argument(
        "--shard",
        help="Run a slice 'N/M' (1-indexed) of all cases, e.g. '2/5' for 2nd of 5 shards",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N cases concurrently (default 1). Spawns N subprocesses.",
    )
    parser.add_argument(
        "--json-out",
        help="Internal flag: write per-row JSON dump instead of markdown report.",
    )
    parser.add_argument(
        "--model",
        help="Override the LLM model name (sets EVAL_MODEL env var). Default qwen2.5:14b.",
    )
    parser.add_argument(
        "--cases-file",
        default=str(Path(__file__).parent / "cases.yaml"),
    )
    parser.add_argument(
        "--report",
        default=str(Path(__file__).parent / "report.md"),
    )
    args = parser.parse_args()

    # Propagate --model to harness via env var (subprocesses inherit).
    if args.model:
        import os as _os
        _os.environ["EVAL_MODEL"] = args.model

    cases = yaml.safe_load(Path(args.cases_file).read_text())
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}", file=sys.stderr)
            return 2
    if args.cases:
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"Warning: case ids not found: {sorted(missing)}", file=sys.stderr)
        if not cases:
            print(f"No matching cases for --cases {args.cases!r}", file=sys.stderr)
            return 2
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
        if not cases:
            print(f"No cases in category {args.category!r}", file=sys.stderr)
            return 2
    if args.shard:
        n_str, m_str = args.shard.split("/")
        n, m = int(n_str), int(m_str)
        # Round-robin so each shard gets a mix of categories.
        cases = [c for i, c in enumerate(cases) if i % m == (n - 1)]

    rows: list[dict] = []
    all_llm_call_ms: list[int] = []  # for the latency summary
    wall_s: float | None = None

    if args.concurrency > 1 and not args.shard:
        # Subprocess fan-out. Each child gets a shard and writes JSON. We merge.
        # (asyncio + AsyncOpenAI + Ollama hangs after a few requests; subprocesses
        # with separate clients work reliably — that's what we use.)
        import subprocess, tempfile, time as _time
        n = args.concurrency
        with tempfile.TemporaryDirectory() as td:
            json_paths = [Path(td) / f"shard_{i}.json" for i in range(1, n + 1)]
            extra_args = []
            if args.category:
                extra_args += ["--category", args.category]
            procs = []
            wall_t0 = _time.monotonic()
            for i in range(1, n + 1):
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--shard", f"{i}/{n}",
                    "--json-out", str(json_paths[i - 1]),
                    "--cases-file", args.cases_file,
                ] + extra_args
                procs.append(subprocess.Popen(cmd))
            for p in procs:
                p.wait()
            wall_s = _time.monotonic() - wall_t0
            print(f"all {n} shards finished in {wall_s:.1f}s", flush=True)
            for jp in json_paths:
                if jp.exists():
                    shard = json.loads(jp.read_text())
                    rows.extend(shard["rows"])
                    all_llm_call_ms.extend(shard.get("llm_call_ms", []))
    else:
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
            all_llm_call_ms.extend(result.llm_call_ms)
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
                "llm_call_ms": result.llm_call_ms,
                "prompt_tokens": getattr(result, "prompt_tokens", 0),
                "completion_tokens": getattr(result, "completion_tokens", 0),
            })

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "rows": rows,
            "llm_call_ms": all_llm_call_ms,
            "wall_s": wall_s,
            "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        }))
        return 0 if all(r["passed"] for r in rows) else 1

    report = render_report(
        rows,
        latency_ms=all_llm_call_ms or None,
        concurrency=args.concurrency,
        wall_s=wall_s,
    )
    Path(args.report).write_text(report)
    print(report)
    print(f"\nReport written to {args.report}")
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
