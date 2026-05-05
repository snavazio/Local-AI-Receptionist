"""Regression watcher for the receptionist eval suite.

Runs the 100-case eval, compares the result against a saved baseline, and
emits a diff report:

  - which cases newly failed (regressions)
  - which cases newly passed (recoveries)
  - which cases flipped between this run and last (instability signal)
  - per-category pass-rate movement
  - LLM-call latency p50/p95 movement

Designed to run unattended on a schedule (cron / systemd timer / CI). When
something regresses, the script exits non-zero and prints a focused report
suitable for a Slack post or GitHub issue body.

Usage:
  python eval/watch.py                   # run, compare, update baseline
  python eval/watch.py --update-baseline # always overwrite, ignore comparison
  python eval/watch.py --no-update       # compare only, don't overwrite

State files (gitignored):
  eval/baseline.json   # last accepted run — {case_id: passed_bool, ...}
  eval/history.jsonl   # append-only log of every run for trend tracking
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RUN_EVAL = EVAL_DIR / "run_eval.py"
BASELINE = EVAL_DIR / "baseline.json"
HISTORY = EVAL_DIR / "history.jsonl"


def run_eval(concurrency: int = 2, model: str | None = None) -> dict:
    """Spawn run_eval.py with --json-out, parse and return the structured
    result. Returns {rows, llm_call_ms, wall_s}."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    try:
        cmd = [
            sys.executable,
            str(RUN_EVAL),
            "--concurrency", str(concurrency),
            "--json-out", str(json_path),
        ]
        if model:
            cmd += ["--model", model]
        # run_eval.py exits non-zero when any case fails — that's expected,
        # we still want the JSON output.
        env = os.environ.copy()
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if not json_path.exists() or json_path.stat().st_size == 0:
            raise RuntimeError(
                f"run_eval.py produced no JSON output at {json_path}"
            )
        return json.loads(json_path.read_text())
    finally:
        json_path.unlink(missing_ok=True)


def percentile(xs: list[int], pct: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(result: dict) -> dict:
    """Reduce a run JSON to the small set of fields we want to track."""
    rows = result.get("rows", [])
    by_case = {r["id"]: bool(r["passed"]) for r in rows}
    by_category: dict[str, dict[str, int]] = {}
    for r in rows:
        c = r.get("category", "")
        by_category.setdefault(c, {"pass": 0, "total": 0})
        by_category[c]["total"] += 1
        if r["passed"]:
            by_category[c]["pass"] += 1
    latencies = result.get("llm_call_ms", []) or []
    return {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": len(rows),
        "passed": sum(1 for r in rows if r["passed"]),
        "by_case": by_case,
        "by_category": by_category,
        "wall_s": result.get("wall_s"),
        "latency": {
            "n": len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else 0,
        },
    }


def diff_report(curr: dict, prev: dict | None) -> tuple[str, bool]:
    """Render a regression report comparing curr to prev. Returns
    (markdown_text, has_regression)."""
    lines: list[str] = []
    lines.append(f"# Regression watch — {curr['ts']}")
    lines.append("")
    lines.append(f"**{curr['passed']}/{curr['total']}** cases passed.")
    if prev:
        delta = curr["passed"] - prev["passed"]
        sign = "+" if delta >= 0 else ""
        lines.append(f"Previous: {prev['passed']}/{prev['total']} ({sign}{delta})")
    lines.append("")

    has_regression = False

    # Per-category movement
    if prev:
        lines.append("## Per-category movement")
        lines.append("")
        lines.append("| Category | Now | Then | Δ |")
        lines.append("| --- | --- | --- | --- |")
        cats = sorted(set(curr["by_category"]) | set(prev["by_category"]))
        for cat in cats:
            now = curr["by_category"].get(cat, {"pass": 0, "total": 0})
            then = prev["by_category"].get(cat, {"pass": 0, "total": 0})
            d = now["pass"] - then["pass"]
            sign = "+" if d >= 0 else ""
            marker = "🔴 " if d < 0 else ("🟢 " if d > 0 else "")
            lines.append(
                f"| {marker}{cat} | {now['pass']}/{now['total']} | "
                f"{then['pass']}/{then['total']} | {sign}{d} |"
            )
        lines.append("")

    # Per-case flips
    if prev:
        regressions = sorted(
            cid for cid, p in curr["by_case"].items()
            if not p and prev["by_case"].get(cid) is True
        )
        recoveries = sorted(
            cid for cid, p in curr["by_case"].items()
            if p and prev["by_case"].get(cid) is False
        )
        new_cases = sorted(set(curr["by_case"]) - set(prev["by_case"]))
        gone_cases = sorted(set(prev["by_case"]) - set(curr["by_case"]))

        if regressions:
            has_regression = True
            lines.append(f"## 🔴 Regressions ({len(regressions)})")
            for cid in regressions:
                lines.append(f"- `{cid}`")
            lines.append("")
        if recoveries:
            lines.append(f"## 🟢 Recoveries ({len(recoveries)})")
            for cid in recoveries:
                lines.append(f"- `{cid}`")
            lines.append("")
        if new_cases:
            lines.append(f"## ➕ New cases ({len(new_cases)})")
            for cid in new_cases:
                p = curr["by_case"][cid]
                lines.append(f"- `{cid}` — {'PASS' if p else 'FAIL'}")
            lines.append("")
        if gone_cases:
            lines.append(f"## ➖ Removed cases ({len(gone_cases)})")
            for cid in gone_cases:
                lines.append(f"- `{cid}`")
            lines.append("")

    # Latency
    lines.append("## LLM-call latency")
    lines.append("")
    lat = curr["latency"]
    if prev and prev.get("latency"):
        plat = prev["latency"]
        def fmt(now: int, then: int) -> str:
            d = now - then
            sign = "+" if d >= 0 else ""
            marker = "🔴 " if d > 200 else ("🟢 " if d < -200 else "")
            return f"{marker}{now} ms (was {then}, {sign}{d})"
        lines.append(f"- p50: {fmt(lat['p50'], plat['p50'])}")
        lines.append(f"- p95: {fmt(lat['p95'], plat['p95'])}")
        lines.append(f"- p99: {fmt(lat['p99'], plat['p99'])}")
        lines.append(f"- max: {fmt(lat['max'], plat['max'])}")
        # Big latency jumps count as regressions too
        if lat["p95"] - plat["p95"] > 1000:
            has_regression = True
    else:
        lines.append(f"- p50: {lat['p50']} ms")
        lines.append(f"- p95: {lat['p95']} ms")
        lines.append(f"- p99: {lat['p99']} ms")
        lines.append(f"- max: {lat['max']} ms")
    if curr.get("wall_s") is not None:
        lines.append(f"- wall: {curr['wall_s']:.1f}s")

    return "\n".join(lines), has_regression


def append_history(summary: dict) -> None:
    """Append a slim record for trend plots later. Drops the per-case map
    so this file stays small over time."""
    slim = {
        "ts": summary["ts"],
        "passed": summary["passed"],
        "total": summary["total"],
        "wall_s": summary["wall_s"],
        "latency": summary["latency"],
        "by_category": summary["by_category"],
    }
    with HISTORY.open("a") as f:
        f.write(json.dumps(slim) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Default 1 for deterministic results — concurrency >1 introduces "
             "false-alarm flips of ~10pp due to Ollama context-cache thrashing.",
    )
    parser.add_argument("--model", help="Override LLM model (sets EVAL_MODEL)")
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Skip comparison and overwrite baseline with this run.",
    )
    parser.add_argument(
        "--no-update", action="store_true",
        help="Compare only; don't overwrite the baseline.",
    )
    parser.add_argument(
        "--report", default=str(EVAL_DIR / "regression_report.md"),
        help="Path to write the regression markdown report.",
    )
    args = parser.parse_args()

    print(f"[watch] running eval at concurrency={args.concurrency}...", flush=True)
    result = run_eval(concurrency=args.concurrency, model=args.model)
    curr = summarize(result)

    prev: dict | None = None
    if BASELINE.exists() and not args.update_baseline:
        try:
            prev = json.loads(BASELINE.read_text())
        except (json.JSONDecodeError, OSError):
            prev = None

    report, has_regression = diff_report(curr, prev)
    Path(args.report).write_text(report)
    print(report)

    append_history(curr)

    # Slack/email-friendly one-liner — easy to grep for in cron output.
    delta = ""
    if prev is not None:
        d = curr["passed"] - prev["passed"]
        sign = "+" if d >= 0 else ""
        delta = f", {sign}{d}"
    print(
        f"\nSUMMARY: {curr['passed']}/{curr['total']} passed{delta} | "
        f"p50={curr['latency']['p50']}ms p95={curr['latency']['p95']}ms | "
        f"regressions={'yes' if has_regression else 'no'}",
        flush=True,
    )

    if args.update_baseline or (prev is None) or (not args.no_update and not has_regression):
        BASELINE.write_text(json.dumps(curr, indent=2))
        print(f"\n[watch] baseline updated -> {BASELINE}")
    elif args.no_update:
        print(f"\n[watch] --no-update set; baseline unchanged")
    else:
        print(f"\n[watch] regression detected; baseline NOT updated.")
        print(f"        Review {args.report} and re-run with --update-baseline if intentional.")

    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
