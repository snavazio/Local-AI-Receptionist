"""Autonomous QA runner for the receptionist project.

Runs the full QA pipeline without anyone watching, then writes ONE
comprehensive report you can share with Claude (or a teammate, or read
yourself). The script is the runner; the AI is the analyst.

What it does, in order:
  1. `pytest tests/` — unit tests (~2 seconds)
  2. `eval/run_eval.py` — full 100+ case eval against the live LLM
  3. Compare to last accepted baseline (`eval/baseline.json`)
  4. Append a slim record to `eval/history.jsonl` for trend tracking
  5. Generate `qa_runs/qa_<timestamp>.md` — combined report:
       - exec summary (pass rates, deltas, regressions, latency)
       - unit-test results (pass/fail counts; any failures verbatim)
       - eval result table by category
       - regression diff vs baseline
       - trend sparkline (last 10 runs)
       - **transcripts of every failing case** (so an analyst can diagnose
         without re-running anything)
  6. Update baseline.json IF no regression detected (configurable)

Usage:
  python qa.py              # standard run; updates baseline if no regression
  python qa.py --no-update  # don't touch baseline (e.g., for ad-hoc checks)
  python qa.py --model qwen2.5:7b   # try a different model
  python qa.py --concurrency 2      # parallelize (default 1 for clean latency)

When complete, the script prints:
  - the path to the report
  - a one-line summary (pass rate, regression flag)
  - a suggested commit message if you want to log this run

Designed to be wired into cron, a systemd timer, or run manually after
each prompt change. Claude / a teammate reads the report later and
proposes fixes; nobody has to babysit the eval.
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

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
TESTS_DIR = ROOT / "tests"
RUNS_DIR = ROOT / "qa_runs"
BASELINE = EVAL_DIR / "baseline.json"
HISTORY = EVAL_DIR / "history.jsonl"

# Always use the project's venv python for subprocesses, even if the user
# launched qa.py with a different Python (e.g. conda's `python`). The venv
# is where pytest, pyyaml, openai, etc. live; sys.executable can point at
# an env that doesn't have these packages.
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable


# ============================================================================
# Helpers
# ============================================================================

def percentile(xs: list[int], pct: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    chars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return chars[4] * len(values)
    span = hi - lo
    return "".join(
        chars[max(0, min(len(chars) - 1, int(round((v - lo) / span * (len(chars) - 1)))))]
        for v in values
    )


def run_unit_tests() -> dict:
    """Returns {passed: int, failed: int, total: int, output: str}."""
    print(f"[qa] running unit tests (python: {PY})...", flush=True)
    proc = subprocess.run(
        [PY, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    # Parse the pytest summary line e.g. "121 passed in 1.62s"
    import re
    passed = 0
    failed = 0
    parsed = False
    for line in out.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m:
            passed = int(m.group(1))
            parsed = True
        m = re.search(r"(\d+) failed", line)
        if m:
            failed = int(m.group(1))
            parsed = True
        if parsed and ("in " in line and "s" in line):
            break
    if not parsed:
        # pytest didn't emit a summary line — likely import error / missing pytest
        print(f"[qa] WARNING: pytest produced no summary line. Exit code {proc.returncode}.", flush=True)
        if proc.returncode != 0:
            print(f"[qa] First 500 chars of pytest output:\n{out[:500]}", flush=True)
    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "exit_code": proc.returncode,
        "output": out,
    }


def run_full_eval(concurrency: int, model: str | None, progress_log: Path) -> dict:
    """Spawn run_eval.py with --json-out, stream its stdout line-by-line to
    `progress_log` (so the user can `tail -f` the file to watch progress in
    real time), then parse the JSON output it produced. Returns
    {rows, llm_call_ms, wall_s}.

    Progress log format:
        [HH:MM:SS] [3/100] running happy_path_basic...
    Each "running <case>..." line increments the in-flight counter; the log
    survives qa.py crashes so you can see how far the run got.
    """
    print(f"[qa] running full eval (concurrency={concurrency})...", flush=True)
    print(f"[qa] live progress: tail -f {progress_log}", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    try:
        cmd = [
            PY,
            str(EVAL_DIR / "run_eval.py"),
            "--concurrency", str(concurrency),
            "--json-out", str(json_path),
        ]
        if model:
            cmd += ["--model", model]
        env = os.environ.copy()

        # Stream stdout/stderr line-by-line into the progress log.
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        case_count = 0
        with open(progress_log, "w") as plog:
            plog.write(f"# qa.py progress log\n# started: {datetime.datetime.now().isoformat(timespec='seconds')}\n# cmd: {' '.join(cmd)}\n\n")
            plog.flush()
            for raw_line in proc.stdout:  # type: ignore[union-attr]
                line = raw_line.rstrip("\n")
                if line.startswith("running "):
                    case_count += 1
                    prefix = f"[{datetime.datetime.now():%H:%M:%S}] [{case_count:>3}] "
                else:
                    prefix = f"[{datetime.datetime.now():%H:%M:%S}]      "
                plog.write(prefix + line + "\n")
                plog.flush()
        proc.wait()
        plog_msg = f"# finished: {datetime.datetime.now().isoformat(timespec='seconds')}  exit={proc.returncode}  cases_seen={case_count}\n"
        with open(progress_log, "a") as plog:
            plog.write(plog_msg)
        print(f"[qa] eval subprocess exited (code {proc.returncode}, {case_count} cases observed)", flush=True)

        if not json_path.exists() or json_path.stat().st_size == 0:
            raise RuntimeError(f"run_eval.py produced no JSON output at {json_path}")
        return json.loads(json_path.read_text())
    finally:
        json_path.unlink(missing_ok=True)


def summarize_eval(result: dict) -> dict:
    """Reduce a run JSON to summary fields + per-category counts."""
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


def load_baseline() -> dict | None:
    if BASELINE.exists():
        try:
            return json.loads(BASELINE.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def diff_categories(curr: dict, prev: dict | None) -> tuple[list[str], list[str]]:
    """Returns (regressed_cases, recovered_cases)."""
    if not prev:
        return [], []
    regressed = sorted(
        cid for cid, p in curr["by_case"].items()
        if not p and prev["by_case"].get(cid) is True
    )
    recovered = sorted(
        cid for cid, p in curr["by_case"].items()
        if p and prev["by_case"].get(cid) is False
    )
    return regressed, recovered


def append_history(summary: dict) -> None:
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


def parse_case_durations(progress_log: Path) -> list[dict]:
    """Read the progress log, infer per-case duration from the timestamps
    on consecutive 'running CASE_ID...' lines. Returns a list of
    {id, started_at, duration_s} dicts. Last case's duration is computed
    against the '# finished:' footer if present, else None."""
    if not progress_log.exists():
        return []
    import re
    line_re = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s+\[\s*\d+\]\s+running ([\w_]+)\.\.\.")
    finished_re = re.compile(r"^# finished:\s*(\S+)")
    entries: list[tuple[str, str]] = []  # (timestamp_str, case_id)
    finished_at: str | None = None
    for line in progress_log.read_text().splitlines():
        m = line_re.match(line)
        if m:
            entries.append((m.group(1), m.group(2)))
            continue
        m = finished_re.match(line)
        if m:
            finished_at = m.group(1).split("T")[-1] if "T" in m.group(1) else None

    def to_secs(hms: str) -> int:
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    durations: list[dict] = []
    for i, (ts, cid) in enumerate(entries):
        if i + 1 < len(entries):
            next_ts = entries[i + 1][0]
            dur = max(0, to_secs(next_ts) - to_secs(ts))
        elif finished_at:
            dur = max(0, to_secs(finished_at[:8]) - to_secs(ts))
        else:
            dur = None
        durations.append({"id": cid, "started_at": ts, "duration_s": dur})
    return durations


def trend_lines(last_n: int = 10) -> list[str]:
    """Read history.jsonl and return ASCII sparklines for trend."""
    if not HISTORY.exists():
        return ["(no history yet)"]
    rows = []
    for line in HISTORY.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    rows = rows[-last_n:]
    if not rows:
        return ["(history empty)"]
    pass_rates = [r["passed"] / r["total"] * 100 for r in rows]
    p50s = [r["latency"]["p50"] for r in rows]
    p95s = [r["latency"]["p95"] for r in rows]
    out = [
        f"Pass rate: {sparkline(pass_rates)}  "
        f"range [{min(pass_rates):.0f}%, {max(pass_rates):.0f}%]  current: {pass_rates[-1]:.0f}%",
        f"LLM p50:   {sparkline(p50s)}  "
        f"range [{min(p50s)} ms, {max(p50s)} ms]  current: {p50s[-1]} ms",
        f"LLM p95:   {sparkline(p95s)}  "
        f"range [{min(p95s)} ms, {max(p95s)} ms]  current: {p95s[-1]} ms",
    ]
    return out


def git_state() -> dict:
    def cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git"] + args, cwd=ROOT, text=True).strip()
        except Exception:
            return "?"
    return {
        "branch": cmd(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": cmd(["rev-parse", "--short", "HEAD"]),
        "subject": cmd(["log", "-1", "--pretty=%s"]),
        "dirty": bool(cmd(["status", "--porcelain"])),
    }


# ============================================================================
# Report generation
# ============================================================================

def render_report(
    *,
    started_at: str,
    finished_at: str,
    git: dict,
    unit: dict,
    eval_result: dict,
    eval_summary: dict,
    baseline: dict | None,
    regressed: list[str],
    recovered: list[str],
    history: list[str],
    has_regression: bool,
    durations: list[dict] | None = None,
) -> str:
    L: list[str] = []
    L.append(f"# QA report — {finished_at}")
    L.append("")
    L.append("## Exec summary")
    L.append("")

    # Top-level pass rates
    pass_rate = eval_summary["passed"] / max(eval_summary["total"], 1) * 100
    L.append(f"- **Eval**: **{eval_summary['passed']}/{eval_summary['total']}** passed ({pass_rate:.0f}%)")
    if baseline:
        delta = eval_summary["passed"] - baseline["passed"]
        sign = "+" if delta >= 0 else ""
        L.append(f"  - vs baseline ({baseline.get('ts','?')}): **{baseline['passed']}/{baseline['total']}** ({sign}{delta})")
        L.append(f"  - regressions: **{len(regressed)}** | recoveries: **{len(recovered)}**")
    L.append(f"- **Unit tests**: {unit['passed']} passed, {unit['failed']} failed (exit {unit['exit_code']})")
    lat = eval_summary["latency"]
    L.append(f"- **Latency**: p50 {lat['p50']} ms · p95 {lat['p95']} ms · p99 {lat['p99']} ms · max {lat['max']} ms")
    if eval_summary.get("wall_s"):
        L.append(f"- **Wall**: {eval_summary['wall_s']:.0f}s")
    L.append(f"- **Git**: `{git['branch']}` @ `{git['commit']}` — _{git['subject']}_" +
             (" *(dirty)*" if git["dirty"] else ""))
    L.append(f"- **Started**: {started_at} · **Finished**: {finished_at}")
    if has_regression:
        L.append("")
        L.append("> 🔴 **REGRESSION DETECTED** — baseline NOT updated. Investigate the failing cases below.")
    L.append("")

    # Category breakdown
    L.append("## Eval per-category")
    L.append("")
    L.append("| Category | Pass | Total | vs baseline |")
    L.append("| --- | --- | --- | --- |")
    cats = sorted(eval_summary["by_category"].keys() |
                  (set(baseline["by_category"].keys()) if baseline else set()))
    for cat in cats:
        now = eval_summary["by_category"].get(cat, {"pass": 0, "total": 0})
        if baseline:
            prev = baseline["by_category"].get(cat, {"pass": 0, "total": 0})
            d = now["pass"] - prev["pass"]
            sign = "+" if d >= 0 else ""
            marker = "🔴 " if d < 0 else ("🟢 " if d > 0 else "")
            delta_col = f"{marker}{prev['pass']}/{prev['total']} ({sign}{d})"
        else:
            delta_col = "—"
        L.append(f"| {cat} | {now['pass']} | {now['total']} | {delta_col} |")
    L.append("")

    # Trend
    L.append("## Trend (last 10 runs)")
    L.append("")
    L.append("```")
    for line in history:
        L.append(line)
    L.append("```")
    L.append("")

    # Regressions / recoveries
    if baseline and regressed:
        L.append(f"## 🔴 Regressed cases ({len(regressed)})")
        L.append("")
        for cid in regressed:
            L.append(f"- `{cid}`")
        L.append("")
    if baseline and recovered:
        L.append(f"## 🟢 Recovered cases ({len(recovered)})")
        L.append("")
        for cid in recovered:
            L.append(f"- `{cid}`")
        L.append("")

    # Unit-test details (failures only — keep report compact)
    if unit["failed"] > 0:
        L.append("## Unit test failures")
        L.append("")
        L.append("```")
        L.append(unit["output"])
        L.append("```")
        L.append("")

    # Per-case duration (helps spot speed-up opportunities)
    if durations:
        # Build case_id -> category map from this run
        cid_to_cat = {r["id"]: r.get("category", "?") for r in eval_result.get("rows", [])}
        # Filter to entries with known durations
        d_with = [d for d in durations if d.get("duration_s") is not None]
        if d_with:
            L.append("## Per-case duration")
            L.append("")
            total = sum(d["duration_s"] for d in d_with)
            n = len(d_with)
            L.append(f"- Total measured: **{total}s** across {n} cases (mean {total/n:.1f}s/case)")
            # Top-10 slowest
            slow = sorted(d_with, key=lambda d: -d["duration_s"])[:10]
            L.append("")
            L.append("**Slowest 10 cases:**")
            L.append("")
            L.append("| # | Case | Category | Duration |")
            L.append("| --- | --- | --- | --- |")
            for i, d in enumerate(slow, 1):
                cat = cid_to_cat.get(d["id"], "?")
                L.append(f"| {i} | `{d['id']}` | {cat} | {d['duration_s']}s |")
            L.append("")
            # Per-category mean
            from collections import defaultdict
            cat_totals: dict[str, list[int]] = defaultdict(list)
            for d in d_with:
                cat_totals[cid_to_cat.get(d["id"], "?")].append(d["duration_s"])
            L.append("**Mean duration per category:**")
            L.append("")
            L.append("| Category | N | Mean | Max |")
            L.append("| --- | --- | --- | --- |")
            for cat in sorted(cat_totals):
                vals = cat_totals[cat]
                mean = sum(vals) / len(vals)
                L.append(f"| {cat} | {len(vals)} | {mean:.1f}s | {max(vals)}s |")
            L.append("")

    # Failing-case transcripts — the key value for an analyst
    rows = eval_result.get("rows", [])
    failures = [r for r in rows if not r["passed"]]
    if failures:
        L.append(f"## Failing-case transcripts ({len(failures)})")
        L.append("")
        L.append("Each block has the case description, why it failed, the assistant transcript, and the actual tool calls. An analyst should be able to propose a fix from these blocks alone.")
        L.append("")
        for r in failures:
            L.append(f"### `{r['id']}`  ({r['category']})")
            desc = r.get('description', '').strip()
            if desc:
                L.append(f"_{desc}_")
                L.append("")
            L.append("**Why it failed:**")
            for f in r.get("failures", []) or []:
                L.append(f"- {f}")
            L.append("")
            texts = r.get("assistant_texts", []) or []
            if texts:
                L.append("**Assistant transcript:**")
                for t in texts:
                    L.append(f"> {t}")
                L.append("")
            tcs = r.get("tool_calls", []) or []
            if tcs:
                L.append("**Tool calls:**")
                for tc in tcs:
                    ok = "ok" if tc.get("result", {}).get("ok") else "FAIL"
                    L.append(f"- `{tc['name']}` [{ok}] args=`{tc['args']}`")
                L.append("")

    # Closing
    L.append("---")
    L.append("")
    L.append("**For the analyst (Claude / teammate):** open this file, scan exec summary, look at any 🔴 regressions, then dive into the per-case transcripts. Propose fixes by category. Don't run anything — the next `qa.py` invocation will measure your suggestions.")
    return "\n".join(L)


# ============================================================================
# Main
# ============================================================================

def run_one(*, model: str | None, args, started_at: str) -> tuple[Path, dict]:
    """Run unit tests + eval for ONE model and produce one markdown report.
    Returns (report_path, eval_summary). Used both for single-model runs and
    multi-model comparison loops."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = (model or "default").replace(":", "_").replace("/", "_")
    progress_log = RUNS_DIR / f"qa_{timestamp}_{slug}_progress.log"
    if not args.skip_unit_tests:
        unit = run_unit_tests()
    else:
        unit = {"passed": 0, "failed": 0, "total": 0, "exit_code": 0,
                "output": "(skipped via --skip-unit-tests)"}
    eval_result = run_full_eval(args.concurrency, model, progress_log)
    eval_summary = summarize_eval(eval_result)
    baseline = load_baseline()
    regressed, recovered = diff_categories(eval_summary, baseline)
    has_regression = len(regressed) > 0
    append_history(eval_summary)
    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    durations = parse_case_durations(progress_log)
    report = render_report(
        started_at=started_at,
        finished_at=finished_at,
        git=git_state(),
        unit=unit,
        eval_result=eval_result,
        eval_summary=eval_summary,
        baseline=baseline,
        regressed=regressed,
        recovered=recovered,
        history=trend_lines(),
        has_regression=has_regression,
        durations=durations,
    )
    report_path = RUNS_DIR / f"qa_{timestamp}_{slug}.md"
    report_path.write_text(report)
    eval_summary["_model"] = model or "default"
    eval_summary["_report_path"] = str(report_path)
    eval_summary["_progress_log"] = str(progress_log)
    eval_summary["_has_regression"] = has_regression
    eval_summary["_unit_failed"] = unit["failed"]
    return report_path, eval_summary


def render_comparison(summaries: list[dict], started_at: str) -> str:
    """Side-by-side comparison report when running multiple models."""
    L: list[str] = []
    L.append(f"# Multi-model comparison — {datetime.datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    L.append(f"Started: {started_at}")
    L.append(f"Models: {', '.join(s['_model'] for s in summaries)}")
    L.append("")
    L.append("## Overall pass rate")
    L.append("")
    L.append("| Model | Pass | Total | Rate | Wall | LLM p50 | LLM p95 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in summaries:
        rate = s["passed"] / max(s["total"], 1) * 100
        wall = f"{s['wall_s']:.0f}s" if s.get("wall_s") else "?"
        L.append(
            f"| `{s['_model']}` | {s['passed']} | {s['total']} | {rate:.0f}% | "
            f"{wall} | {s['latency']['p50']} ms | {s['latency']['p95']} ms |"
        )
    L.append("")
    cats = sorted({c for s in summaries for c in s.get("by_category", {})})
    if cats:
        L.append("## Per-category pass count")
        L.append("")
        header = "| Category | " + " | ".join(f"`{s['_model']}`" for s in summaries) + " |"
        sep = "| --- " * (1 + len(summaries)) + "|"
        L.append(header); L.append(sep)
        for cat in cats:
            row = [cat]
            for s in summaries:
                c = s.get("by_category", {}).get(cat, {"pass": 0, "total": 0})
                row.append(f"{c['pass']}/{c['total']}")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L.append("## Per-case agreement matrix")
    L.append("")
    L.append("Cases where models disagree are the most informative.")
    L.append("")
    all_cases = sorted({cid for s in summaries for cid in s.get("by_case", {})})
    disagree = []
    for cid in all_cases:
        results = [s.get("by_case", {}).get(cid) for s in summaries]
        if len(set(results)) > 1 and None not in results:
            disagree.append((cid, results))
    L.append(f"**{len(disagree)} cases** where models disagree (out of {len(all_cases)}).")
    L.append("")
    if disagree:
        header = "| Case | " + " | ".join(f"`{s['_model']}`" for s in summaries) + " |"
        L.append(header)
        L.append("| --- " * (1 + len(summaries)) + "|")
        for cid, results in disagree[:50]:  # cap to first 50
            row = [f"`{cid}`"] + ["✅" if r else "❌" for r in results]
            L.append("| " + " | ".join(row) + " |")
        if len(disagree) > 50:
            L.append(f"_({len(disagree) - 50} more disagreements omitted)_")
        L.append("")
    L.append("## Individual reports")
    L.append("")
    for s in summaries:
        L.append(f"- `{s['_model']}`: [{Path(s['_report_path']).name}]({Path(s['_report_path']).name})")
    L.append("")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel subprocess shards (default 1; set higher only if "
                             "you accept noisier latency numbers).")
    parser.add_argument(
        "--model",
        help=(
            "Override LLM. Comma-separate to compare multiple models in one run "
            "(e.g. --model qwen2.5:14b,qwen2.5:7b,hermes3:8b). Each model gets its "
            "own report; a comparison report is written for any 2+ model run."
        ),
    )
    parser.add_argument("--no-update", action="store_true",
                        help="Don't update baseline.json even if there's no regression.")
    parser.add_argument("--skip-unit-tests", action="store_true",
                        help="Skip pytest (rare; usually you want them).")
    args = parser.parse_args()

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[qa] starting at {started_at}", flush=True)
    print(f"[qa] cwd: {ROOT}", flush=True)
    RUNS_DIR.mkdir(exist_ok=True)

    # Parse model list. Empty / not set = single default-model run.
    if args.model:
        model_list = [m.strip() for m in args.model.split(",") if m.strip()]
    else:
        model_list = [None]  # use harness default

    summaries: list[dict] = []
    overall_regression = False
    overall_unit_failed = 0
    for i, model in enumerate(model_list, 1):
        if len(model_list) > 1:
            print(f"\n[qa] === model {i}/{len(model_list)}: {model or '(default)'} ===\n", flush=True)
        report_path, summary = run_one(model=model, args=args, started_at=started_at)
        summaries.append(summary)
        overall_regression = overall_regression or summary["_has_regression"]
        overall_unit_failed += summary["_unit_failed"]
        pass_rate = summary["passed"] / max(summary["total"], 1) * 100
        print(f"[qa] {model or 'default'}: {summary['passed']}/{summary['total']} ({pass_rate:.0f}%) -> {report_path.name}", flush=True)

    # Multi-model comparison report
    comparison_path: Path | None = None
    if len(summaries) > 1:
        comparison = render_comparison(summaries, started_at)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = RUNS_DIR / f"qa_{ts}_comparison.md"
        comparison_path.write_text(comparison)
        print(f"[qa] multi-model comparison -> {comparison_path}", flush=True)

    # Update baseline if a SINGLE-model run AND no regression AND tests pass.
    # In multi-model mode we don't auto-update — too easy to overwrite a
    # canonical baseline with a non-canonical model's run.
    if len(model_list) == 1 and not args.no_update:
        s = summaries[0]
        if not s["_has_regression"] and s["_unit_failed"] == 0:
            BASELINE.write_text(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, indent=2))
            print(f"[qa] baseline updated -> {BASELINE}", flush=True)

    print()
    print("=" * 60)
    if comparison_path:
        print(f"COMPARISON REPORT: {comparison_path}")
    for s in summaries:
        rate = s["passed"] / max(s["total"], 1) * 100
        flag = "🔴" if s["_has_regression"] else ("🟡" if s["_unit_failed"] else "🟢")
        print(f"  {flag} {s['_model']:<22} {s['passed']:>3}/{s['total']} ({rate:>3.0f}%)  ->  {Path(s['_report_path']).name}")
    print("=" * 60)
    print()
    return 1 if overall_regression else 0


if __name__ == "__main__":
    sys.exit(main())
