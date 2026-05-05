"""Quick ASCII trend view over eval/history.jsonl.

After watch.py has run a few times you'll have a record per run. This
script prints:

  - last N runs' pass rate as a sparkline + table
  - per-category last-N pass-rate sparkline
  - LLM-call p50 / p95 latency sparkline

No deps beyond stdlib so it runs anywhere. For richer plots use
something else — this is the "did anything change this week" glance tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HISTORY = Path(__file__).resolve().parent / "history.jsonl"

# Eight steps from low to high — readable on terminals.
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return SPARK_CHARS[4] * len(values)
    out = []
    span = hi - lo
    for v in values:
        i = int(round((v - lo) / span * (len(SPARK_CHARS) - 1)))
        out.append(SPARK_CHARS[max(0, min(len(SPARK_CHARS) - 1, i))])
    return "".join(out)


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="Show last N runs")
    parser.add_argument("--history", default=str(HISTORY))
    args = parser.parse_args()

    rows = load_history(Path(args.history))
    if not rows:
        print(f"No history at {args.history!r}. Run eval/watch.py to populate.")
        return 1

    rows = rows[-args.last :]

    pass_rates = [r["passed"] / r["total"] * 100 for r in rows]
    p50s = [r["latency"]["p50"] for r in rows]
    p95s = [r["latency"]["p95"] for r in rows]

    print(f"Showing last {len(rows)} runs from {args.history}\n")

    # Header sparkline
    print("Pass rate (%):")
    print(f"  {sparkline(pass_rates)}  range [{min(pass_rates):.0f}, {max(pass_rates):.0f}]")
    print(f"  current: {pass_rates[-1]:.0f}%   first shown: {pass_rates[0]:.0f}%")
    print()

    print("LLM p50 (ms):")
    print(f"  {sparkline(p50s)}  range [{min(p50s)}, {max(p50s)}]")
    print(f"  current: {p50s[-1]} ms   first shown: {p50s[0]} ms")
    print()

    print("LLM p95 (ms):")
    print(f"  {sparkline(p95s)}  range [{min(p95s)}, {max(p95s)}]")
    print(f"  current: {p95s[-1]} ms   first shown: {p95s[0]} ms")
    print()

    # Per-category trend
    cats = sorted({c for r in rows for c in r.get("by_category", {})})
    if cats:
        print("Per-category pass count:")
        for c in cats:
            series = [r.get("by_category", {}).get(c, {}).get("pass", 0) for r in rows]
            cur = series[-1]
            tot = rows[-1].get("by_category", {}).get(c, {}).get("total", 0)
            print(f"  {c:<14} {sparkline([float(x) for x in series])}  {cur}/{tot}")
        print()

    # Tabular bottom — last 10
    n = min(10, len(rows))
    print(f"Last {n} runs:")
    print(f"  {'when':<22} {'pass':<8} {'wall':<7} {'p50':<6} {'p95':<6}")
    for r in rows[-n:]:
        ts = r.get("ts", "?")[:19]
        passed = f"{r['passed']}/{r['total']}"
        wall = f"{r.get('wall_s') or 0:.0f}s"
        lat = r.get("latency", {})
        p50 = str(lat.get("p50", "?"))
        p95 = str(lat.get("p95", "?"))
        print(f"  {ts:<22} {passed:<8} {wall:<7} {p50:<6} {p95:<6}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
