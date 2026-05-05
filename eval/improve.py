"""Autonomous prompt-only improvement loop for the receptionist bot.

This is the constrained version of the "agent that does what I'm doing" pattern.
It:

  1. Reads the current SYSTEM_PROMPT from bot.py / harness.py.
  2. Identifies the worst-performing case category from the latest baseline.json.
  3. Generates N candidate prompt variants (small, targeted edits — extra
     emphasis on the failing rule, an example, etc.).
  4. For each candidate: writes it to a temp prompt, runs the eval on the
     failing category (~10 cases × 30s = ~5 min/candidate), records pass rate.
  5. If a candidate beats baseline by ≥3 percentage points, *prints* the diff
     to candidate.txt for human review. Does NOT auto-merge.
  6. Stops after MAX_ITERATIONS or when no candidate beats baseline.

What this is NOT:
  - It does not edit code outside the SYSTEM_PROMPT string.
  - It does not commit or push changes.
  - It does not run the FULL 100-case eval per candidate (too slow).

Why constrained: prompt-tuning against a noisy eval can drift. The point is
to produce *candidates for human review*, not to autonomously rewrite the bot.

Status: scaffold. The candidate-generation step is hand-written templates;
swapping in a real LLM-driven generator (DSPy-style) would be the next step.

Usage (when you're ready):
  python eval/improve.py --max-iterations 5 --category message
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
HARNESS = EVAL_DIR / "harness.py"
BASELINE = EVAL_DIR / "baseline.json"
RUN_EVAL = EVAL_DIR / "run_eval.py"

# Hard guardrails so this never burns through the GPU unsupervised.
MAX_ITERATIONS = 10
MIN_IMPROVEMENT_PP = 3  # at least +3 percentage points to be considered a win
RUNS_PER_CANDIDATE = 1  # raise to 3+ once you've characterized noise floor


def read_current_prompt() -> str:
    """Extract the SYSTEM_PROMPT f-string from harness.py.

    We use harness.py rather than bot.py because the former is what the eval
    actually executes. Real prompt updates should land in both; the human
    review step ensures that."""
    src = HARNESS.read_text()
    m = re.search(r'SYSTEM_PROMPT\s*=\s*f"""(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not locate SYSTEM_PROMPT in harness.py")
    return m.group(1)


def write_prompt_candidate(new_prompt: str) -> Path:
    """Patch harness.py with the candidate prompt and return the patched
    path (a temp copy) so we can run it without touching the real file."""
    src = HARNESS.read_text()
    # Escape backslashes and triple-quotes in the candidate
    escaped = new_prompt.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    patched = re.sub(
        r'(SYSTEM_PROMPT\s*=\s*f""")(.*?)(""")',
        lambda m: m.group(1) + escaped + m.group(3),
        src,
        count=1,
        flags=re.DOTALL,
    )
    tmp = Path(tempfile.mkdtemp()) / "harness.py"
    tmp.write_text(patched)
    return tmp


def candidate_variants(prompt: str, weak_category: str) -> list[tuple[str, str]]:
    """Generate variants of the system prompt, targeting a weak category.

    Returns list of (label, prompt_text). Hand-written templates for now;
    swapping for an LLM-generated variant set is one upgrade away."""
    variants = []

    if weak_category == "message":
        # Variant A: stronger insistence
        v = prompt.replace(
            "CRITICAL: as soon as you have all required slots",
            "CRITICAL — THIS IS NON-NEGOTIABLE: the moment you have all required slots",
        )
        if v != prompt:
            variants.append(("message_strong_insist", v))

        # Variant B: contrastive example (right vs wrong)
        addition = (
            "\n\nWRONG: 'I'll save your message now.' (then no tool call) — "
            "the call ends with no record. CORRECT: emit save_request "
            "immediately, no preamble."
        )
        variants.append(("message_contrastive", prompt + addition))

    elif weak_category == "emergency":
        # Stronger trigger language
        v = prompt.replace(
            "If the caller describes severe pain",
            "If the caller mentions any of: severe pain, swelling, "
            "bleeding, knocked-out tooth, trauma, infection, or facial "
            "injury — IMMEDIATELY call escalate_emergency on that turn",
        )
        if v != prompt:
            variants.append(("emergency_strong_trigger", v))

    elif weak_category == "correction":
        v = prompt + (
            "\n\nWhen the caller corrects a previously-stated detail (day, "
            "time, name, or phone), update your understanding to the NEW "
            "value and discard the old one. The latest correction always wins."
        )
        variants.append(("correction_latest_wins", v))

    return variants


def find_weakest_category(baseline_path: Path) -> tuple[str, int, int]:
    if not baseline_path.exists():
        raise RuntimeError(
            f"No baseline at {baseline_path}. Run watch.py --update-baseline first."
        )
    b = json.loads(baseline_path.read_text())
    by_cat = b.get("by_category", {})
    if not by_cat:
        raise RuntimeError("Baseline has no by_category data")
    worst = min(by_cat.items(), key=lambda x: x[1]["pass"] / max(x[1]["total"], 1))
    cat, c = worst
    return cat, c["pass"], c["total"]


def run_eval_on_category(category: str, env: dict) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)
    try:
        cmd = [
            sys.executable, str(RUN_EVAL),
            "--category", category,
            "--json-out", str(out),
        ]
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if not out.exists() or out.stat().st_size == 0:
            return 0, 0
        data = json.loads(out.read_text())
        rows = data.get("rows", [])
        return sum(1 for r in rows if r["passed"]), len(rows)
    finally:
        out.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--category", default=None,
                        help="Force a category to optimize against (overrides baseline lookup)")
    parser.add_argument("--candidates-out", default=str(EVAL_DIR / "candidates.md"))
    args = parser.parse_args()

    if args.category:
        weak_cat = args.category
        # We don't know the baseline number for this case if user forced it
        baseline_pass, baseline_total = 0, 10
    else:
        weak_cat, baseline_pass, baseline_total = find_weakest_category(BASELINE)

    print(f"[improve] weakest category: {weak_cat} ({baseline_pass}/{baseline_total})")

    prompt = read_current_prompt()
    variants = candidate_variants(prompt, weak_cat)
    if not variants:
        print(f"[improve] no candidate templates for category={weak_cat!r}")
        print("         add a branch to candidate_variants() and retry.")
        return 1

    winners: list[tuple[str, int, int]] = []
    candidates_md = [f"# Candidate prompts for {weak_cat} — {datetime.datetime.now().isoformat()}\n"]
    candidates_md.append(f"Baseline on this category: {baseline_pass}/{baseline_total}\n")

    for i, (label, candidate_prompt) in enumerate(variants[: args.max_iterations]):
        print(f"\n[improve] candidate {i + 1}/{len(variants)}: {label}", flush=True)
        patched = write_prompt_candidate(candidate_prompt)
        env = {
            **dict(__import__("os").environ),
            "PYTHONPATH": str(patched.parent) + ":" + dict(__import__("os").environ).get("PYTHONPATH", ""),
        }
        # We can't easily redirect harness.py — for now this scaffold just
        # logs the variant. A real run would import harness from the patched
        # path or hot-swap the module. That's where real-LLM-generated
        # variants would plug in.
        candidates_md.append(f"\n## Candidate: `{label}`\n")
        candidates_md.append("```text")
        candidates_md.append(candidate_prompt[:400] + ("..." if len(candidate_prompt) > 400 else ""))
        candidates_md.append("```")
        candidates_md.append(f"\n_Patched harness at: {patched}_\n")
        # Don't actually run yet — leave the hook for human review.

    Path(args.candidates_out).write_text("\n".join(candidates_md))
    print(f"\n[improve] wrote {len(variants)} candidate(s) to {args.candidates_out}")
    print("[improve] this is currently a scaffold — review the candidates.md,")
    print("          pick one to apply manually, run watch.py to verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
