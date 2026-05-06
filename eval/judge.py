"""LLM-as-judge: grade an eval transcript on conversation quality.

When run_eval.py reports "PASS", it just means the rule-based assertions
held — tool was called, must-say substrings were present, etc. That doesn't
catch problems like:
  - Tool was called correctly but the caller had to repeat themselves 4 times
  - Bot was condescending
  - Bot's prose was robotic / unnatural
  - Bot missed obvious context cues from the caller

This module runs a SECOND LLM pass over the captured transcript, asking it
to grade specific qualitative dimensions on a 1-5 scale.

The output is advisory — it's added to the qa report as a side-channel score.
We don't fail cases on judge scores (too fragile); we use them to flag
candidates for human review.

Usage:
    from eval.judge import judge_case
    score = judge_case(case, result)  # returns dict with scores + reasons

The judge model is configurable (default: same as the eval model). Calling
it adds ~1 LLM call per case, so factor that into runtime.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("EVAL_MODEL", "qwen2.5:14b"))
OLLAMA_BASE_URL = "http://localhost:11434/v1"


JUDGE_PROMPT = """You are an evaluator grading a phone-receptionist bot's conversation quality.

Below is a single test case: the scripted caller turns and the bot's actual responses.

Your job: rate the conversation on the following dimensions, each 1 (terrible) to 5 (excellent):

1. **conciseness** — was the bot brief and direct, or did it ramble?
2. **clarity** — were the bot's questions and confirmations clear and unambiguous?
3. **task_completion** — did the bot drive the conversation toward completing the caller's actual goal?
4. **naturalness** — did the bot sound human-like, or robotic / repetitive / condescending?
5. **error_recovery** — when the caller corrected something or there was confusion, did the bot recover gracefully?

Output JSON ONLY, exactly this shape (no extra prose):
{
  "conciseness": 1-5,
  "clarity": 1-5,
  "task_completion": 1-5,
  "naturalness": 1-5,
  "error_recovery": 1-5,
  "summary": "one short sentence about the conversation's overall quality"
}

CASE DESCRIPTION: {description}

CALLER TURNS:
{user_turns}

BOT RESPONSES:
{assistant_texts}

TOOL CALLS THE BOT MADE:
{tool_calls}
"""


@dataclass
class JudgeScore:
    conciseness: int = 0
    clarity: int = 0
    task_completion: int = 0
    naturalness: int = 0
    error_recovery: int = 0
    summary: str = ""
    raw: str = ""  # the model's raw output, for debugging
    error: str = ""  # populated if parsing fails

    @property
    def average(self) -> float:
        nums = [self.conciseness, self.clarity, self.task_completion,
                self.naturalness, self.error_recovery]
        nums = [n for n in nums if n > 0]
        return round(sum(nums) / len(nums), 2) if nums else 0.0


def judge_case(*, case: dict, user_turns: list[str], assistant_texts: list[str],
               tool_calls: list[dict], client: OpenAI | None = None) -> JudgeScore:
    """Run the judge model and return a JudgeScore."""
    own_client = False
    if client is None:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        own_client = True

    prompt = JUDGE_PROMPT.format(
        description=case.get("description", "(no description)").strip(),
        user_turns="\n".join(f"  caller: {t}" for t in user_turns),
        assistant_texts="\n".join(f"  bot: {t}" for t in assistant_texts),
        tool_calls=json.dumps(tool_calls, indent=2) if tool_calls else "(no tool calls)",
    )

    score = JudgeScore()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        score.raw = raw
        # Extract the JSON object — be lenient about prose around it
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            score.error = f"no JSON object found in judge output"
            return score
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            score.error = f"json parse: {e}"
            return score
        score.conciseness = int(data.get("conciseness", 0) or 0)
        score.clarity = int(data.get("clarity", 0) or 0)
        score.task_completion = int(data.get("task_completion", 0) or 0)
        score.naturalness = int(data.get("naturalness", 0) or 0)
        score.error_recovery = int(data.get("error_recovery", 0) or 0)
        score.summary = str(data.get("summary", "") or "")
    except Exception as e:
        score.error = f"judge failed: {e!r}"
    finally:
        if own_client:
            client.close()
    return score
