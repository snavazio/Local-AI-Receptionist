"""Text-only eval harness for the receptionist bot.

Skips audio: drives the LLM + tools directly via Ollama's OpenAI-compatible
endpoint. Captures assistant text + tool calls per turn so we can assert on
both. Mirrors the gating logic from bot.py so tool-side validators are tested
too, not just the LLM.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI


import os as _os
import sys as _sys
from pathlib import Path as _Path

MODEL = _os.environ.get("EVAL_MODEL", "qwen2.5:14b")
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Domain config (business facts, persona, tool schemas, system prompt, greeting)
# is loaded from config/<domain>.yaml — same source bot.py uses, so the eval
# can never drift from the live agent's prompt/tools.
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_PROJECT_ROOT))
from config.loader import (  # noqa: E402
    PRACTICE, GREETING, SYSTEM_PROMPT, TOOLS,
)


# ---------- gating logic mirrored from bot.py ----------
PLACEHOLDER_VALUES = {
    "", "unknown", "none", "null", "n/a", "na", "tbd", "to be determined",
    "string", "name", "phone", "number", "callback_number", "caller_name",
    "message", "the caller", "caller", "user", "anonymous", "no name",
    "no number", "not provided", "not given", "nil",
    "john doe", "jane doe", "john smith", "jane smith", "test", "test user",
}

PHONE_PATTERN = re.compile(
    r"\(?\b\d{3}\)?[\s.\-]*\d{3}[\s.\-]*\d{4}\b"
    r"|\b\d{7,11}\b"
)


def _normalize_placeholder(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"^[<\[\(\{]+|[>\]\)\}]+$", "", s.strip()).lower()


def _missing(args: dict, *fields) -> list:
    out = []
    for f in fields:
        v = args.get(f)
        if v is None:
            out.append(f); continue
        if not isinstance(v, str):
            continue
        normalized = _normalize_placeholder(v)
        if normalized in PLACEHOLDER_VALUES:
            out.append(f); continue
        if f == "callback_number" and _extract_phone_digits(v) is None:
            out.append(f); continue
        if f == "caller_name" and len(normalized) < 2:
            out.append(f)
    return out


_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _words_to_digits(text: str) -> str:
    out = []
    for tok in re.findall(r"[a-zA-Z]+|\d", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in _WORD_TO_DIGIT:
            out.append(_WORD_TO_DIGIT[tok])
    return "".join(out)


def _extract_phone_digits(text: str) -> str | None:
    if not text:
        return None
    # Try the STRUCTURED phone pattern first (e.g. "201-388-2149" or
    # "(201) 388-2149"). This avoids concatenating unrelated digit
    # groups — e.g. "Tuesday at 2 PM. Phone 201-388-2149" must NOT
    # become "22013882149".
    m = PHONE_PATTERN.search(text)
    if m:
        d = re.sub(r"\D", "", m.group(0))
        if 7 <= len(d) <= 11:
            return d
    # Then try word-form digits (e.g. "two zero one three eight eight
    # two one four nine"). Only when there's no structured phone.
    word_digits = _words_to_digits(text)
    if 7 <= len(word_digits) <= 11:
        return word_digits
    # Last resort: strip all non-digits and check. Risky (can concat
    # unrelated digits) — only fires if the prior two paths failed.
    digits = re.sub(r"\D", "", text)
    if 7 <= len(digits) <= 11:
        return digits
    return None


# ---------- stub tool implementations (record + validate, no disk I/O) ----------
@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: dict


def execute_tool(name: str, args: dict) -> dict:
    """Mirrors bot.py's tool gating, returns the same shape of response, but
    skips writing JSON files. Returns the dict the LLM will see as tool_result."""

    if name == "save_request":
        kind = (args.get("kind") or "").strip().lower()
        if kind == "appointment":
            missing = _missing(args, "caller_name", "callback_number", "preferred_window")
            ok_msg = "Got it, your callback is saved."
        elif kind == "message":
            missing = _missing(args, "caller_name", "callback_number", "message")
            ok_msg = "Message saved."
        else:
            return {"ok": False, "error": "Bad kind"}
        if missing:
            return {"ok": False, "error": f"Missing {missing}"}
        digits = _extract_phone_digits(args.get("callback_number", ""))
        if digits is None or len(digits) < 7:
            return {"ok": False, "error": "Bad phone"}
        return {"ok": True, "kind": kind, "spoken_response": ok_msg}

    if name == "escalate_emergency":
        return {
            "ok": True,
            "spoken_response": (
                f"For dental emergencies please hang up and call "
                f"{PRACTICE['emergency_line']} immediately."
            ),
        }

    return {"ok": False, "error": f"Unknown tool {name}"}


# ---------- production-equivalent post-processors ----------
# These mirror the bot.py FrameProcessors that sit between the LLM and the TTS
# in production. The eval applies them to captured assistant text so assertions
# reflect what the caller actually hears, not raw LLM output.

_TOOLCALL_LEAK_PATTERNS = [
    re.compile(r"<\s*tool_call\s*>.*?</\s*tool_call\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*tool_call\s*>.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"</\s*tool_call\s*>", re.IGNORECASE),
    re.compile(r"_icall_[^\s]*", re.IGNORECASE),
    re.compile(r"\biNdEx[^\s]*", re.IGNORECASE),
    re.compile(
        r'\{\s*"name"\s*:\s*"[A-Za-z_][\w]*"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',
        re.DOTALL,
    ),
]

_FAREWELL_RE = re.compile(
    r"\b(take care|have a (great|good|nice|wonderful|lovely) day"
    r"|good ?bye|bye now|see you|we look forward|talk to you (soon|later))\b",
    re.IGNORECASE,
)


def _strip_malformed_tool_call(txt: str) -> str:
    """Mirror bot.py MalformedToolCallStripper. Returns cleaned text, or '' if
    nothing legible remains."""
    cleaned = txt
    for pat in _TOOLCALL_LEAK_PATTERNS:
        cleaned = pat.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------- conversation runner ----------
@dataclass
class Turn:
    role: str               # "assistant" | "user" | "tool"
    text: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    turns: list[Turn]
    tool_calls: list[ToolCallRecord]
    transcript: list[dict]  # raw OpenAI-format messages for debugging
    llm_call_ms: list[int] = field(default_factory=list)  # per-LLM-request wall time
    prompt_tokens: int = 0   # cumulative input tokens across all LLM calls
    completion_tokens: int = 0  # cumulative output tokens


async def run_case_async(
    case_id: str,
    user_turns: list[str],
    max_tool_loops: int = 5,
    client: AsyncOpenAI | None = None,
) -> CaseResult:
    """Async version of run_case for concurrent execution. Records per-LLM-call
    wall time so we can measure how latency degrades under load."""
    own_client = False
    if client is None:
        client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        own_client = True

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
    ]
    captured_turns = [Turn(role="assistant", text=GREETING)]
    captured_tool_calls: list[ToolCallRecord] = []
    llm_call_ms: list[int] = []
    prompt_tokens = 0
    completion_tokens = 0
    farewell_spoken = False

    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        captured_turns.append(Turn(role="user", text=user_text))

        for _ in range(max_tool_loops):
            t0 = time.monotonic()
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0,
            )
            llm_call_ms.append(int((time.monotonic() - t0) * 1000))
            if getattr(resp, "usage", None):
                prompt_tokens += resp.usage.prompt_tokens or 0
                completion_tokens += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message

            assistant_msg: dict = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            spoken_text = msg.content or ""
            if spoken_text:
                spoken_text = _strip_malformed_tool_call(spoken_text)
                if farewell_spoken and spoken_text:
                    spoken_text = ""
                elif spoken_text and _FAREWELL_RE.search(spoken_text):
                    farewell_spoken = True

            captured_turns.append(
                Turn(
                    role="assistant",
                    text=spoken_text,
                    tool_calls=[
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                )
            )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = execute_tool(tc.function.name, args)
                    captured_tool_calls.append(
                        ToolCallRecord(name=tc.function.name, args=args, result=result)
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                continue
            break

    if own_client:
        await client.close()

    return CaseResult(
        case_id=case_id,
        turns=captured_turns,
        tool_calls=captured_tool_calls,
        transcript=messages,
        llm_call_ms=llm_call_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def run_cases_concurrent(
    cases: list[dict],
    concurrency: int = 1,
) -> list[CaseResult]:
    """Run a list of cases with up to `concurrency` running at any moment.

    Reuses one AsyncOpenAI client across all tasks (Ollama handles concurrent
    requests serverside; the client is connection-pooled). Returns results in
    the same order as the input cases."""
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    sem = asyncio.Semaphore(concurrency)
    results: list[CaseResult | None] = [None] * len(cases)

    async def _run(i: int, case: dict):
        async with sem:
            print(f"  starting {case['id']}", flush=True)
            results[i] = await run_case_async(
                case["id"], case["user_turns"], client=client
            )
            print(f"  done {case['id']}", flush=True)

    await asyncio.gather(*[_run(i, c) for i, c in enumerate(cases)])
    await client.close()
    return [r for r in results if r is not None]


def run_case(case_id: str, user_turns: list[str], max_tool_loops: int = 5) -> CaseResult:
    """Run a single case. user_turns is the scripted list of caller utterances.

    Each user turn triggers an LLM call. The LLM may emit tool calls; we
    execute them and feed results back, looping until the model emits text
    (or hits max_tool_loops). Then we move to the next user turn."""

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    # Seed the context with the hardcoded greeting (matches bot.py's connect handler).
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": GREETING},
    ]

    captured_turns = [Turn(role="assistant", text=GREETING)]
    captured_tool_calls: list[ToolCallRecord] = []
    llm_call_ms: list[int] = []
    prompt_tokens = 0
    completion_tokens = 0
    farewell_spoken = False  # latches once per call (mirrors prod FarewellDeduper)

    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        captured_turns.append(Turn(role="user", text=user_text))

        for _ in range(max_tool_loops):
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0,
            )
            llm_call_ms.append(int((time.monotonic() - t0) * 1000))
            if getattr(resp, "usage", None):
                prompt_tokens += resp.usage.prompt_tokens or 0
                completion_tokens += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message

            # Record assistant message in OpenAI format for the next round.
            assistant_msg: dict = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # Apply production post-processors to the assistant text so the
            # eval sees what the caller would actually hear.
            spoken_text = msg.content or ""
            if spoken_text:
                spoken_text = _strip_malformed_tool_call(spoken_text)
                if farewell_spoken and spoken_text:
                    spoken_text = ""  # FarewellDeduper drops everything post-farewell
                elif spoken_text and _FAREWELL_RE.search(spoken_text):
                    farewell_spoken = True

            captured_turns.append(
                Turn(
                    role="assistant",
                    text=spoken_text,
                    tool_calls=[
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in (msg.tool_calls or [])
                    ],
                )
            )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = execute_tool(tc.function.name, args)
                    captured_tool_calls.append(
                        ToolCallRecord(name=tc.function.name, args=args, result=result)
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
                # Loop again so LLM can respond to the tool result.
                continue

            # No tool calls — assistant turn is done, move to next user turn.
            break

    return CaseResult(
        case_id=case_id,
        turns=captured_turns,
        tool_calls=captured_tool_calls,
        transcript=messages,
        llm_call_ms=llm_call_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
