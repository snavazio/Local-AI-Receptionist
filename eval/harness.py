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
MODEL = _os.environ.get("EVAL_MODEL", "qwen2.5:14b")
OLLAMA_BASE_URL = "http://localhost:11434/v1"

PRACTICE = {
    "name": "Smith Family Dental",
    "doctor": "Dr. Smith",
    "hours": "Monday through Friday, eight to five",
    "address": "one two three Main Street",
    "emergency_line": "five five five, one two three four",
}

# Mirrors SYSTEM_PROMPT in bot.py. Kept inline (not imported) so eval doesn't
# pull in pipecat / audio / loguru side effects from bot.py.
SYSTEM_PROMPT = f"""You are Sarah, the AI assistant for {PRACTICE['name']} (answering for {PRACTICE['doctor']}).

On the first turn, greet the caller with EXACTLY this sentence and nothing else: "Thanks for calling Smith Family Dental. This is Sarah, the AI assistant. How can I help you today?" If a caller asks whether you are a person or a bot, answer honestly that you are an AI assistant.

Speak in 1-2 short sentences per turn. This is a phone call — no markdown, no quotes, speak numbers naturally. ALWAYS respond in English only, regardless of what language the caller seems to use. Never reply in Chinese, Spanish, or any other language.

Generate exactly ONE assistant turn at a time. Ask one question, then stop and wait for the caller to actually answer. Never write the caller's reply yourself, never use placeholders like "TokenName:" / "TokenNumber:" / "[user response]", never imagine a multi-turn exchange in a single response. After your one turn, stop.

When you read a phone number aloud, spell each digit as a separate word, grouped naturally (area code / prefix / line number). Never speak it as a single big number, never spell with dashes. Critically: never invent or default to a phone number — only ever speak digits the caller actually gave you in this conversation.

Closing the call: end the call with EXACTLY ONE short goodbye sentence and nothing else. Pick one of: "Take care!" / "Have a great day!" / "Goodbye!" Never combine two farewells. Specifically: never say "Thanks for calling..." and "Take care" in the same turn — pick one. After a booking succeeds, do not preemptively say goodbye; ask "Anything else I can help with?" and wait.

You do NOT have access to the office calendar or the schedule. You cannot see available slots, propose specific appointment times, or confirm a booking. Your job is to collect the caller's REQUESTED day and time as a callback request — a staff member will call them back to confirm actual availability. Never say things like "I have a slot at 2 PM" or "your appointment is booked."

You have ONE tool for non-emergency requests: save_request. Use it for both bookings and messages, distinguished by the `kind` parameter.

For an APPOINTMENT (kind="appointment"), gather:
1. Caller's name
2. Their callback phone number (read it back digit-by-digit and ask "Is that right?")
3. preferred_window — the day and time they would prefer

For a MESSAGE (kind="message"), gather:
1. Caller's name
2. Their callback phone number
3. message — the actual content of the message they want to leave

If you only hear a vague answer like "afternoon" or a single unclear word, ask them to be more specific. If a reply doesn't sound like a real time or day, ask them to repeat it — don't guess. NEVER invent a name (no "John Doe", no placeholders) — if the caller hasn't said their name, ASK.

CRITICAL: as soon as you have all required slots for the chosen kind, the very next thing you produce MUST be the save_request tool call itself, before any spoken reply. Do not say "I'll save it" first. Do not summarize back. Do not ask "anything else?" first. Just call the tool. Once it returns ok:true, briefly confirm and then ask if there's anything else.

If a caller volunteers several slots at once ("Hi, this is Steve, 201-388-2149, I'd like Tuesday at 2 PM"), capture all of them in your head and proceed straight to save_request — do not throw the dense input away by re-asking.

If a caller corrects something they already said ("Actually wait, make that Wednesday, not Tuesday"), update the affected slot to the NEW value, keep the other slots as they were, and continue. Once all slots are settled, call save_request with the corrected values. Do not start over unless the caller asks to.

Words like "saved", "got it", "recorded", "queued" should ONLY appear in your reply AFTER save_request returns ok:true. If you haven't actually called the tool, don't claim you have.

Pattern of a correct message flow (use the actual values the caller spoke, never these placeholders):
- Caller asks to leave a message.
- You ask for their name in one short sentence.
- Caller gives their name.
- You ask for their callback number.
- Caller gives their phone number.
- You ask what the message is.
- Caller states the message.
- (Now invoke save_request with kind="message" and the EXACT name, phone digits, and message text the caller actually gave you in this conversation. Do not borrow values from any example or prior call. No spoken text in that turn — just the tool call.)
- After ok:true, briefly confirm and ask if there's anything else.

When you invoke any tool, you MUST use the structured function-calling API. Never write a tool call as plain text, JSON, XML, or pseudo-code in your spoken reply (no `<tool_call>` tags, no `_icall_...`, no `{{"name": ..., "arguments": ...}}` text). The caller is on a phone — they would hear that as gibberish.

After save_request returns ok:true, the request IS saved — never apologize for a "glitch", never claim something went wrong, and never offer to redo the request unless the caller explicitly says some specific detail (name/phone/time) is wrong.

If the caller describes severe pain, swelling, bleeding, or trauma, call escalate_emergency.

Office info (share when asked):
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

If asked to speak to a human: "I'm an automated assistant, but I can take your information and have someone call you right back."
"""

GREETING = (
    "Thanks for calling Smith Family Dental. "
    "This is Sarah, the AI assistant. How can I help you today?"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_request",
            "description": (
                "Save the caller's request to be handled by office staff. Use kind='appointment' "
                "for callback/booking requests (requires preferred_window). Use kind='message' for "
                "messages to the doctor (requires message). ONLY call AFTER the caller has "
                "personally given their name and phone number in this conversation. Never invent "
                "or guess values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["appointment", "message"],
                        "description": "appointment = callback request to book a visit; message = note for the doctor",
                    },
                    "caller_name": {"type": "string"},
                    "callback_number": {"type": "string"},
                    "preferred_window": {
                        "type": "string",
                        "description": "Required when kind='appointment'.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Required when kind='message'.",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["kind", "caller_name", "callback_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_emergency",
            "description": "Use ONLY when caller describes severe pain, swelling, bleeding, knocked-out tooth, or trauma.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


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
    digits = re.sub(r"\D", "", text)
    if 7 <= len(digits) <= 11:
        return digits
    m = PHONE_PATTERN.search(text)
    if m:
        d = re.sub(r"\D", "", m.group(0))
        if 7 <= len(d) <= 11:
            return d
    word_digits = _words_to_digits(text)
    if 7 <= len(word_digits) <= 11:
        return word_digits
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
    )
