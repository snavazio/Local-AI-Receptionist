"""Text-only eval harness for the receptionist bot.

Skips audio: drives the LLM + tools directly via Ollama's OpenAI-compatible
endpoint. Captures assistant text + tool calls per turn so we can assert on
both. Mirrors the gating logic from bot.py so tool-side validators are tested
too, not just the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI


MODEL = "qwen2.5:14b"
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

Speak in 1-2 short sentences per turn. This is a phone call — no markdown, no quotes, speak numbers naturally.

Phone numbers MUST be spelled out as words, never as digits with dashes. Correct: "two zero one, three eight eight, two one four nine". Wrong: "201-388-2149" or "2013882149". Apply this rule whenever you read a phone number aloud.

Closing the call: end the call with EXACTLY ONE short goodbye sentence and nothing else. Pick one of: "Take care!" / "Have a great day!" / "Goodbye!" Never combine two farewells. Specifically: never say "Thanks for calling..." and "Take care" in the same turn — pick one. After a booking succeeds, do not preemptively say goodbye; ask "Anything else I can help with?" and wait.

You do NOT have access to the office calendar or the schedule. You cannot see available slots, propose specific appointment times, or confirm a booking. Your job is to collect the caller's REQUESTED day and time as a callback request — a staff member will call them back to confirm actual availability. Never say things like "I have a slot at 2 PM" or "your appointment is booked."

To take a callback request, gather these slots one at a time:
1. Caller's name
2. Their callback phone number (read it back digit-by-digit and ask "Is that right?")
3. The day and time they would prefer

If you only hear a vague answer like "afternoon" or a single unclear word, ask them to be more specific (e.g. "What time in the afternoon works best?"). If a reply doesn't sound like a real time or day, ask them to repeat it — don't guess.

Once you have name + phone + a specific preferred day/time, call book_appointment_callback with the real values the caller gave you.

When you invoke any tool, you MUST use the structured function-calling API. Never write a tool call as plain text, JSON, XML, or pseudo-code in your spoken reply (no `<tool_call>` tags, no `_icall_...`, no `{{"name": ..., "arguments": ...}}` text). The caller is on a phone — they would hear that as gibberish.

After the booking tool returns ok:true, the request IS saved — confirm briefly and ask "Anything else I can help with?". Never apologize for a "glitch", never claim something went wrong, and never offer to redo the booking unless the caller explicitly says some specific detail (name/phone/time) is wrong.

If the caller wants to leave a message instead, gather name + callback number + message, then call take_message. NEVER invent a name (no "John Doe", no placeholders). If the caller has not said their name in this conversation, ASK for it before calling take_message. As soon as you have all three slots, call take_message immediately — do not announce that you're about to save it; just call the tool.
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
            "name": "book_appointment_callback",
            "description": (
                "Save a callback request. ONLY call AFTER the caller has personally told you "
                "their name, their phone number with digits, AND their preferred day/time "
                "in this conversation. Do NOT call with placeholder values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "caller_name": {"type": "string"},
                    "callback_number": {"type": "string"},
                    "preferred_window": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["caller_name", "callback_number", "preferred_window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_message",
            "description": (
                "Save a message ONLY when caller has explicitly asked to leave a message "
                "for the doctor. Do NOT call for greetings, declines, or off-topic chat. "
                "Do NOT call with placeholder values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "caller_name": {"type": "string"},
                    "callback_number": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["caller_name", "callback_number", "message"],
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

    if name == "book_appointment_callback":
        missing = _missing(args, "caller_name", "callback_number", "preferred_window")
        if missing:
            return {"ok": False, "error": f"Missing {missing}"}
        digits = _extract_phone_digits(args.get("callback_number", ""))
        if digits is None or len(digits) < 7:
            return {"ok": False, "error": "Bad phone"}
        return {"ok": True, "spoken_response": "Got it, your callback is saved."}

    if name == "take_message":
        missing = _missing(args, "caller_name", "callback_number", "message")
        if missing:
            return {"ok": False, "error": f"Missing {missing}"}
        digits = _extract_phone_digits(args.get("callback_number", ""))
        if digits is None:
            return {"ok": False, "error": "Bad phone"}
        return {"ok": True, "spoken_response": "Message saved."}

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
    farewell_spoken = False  # latches once per call (mirrors prod FarewellDeduper)

    for user_text in user_turns:
        messages.append({"role": "user", "content": user_text})
        captured_turns.append(Turn(role="user", text=user_text))

        for _ in range(max_tool_loops):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0,
            )
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
    )
