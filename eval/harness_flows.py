"""FSM-driven eval harness — mirrors bot_flows.py without pipecat.

Drives the LLM through a state machine where each node exposes a focused
prompt + a small set of "transition" tools. The LLM cannot save a
request until the FSM has gathered all required slots, which closes the
failure modes the free-form bot.py exhibits:

  - calling save_request with placeholders like '<CALLER_NUMBER>'
  - extracting "Sometime" as caller_name
  - never finalizing with save_request after collecting everything
  - language drift on weird inputs (FSM constrains to function calls)

At the end of each case we synthesize a single save_request /
escalate_emergency call from the accumulated state, so the existing
assertion logic + alias map in run_eval.py works unchanged.

Usage:
    .venv/bin/python eval/run_eval.py --smoke --flows
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

# Reuse what we already have in harness.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.harness import (
    OLLAMA_BASE_URL, MODEL, GREETING, PRACTICE,
    Turn, ToolCallRecord, CaseResult,
    _extract_phone_digits, _strip_malformed_tool_call, _FAREWELL_RE,
)


# ──────────────────── role + node prompts ────────────────────────

ROLE = (
    f"You are Sarah, an AI phone receptionist for {PRACTICE['name']}. "
    f"You speak in 1-2 short sentences per turn. This is a phone call — "
    f"no markdown, no quotes, speak in English. You do NOT have access "
    f"to the office calendar; you only collect a *requested* day/time as "
    f"a callback. Office hours: {PRACTICE['hours']}. "
    f"Emergency line: {PRACTICE['emergency_line']}."
)


# Each node spec: prompt, tool names exposed in this node, tool schemas.
# Tools are defined once below and indexed by name for selection per node.

def tool_schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOLS = {
    "set_intent": tool_schema(
        "set_intent",
        "Record what the caller wants. intent must be one of: 'appointment' "
        "(book a visit), 'message' (leave a note for the doctor — also use "
        "this for pricing requests, rescheduling existing appointments, or "
        "any topic this assistant can't directly handle), or 'emergency' "
        "(severe pain/swelling/bleeding/trauma).",
        {"intent": {"type": "string", "enum": ["appointment", "message", "emergency"]}},
        ["intent"],
    ),
    "set_name": tool_schema(
        "set_name",
        "Record the caller's name. Pass exactly what the caller said. Never "
        "invent or use a placeholder. If you didn't actually hear a name, "
        "do not call this — ask again instead.",
        {"caller_name": {"type": "string"}},
        ["caller_name"],
    ),
    "set_phone": tool_schema(
        "set_phone",
        "Record the caller's callback phone number. Pass exactly what the "
        "caller spoke — digits, words, or a mix all OK. Never invent digits.",
        {"callback_number": {"type": "string"}},
        ["callback_number"],
    ),
    "set_window": tool_schema(
        "set_window",
        "Record the day/time the caller wants. MUST include both a specific "
        "day (Mon-Sun) AND a specific time of day (e.g. '3 PM'). If they "
        "said something vague like 'afternoon' or 'anytime', do NOT call "
        "this — keep asking until you have a real time.",
        {"preferred_window": {"type": "string"}},
        ["preferred_window"],
    ),
    "set_message": tool_schema(
        "set_message",
        "Record the message/topic the caller wants forwarded.",
        {"message": {"type": "string"}},
        ["message"],
    ),
    "acknowledge_emergency": tool_schema(
        "acknowledge_emergency",
        "Acknowledge a dental emergency (pain/swelling/bleeding/trauma).",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
}


def node(prompt: str, *fn_names: str) -> dict:
    return {
        "prompt": prompt,
        "tools": [TOOLS[n] for n in fn_names],
        "tool_names": set(fn_names),
    }


def NODE_TRIAGE() -> dict:
    return node(
        "Listen to what the caller wants. As soon as you can tell, call "
        "set_intent with one of: 'appointment' (book a visit), 'message' "
        "(leave a note — also covers pricing questions, reschedule "
        "requests, or anything this assistant can't handle), or "
        "'emergency'. If unclear, ask one short clarifying question.",
        "set_intent",
    )


def NODE_NAME() -> dict:
    return node(
        "Ask for the caller's name in one short sentence. When they "
        "answer, call set_name with their actual name. Never invent. "
        "If you didn't hear a name, ask again.",
        "set_name",
    )


def NODE_PHONE() -> dict:
    return node(
        "Ask for the caller's callback phone number in one short "
        "sentence. When they say it, repeat the digits back as words to "
        "confirm. Then call set_phone with the number you heard.",
        "set_phone",
    )


def NODE_WINDOW() -> dict:
    return node(
        "Confirm the day and time the caller wants for the appointment. "
        "FIRST scan the conversation history above — if the caller has "
        "ALREADY said a specific day AND a specific time anywhere in "
        "this call (e.g. 'Thursday' + 'ten AM'), CALL set_window NOW "
        "with that combination. Do NOT ask again. Do NOT say 'I have "
        "your appointment set' — only set_window can save it. "
        "If only a day OR only a vague time was given, ask one short "
        "question for the missing piece. You do NOT have a calendar — "
        "never propose specific slots.",
        "set_window",
    )


def NODE_MESSAGE() -> dict:
    return node(
        "Capture the message the caller wants forwarded. FIRST scan the "
        "conversation above — if the topic is already clear from earlier "
        "(e.g. 'crown pricing', 'reschedule Tuesday to Thursday'), CALL "
        "set_message NOW with a one-line summary. Do NOT re-ask if you "
        "already have it. Otherwise, ask one short question.",
        "set_message",
    )


def NODE_EMERGENCY() -> dict:
    return node(
        "Acknowledge the emergency briefly and direct the caller to the "
        f"emergency line: {PRACTICE['emergency_line']}. Call "
        "acknowledge_emergency once you've named the issue.",
        "acknowledge_emergency",
    )


def NODE_CONFIRM() -> dict:
    return node(
        "Tell the caller their request is saved and ask if there's "
        "anything else. One short sentence.",
        # No tools — just close out.
    )


def NODE_END() -> dict:
    return node(
        "Say one short goodbye. 'Take care!' or 'Goodbye!' — pick exactly "
        "one. No 'thanks for calling' on top.",
    )


# ──────────────────── transition logic ────────────────────────────


def transition(state: dict, tool_name: str, args: dict) -> tuple[str, dict, dict]:
    """Apply a tool call to the FSM. Returns (next_node_key, validation_result, args_normalized).

    validation_result includes {"ok": bool, ...} like execute_tool, so the
    LLM gets feedback if it tried to set a vague window.
    """
    if tool_name == "set_intent":
        intent = (args.get("intent") or "").strip().lower()
        if intent == "emergency":
            state["intent"] = "emergency"
            return "emergency", {"ok": True}, args
        if intent in ("appointment", "message"):
            state["intent"] = intent
            return "name", {"ok": True}, args
        return "triage", {"ok": False, "error": "intent must be appointment/message/emergency"}, args

    if tool_name == "set_name":
        name = (args.get("caller_name") or "").strip()
        if not name or len(name) < 2 or name.lower() in {"sometime", "anytime", "tomorrow"}:
            return "name", {"ok": False, "error": "Bad or missing name — ask again"}, args
        state["caller_name"] = name
        return "phone", {"ok": True}, args

    if tool_name == "set_phone":
        digits = _extract_phone_digits(args.get("callback_number") or "")
        if digits is None or len(digits) < 7:
            return "phone", {"ok": False, "error": "Couldn't make out a phone number — ask again"}, args
        state["callback_number"] = digits
        return ("window" if state.get("intent") == "appointment" else "message"), {"ok": True}, {**args, "callback_number": digits}

    if tool_name == "set_window":
        win = (args.get("preferred_window") or "").strip()
        # Reject vague: must contain a digit OR a written-out time word
        # ('three', 'ten', 'two thirty', 'noon', 'morning' alone is too vague).
        # Heuristic: require a digit anywhere OR one of the explicit time words.
        time_words = {"noon", "midnight", "one", "two", "three", "four", "five",
                      "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}
        has_digit = any(ch.isdigit() for ch in win)
        has_time_word = any(w in win.lower().split() for w in time_words)
        if not win or len(win) < 4 or not (has_digit or has_time_word):
            return "window", {"ok": False, "error": "Need a specific time, not just a vague phrase — ask again"}, args
        state["preferred_window"] = win
        return "confirm", {"ok": True}, args

    if tool_name == "set_message":
        msg = (args.get("message") or "").strip()
        if not msg:
            return "message", {"ok": False, "error": "Empty message — ask again"}, args
        state["message"] = msg
        return "confirm", {"ok": True}, args

    if tool_name == "acknowledge_emergency":
        state["emergency_reason"] = (args.get("reason") or "unspecified").strip()
        return "end", {"ok": True}, args

    # Unknown tool — keep current node, mark error
    return state.get("_node", "triage"), {"ok": False, "error": f"Unknown tool {tool_name!r}"}, args


NODE_LOOKUP = {
    "triage": NODE_TRIAGE, "name": NODE_NAME, "phone": NODE_PHONE,
    "window": NODE_WINDOW, "message": NODE_MESSAGE, "emergency": NODE_EMERGENCY,
    "confirm": NODE_CONFIRM, "end": NODE_END,
}


def synthesize_logical_call(state: dict) -> ToolCallRecord | None:
    """At the end of the case, produce a save_request / escalate_emergency
    call from the FSM state — so existing assertions / alias mapping work."""
    intent = state.get("intent")
    if intent == "emergency":
        return ToolCallRecord(
            name="escalate_emergency",
            args={"reason": state.get("emergency_reason", "unspecified")},
            result={"ok": True},
        )
    if intent == "appointment" and all(k in state for k in ("caller_name", "callback_number", "preferred_window")):
        return ToolCallRecord(
            name="save_request",
            args={"kind": "appointment", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"], "preferred_window": state["preferred_window"]},
            result={"ok": True, "kind": "appointment"},
        )
    if intent == "message" and all(k in state for k in ("caller_name", "callback_number", "message")):
        return ToolCallRecord(
            name="save_request",
            args={"kind": "message", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"], "message": state["message"]},
            result={"ok": True, "kind": "message"},
        )
    return None


# ──────────────────── main loop ───────────────────────────────────

async def run_case_async(
    case_id: str,
    user_turns: list[str],
    max_tool_loops: int = 5,
    client: AsyncOpenAI | None = None,
) -> CaseResult:
    own_client = False
    if client is None:
        client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        own_client = True

    state: dict = {"_node": "triage"}
    node_key = "triage"

    captured_turns = [Turn(role="assistant", text=GREETING)]
    captured_tool_calls: list[ToolCallRecord] = []
    transcript: list[dict] = [
        {"role": "system", "content": ROLE},
        {"role": "system", "content": NODE_LOOKUP[node_key]()["prompt"]},
        {"role": "assistant", "content": GREETING},
    ]
    llm_call_ms: list[int] = []
    prompt_tokens = 0
    completion_tokens = 0
    farewell_spoken = False

    # Phrases that indicate the LLM is *claiming* to have completed an
    # action without actually calling the tool — common Qwen failure
    # mode. When we see these in prose-only responses, retry with
    # tool_choice="required" to force the tool call.
    LYING_PATTERNS = re.compile(
        r"\b(appointment is (now |all )?(set|booked|scheduled|confirmed)|"
        r"your appointment (is|for|has been|will be)|"
        r"i('ve| have) (recorded|noted|set|saved|booked|got|captured|noted down)|"
        r"i'?ll (have|let) someone (call|reach)|"
        r"message (is|has been) (saved|noted|recorded|forwarded)|"
        r"request (is|has been) (saved|noted)|"
        r"is now (saved|booked|scheduled|noted)|"
        r"all set|noted (down|that))\b",
        re.IGNORECASE,
    )

    async def call_llm(force_tool: bool = False):
        """One LLM round-trip. force_tool=True passes tool_choice='required'
        which makes Qwen emit a function call instead of free-form prose."""
        current = NODE_LOOKUP[node_key]()
        messages = transcript + [{"role": "system", "content": f"(current step: {node_key}) {current['prompt']}"}]
        kwargs = dict(model=MODEL, messages=messages, tools=current["tools"] or None, temperature=0)
        if force_tool and current["tools"]:
            kwargs["tool_choice"] = "required"
        t0 = time.monotonic()
        resp = await client.chat.completions.create(**kwargs)
        return resp, current, int((time.monotonic() - t0) * 1000)

    for user_text in user_turns:
        transcript.append({"role": "user", "content": user_text})
        captured_turns.append(Turn(role="user", text=user_text))

        forced_this_turn = False
        for loop_i in range(max_tool_loops):
            resp, current, ms = await call_llm(force_tool=False)
            llm_call_ms.append(ms)

            # Check if Qwen is "lying" — claiming completion in prose
            # while not calling the tool. If so, immediately retry with
            # tool_choice="required" so the FSM actually advances.
            msg_peek = resp.choices[0].message
            if (not msg_peek.tool_calls and current["tools"]
                    and msg_peek.content and not forced_this_turn
                    and LYING_PATTERNS.search(msg_peek.content)):
                resp, current, ms2 = await call_llm(force_tool=True)
                llm_call_ms.append(ms2)
                forced_this_turn = True
            if getattr(resp, "usage", None):
                prompt_tokens += resp.usage.prompt_tokens or 0
                completion_tokens += resp.usage.completion_tokens or 0

            msg = resp.choices[0].message
            assistant_msg: dict = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            transcript.append(assistant_msg)

            spoken = msg.content or ""
            if spoken:
                spoken = _strip_malformed_tool_call(spoken)
                if farewell_spoken:
                    spoken = ""
                elif _FAREWELL_RE.search(spoken):
                    farewell_spoken = True

            captured_turns.append(Turn(
                role="assistant", text=spoken,
                tool_calls=[{"name": tc.function.name, "arguments": tc.function.arguments}
                            for tc in (msg.tool_calls or [])],
            ))

            if msg.tool_calls:
                progressed = False
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    # Reject tools the current node doesn't expose.
                    if tc.function.name not in current["tool_names"]:
                        result = {"ok": False, "error": f"Tool {tc.function.name} not available in this step"}
                    else:
                        new_node, result, args = transition(state, tc.function.name, args)
                        if result.get("ok"):
                            node_key = new_node
                            state["_node"] = node_key
                            progressed = True
                    captured_tool_calls.append(ToolCallRecord(name=tc.function.name, args=args, result=result))
                    transcript.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                # Loop again if a transition happened — let the LLM speak the next prompt.
                if progressed:
                    continue
                break
            break

        # Bail early if we've reached a terminal state.
        if node_key == "end":
            break

    if own_client:
        await client.close()

    # Synthesize the logical save_request / escalate_emergency call so that
    # cases.yaml expectations against book_appointment_callback / take_message
    # / escalate_emergency match via run_eval.py's alias map.
    logical = synthesize_logical_call(state)
    if logical:
        captured_tool_calls.append(logical)

    return CaseResult(
        case_id=case_id, turns=captured_turns, tool_calls=captured_tool_calls,
        transcript=transcript, llm_call_ms=llm_call_ms,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def run_case(case_id: str, user_turns: list[str], max_tool_loops: int = 5) -> CaseResult:
    """Sync wrapper for the FSM harness."""
    import asyncio
    return asyncio.run(run_case_async(case_id, user_turns, max_tool_loops))


async def run_cases_concurrent(cases, concurrency=2):
    """Mirror harness.run_cases_concurrent."""
    import asyncio
    sem = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    results: list[CaseResult | None] = [None] * len(cases)

    async def worker(i, c):
        async with sem:
            results[i] = await run_case_async(c["id"], c["user_turns"], client=client)

    await asyncio.gather(*[worker(i, c) for i, c in enumerate(cases)])
    await client.close()
    return results
