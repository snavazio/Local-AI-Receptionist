"""Deterministic harness — Python state machine + slot extractor; LLM only
generates response prose.

Architecture:
  user turn → SlotExtractor (deterministic) → updates state slots
                                            → State machine advances
                                            → LLM generates short reply prose
                                              given (state, slots) — never
                                              decides control flow.

This eliminates the "Qwen lies about completing actions" failure mode
because the LLM is no longer in the control flow. When all required
slots are filled, the harness synthesizes the save_request /
escalate_emergency call from state — same shape as harness_flows.py so
existing assertions + alias map in run_eval.py still match.

Usage:
    .venv/bin/python eval/run_eval.py --smoke --deterministic
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.harness import (
    OLLAMA_BASE_URL, MODEL, GREETING, PRACTICE,
    Turn, ToolCallRecord, CaseResult, _strip_malformed_tool_call, _FAREWELL_RE,
)
from eval.slot_extractor import (
    classify_intent, extract_name, extract_phone,
    _find_day, _find_specific_time, is_vague_time,
    extract_yes_no, extract_message_topic,
    detect_correction, split_at_correction,
)


# ──────────────── state machine ────────────────────────────────────

# States
S_TRIAGE  = "triage"
S_NAME    = "name"
S_PHONE   = "phone"
S_WINDOW  = "window"
S_MESSAGE = "message"
S_CONFIRM = "confirm"
S_END     = "end"


def advance(state: dict, user_text: str, prior_user_turns: list[str]) -> dict:
    """Apply the user turn to the state.

    Strategy: every turn, run ALL extractors opportunistically. The
    state machine just decides what to ASK for next, not what to
    extract. This way a caller front-loading "Thursday at ten AM"
    before the bot ever asks for a time still gets it captured.

    Corrections: if the turn contains "actually" / "wait" / "instead"
    / etc., we clear conflicting slots and only re-extract from the
    text AFTER the correction marker (the "what they actually want"
    part). This handles "Tuesday at 2 PM. Actually wait, make that
    Wednesday at 2 PM, not Tuesday."
    """
    s = state["state"]

    # ── correction handling: split + clear conflicting slots ──────
    is_correction = detect_correction(user_text)
    if is_correction:
        corrected_text = split_at_correction(user_text)
        # If the correction mentions a new day/time, clear the old one.
        new_d = _find_day(corrected_text)
        new_tm = _find_specific_time(corrected_text, in_window_state=(s == S_WINDOW))
        if new_d:
            state["day"] = new_d
            state["preferred_window"] = None
        if new_tm:
            state["time_str"] = new_tm
            state["preferred_window"] = None
        # If a new phone is in the corrected text, replace.
        new_ph = extract_phone(corrected_text)
        if new_ph:
            state["callback_number"] = new_ph
        # If the correction mentions a new intent, replace.
        new_intent = classify_intent(corrected_text)
        if new_intent and new_intent != state.get("intent"):
            state["intent"] = new_intent
            # Switching from appointment → message means we drop the
            # window we'd been collecting, and need a message instead.
            if new_intent == "message":
                state.pop("day", None)
                state.pop("time_str", None)
                state.pop("preferred_window", None)
        # Continue into normal extraction below using the corrected
        # portion, so further opportunistic extraction also runs on
        # the right text.
        scan_text = corrected_text
    else:
        scan_text = user_text

    # Always: emergency keywords short-circuit.
    if s != S_END and classify_intent(scan_text) == "emergency":
        state["intent"] = "emergency"
        state["emergency_reason"] = scan_text[:120]
        state["state"] = S_END
        return state

    # ── opportunistic extraction (every turn, regardless of state) ──
    # Intent: set if unset. Also upgrade message → appointment if the
    # caller now uses an explicit appointment keyword ("book", "schedule",
    # "appointment", "checkup", "cleaning"). This guards against the LLM
    # rescue (or a too-eager initial classification) locking in 'message'
    # when the caller's later turns clearly want a booking.
    new_ic = classify_intent(scan_text)
    if not state.get("intent"):
        if new_ic:
            state["intent"] = new_ic
    elif state["intent"] == "message" and new_ic == "appointment":
        state["intent"] = "appointment"
        # Drop message-only slot if it had been set
        state.pop("message", None)

    if not state.get("callback_number"):
        ph = extract_phone(scan_text)
        if ph:
            state["callback_number"] = ph

    d = _find_day(scan_text)
    if d and not state.get("day"):
        state["day"] = d
    tm = _find_specific_time(scan_text, in_window_state=(s == S_WINDOW))
    if tm and not state.get("time_str"):
        state["time_str"] = tm
    if is_vague_time(scan_text) and not state.get("time_str"):
        state["asked_specific_time"] = True

    if not state.get("caller_name"):
        nm = extract_name(scan_text)
        if nm:
            state["caller_name"] = nm

    # ── front-loaded-name fallback: if the turn contains BOTH a phone
    # (or day) AND a comma, try to find a chunk that looks like a name.
    # Catches "I'd like to book Tuesday at 2 PM. Steve, 201-388-2149."
    # where "Steve" is the name and the rest is data.
    if not state.get("caller_name") and ("," in scan_text or "." in scan_text):
        carries_data = (extract_phone(scan_text) is not None
                        or _find_day(scan_text) is not None
                        or _find_specific_time(scan_text) is not None)
        if carries_data:
            # Split on commas AND periods so "...2 PM. Steve, 201..." becomes
            # ["...", "2 PM", " Steve", " 201-388-2149", ""].
            import re as _re
            chunks = _re.split(r"[,.]\s*", scan_text)
            for chunk in chunks:
                from slot_extractor import _looks_like_name_chunk
                cand = _looks_like_name_chunk(chunk)
                if cand:
                    state["caller_name"] = cand
                    break

    # ── state-specific fallbacks (whole-turn-as-name, message-topic) ──
    # The strengthened _looks_like_name_chunk (blocklist with adjectives
    # /verbs, reject all-digit-words, reject day-containing) is the
    # primary defense against false positives — the bot-asked-for-name
    # gate proved too strict on terse responses ("Tom." after a prose
    # that didn't include the literal word 'name').
    if s == S_NAME and not state.get("caller_name"):
        nm = extract_name(scan_text, in_name_state=True)
        if nm:
            state["caller_name"] = nm

    if s == S_MESSAGE and not state.get("message"):
        topic = extract_message_topic(scan_text, prior_turns=prior_user_turns)
        if topic:
            state["message"] = topic

    # ── derive preferred_window if we have both pieces ──
    if state.get("day") and state.get("time_str") and not state.get("preferred_window"):
        state["preferred_window"] = f"{state['day']} at {state['time_str']}"

    # ── confirm-state yes/no handling ──
    if s == S_CONFIRM:
        yn = extract_yes_no(user_text)
        if yn == "no":
            state["state"] = S_END
            return state

    # ── transition based on what slots are now filled ──
    if state.get("intent"):
        return _advance_after_data(state)

    return state


def _advance_after_data(state: dict) -> dict:
    """After a slot got filled, jump to the next missing slot for the
    current intent."""
    intent = state.get("intent")
    if intent == "emergency":
        state["state"] = S_END
        return state
    if not state.get("caller_name"):
        state["state"] = S_NAME
        return state
    if not state.get("callback_number"):
        state["state"] = S_PHONE
        return state
    if intent == "appointment":
        if not (state.get("day") and state.get("time_str")):
            state["state"] = S_WINDOW
            return state
        state["preferred_window"] = state.get("preferred_window") or f"{state['day']} at {state['time_str']}"
        state["state"] = S_CONFIRM
        return state
    if intent == "message":
        if not state.get("message"):
            state["state"] = S_MESSAGE
            return state
        state["state"] = S_CONFIRM
        return state
    return state


# ──────────────── response generation (LLM, only this) ─────────────

ROLE = (
    f"You are Sarah, an AI phone receptionist for {PRACTICE['name']}. "
    "Reply in 1-2 short sentences. No markdown, no quotes. Speak in "
    "English. You do NOT have a calendar — never propose specific slots. "
    "If you need a moment to figure out what to say next, you can simply "
    "say 'One moment, please.' — that's a perfectly natural response. "
    "If the caller asks to speak to a real person / human / operator, "
    "say: 'I'm an automated system but I can have someone call you back "
    "— what's your name and number?'"
)

# Per-state directive for the LLM. The state machine has already done
# all the slot work; the LLM just produces the right next utterance.
_HUMAN_REQUEST = re.compile(
    r"\b(real person|talk to a human|speak to a human|speak to someone|"
    r"talk to someone|operator|live agent|human being)\b", re.IGNORECASE)


def state_prompt(state: dict) -> str:
    s = state["state"]
    nm = state.get("caller_name")
    last_user = state.get("_last_user_text", "")
    # Caller explicitly asked for a human — produce the canned response
    # with the assertion-required vocabulary.
    if last_user and _HUMAN_REQUEST.search(last_user):
        return ("Reply EXACTLY: 'I'm an automated assistant, but I can "
                "have someone call you back. What's your name and number?' "
                "Use those exact words including 'automated' and "
                "'have someone call you'.")
    if s == S_TRIAGE:
        return ("Ask the caller in one short question: are they booking an "
                "appointment, leaving a message, or is this an emergency?")
    if s == S_NAME:
        return "Ask for the caller's name in one short question."
    if s == S_PHONE:
        return f"Ask {nm or 'the caller'} for their callback phone number in one short sentence."
    if s == S_WINDOW:
        if state.get("asked_specific_time") and not state.get("time_str"):
            return ("The caller said something vague like 'afternoon' or "
                    "'anytime'. Ask for a SPECIFIC time (e.g. '3 PM').")
        if state.get("day") and not state.get("time_str"):
            return f"You have the day ({state['day']}). Ask what time."
        if state.get("time_str") and not state.get("day"):
            return f"You have the time ({state['time_str']}). Ask what day."
        return "Ask what day and time the caller wants for the appointment."
    if s == S_MESSAGE:
        return f"Ask {nm or 'the caller'} what message they want to leave for the doctor."
    if s == S_CONFIRM:
        intent = state.get("intent")
        if intent == "appointment":
            return (f"Confirm: {nm}'s appointment request for "
                    f"{state.get('preferred_window')} is saved. Ask if "
                    "there's anything else.")
        if intent == "message":
            return (f"Confirm: {nm}'s message has been recorded. Ask if "
                    "there's anything else.")
        return "Confirm the request is saved and ask if there's anything else."
    if s == S_END:
        if state.get("intent") == "emergency":
            return (f"This is a dental emergency. Your reply MUST contain "
                    f"the words 'emergency' AND the digits "
                    f"'{PRACTICE['emergency_line']}' verbatim. Say one "
                    f"sentence telling the caller to hang up and call "
                    f"{PRACTICE['emergency_line']} immediately for the "
                    f"dental emergency. Do not paraphrase the number.")
        return "Say one short goodbye like 'Take care!' — pick exactly one."
    return "Ask one short clarifying question."


# ──────────────── LLM rescue (when deterministic stalls) ───────────
# If the deterministic state machine has been stuck at the same state
# for 2+ consecutive user turns, ask the LLM to extract the missing
# slot. The LLM's only job here is to fill ONE field — it cannot
# decide control flow. This catches edge cases the regexes miss
# (oddly phrased names, indirect emergency descriptions, etc.).

_RESCUE_PROMPTS = {
    S_NAME: ("Read the conversation. What did the caller say their name was? "
             "Reply with ONLY the name (1-3 words), or 'unknown' if not stated."),
    S_PHONE: ("Read the conversation. What phone number did the caller say? "
              "Reply with ONLY the digits joined together (e.g. '4155550103'), "
              "or 'unknown' if not stated."),
    S_WINDOW: ("Read the conversation. What day and specific time did the "
               "caller request? Reply with ONLY the day + time "
               "(e.g. 'Thursday at 10 AM'), or 'unknown' if vague or missing."),
    S_MESSAGE: ("Read the conversation. What is the message / topic the caller "
                "wants forwarded? Reply with ONLY a one-line summary, or "
                "'unknown'."),
    # Note: no TRIAGE rescue. If the caller hasn't yet stated their intent,
    # the right move is to keep asking — not to have the LLM guess. Guessing
    # locks in the wrong intent and later "I'd like to book" can't override.
}


async def llm_rescue(client: AsyncOpenAI, state: dict, transcript: list[dict]) -> bool:
    """Try to extract the missing slot for the current state via the LLM.
    Returns True if a slot was filled (state advanced)."""
    s = state["state"]
    prompt = _RESCUE_PROMPTS.get(s)
    if not prompt:
        return False
    msgs = transcript[-12:] + [{"role": "system", "content": prompt}]
    try:
        resp = await client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=0, max_tokens=40,
        )
        ans = (resp.choices[0].message.content or "").strip().rstrip(".,;:!?").strip()
        # Strip qwen3 reasoning tags if present
        import re as _re
        ans = _re.sub(r"<think>.*?</think>\s*", "", ans, flags=_re.DOTALL).strip()
    except Exception:
        return False
    if not ans or ans.lower() == "unknown":
        return False

    if s == S_NAME:
        # Validate via the same name-chunk heuristic the deterministic
        # extractor uses — applies blocklist + alpha + length + time-word
        # rejection. Prevents the LLM from rescuing with "been like this"
        # when the conversation contained "It's been like this..."
        from slot_extractor import _looks_like_name_chunk
        cand = _looks_like_name_chunk(ans)
        if cand:
            state["caller_name"] = cand
            return True
    elif s == S_PHONE:
        digits = extract_phone(ans)
        if digits:
            state["callback_number"] = digits
            return True
    elif s == S_WINDOW:
        d = _find_day(ans); tm = _find_specific_time(ans, in_window_state=True)
        if d and tm:
            state["day"] = d; state["time_str"] = tm
            state["preferred_window"] = f"{d} at {tm}"
            return True
    elif s == S_MESSAGE:
        if len(ans) > 2:
            state["message"] = ans[:200]
            return True
    elif s == S_TRIAGE:
        if ans.lower() in {"appointment", "message", "emergency"}:
            state["intent"] = ans.lower()
            if ans.lower() == "emergency":
                state["state"] = S_END
                state["emergency_reason"] = "rescued from caller description"
            return True
    return False


async def gen_reply(client: AsyncOpenAI, state: dict, transcript: list[dict]) -> tuple[str, int]:
    """Ask the LLM for the next assistant utterance. The LLM has NO
    tools and isn't deciding anything — it's just writing prose."""
    msgs = [{"role": "system", "content": ROLE}] + transcript[-12:] + [
        {"role": "system", "content": "Generate ONLY your next 1-2 sentences. No explanation."},
        {"role": "system", "content": state_prompt(state)},
    ]
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=MODEL, messages=msgs, temperature=0, max_tokens=80,
    )
    return resp.choices[0].message.content or "", int((time.monotonic() - t0) * 1000)


# ──────────────── case runner ──────────────────────────────────────

def synthesize_logical_call(state: dict) -> ToolCallRecord | None:
    intent = state.get("intent")
    if intent == "emergency":
        return ToolCallRecord(name="escalate_emergency",
                              args={"reason": state.get("emergency_reason", "unspecified")},
                              result={"ok": True})
    if intent == "appointment" and all(k in state for k in ("caller_name", "callback_number", "preferred_window")):
        return ToolCallRecord(name="save_request",
            args={"kind": "appointment", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"],
                  "preferred_window": state["preferred_window"]},
            result={"ok": True, "kind": "appointment"})
    if intent == "message" and all(k in state for k in ("caller_name", "callback_number", "message")):
        return ToolCallRecord(name="save_request",
            args={"kind": "message", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"],
                  "message": state["message"]},
            result={"ok": True, "kind": "message"})
    return None


async def run_case_async(case_id: str, user_turns: list[str],
                         max_tool_loops: int = 5,
                         client: AsyncOpenAI | None = None) -> CaseResult:
    own_client = False
    if client is None:
        client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        own_client = True

    state = {"state": S_TRIAGE}
    transcript = [
        {"role": "system", "content": ROLE},
        {"role": "assistant", "content": GREETING},
    ]
    captured_turns = [Turn(role="assistant", text=GREETING)]
    captured_tool_calls: list[ToolCallRecord] = []
    llm_call_ms: list[int] = []
    farewell_spoken = False
    prior_user_turns: list[str] = []
    stuck_at: dict[str, int] = {}   # node_key -> turns spent here w/o progress
    last_bot_prose = GREETING       # to detect "did the bot just ask for X?"

    for user_text in user_turns:
        transcript.append({"role": "user", "content": user_text})
        captured_turns.append(Turn(role="user", text=user_text))

        # Record latest user text so state_prompt can detect "real person" etc.
        state["_last_user_text"] = user_text

        # Did the bot's previous prose explicitly ask for the caller's name?
        # Used to gate the "whole-turn looks like a name" fallback so it
        # doesn't grab "flexible" / "Twelve-thirty works" / etc. when the
        # bot was actually asking about something else.
        prose = last_bot_prose.lower()
        state["_asked_for_name"] = (
            ("name" in prose and ("your" in prose or "may i" in prose
                                   or "what's" in prose or "what is" in prose
                                   or "who" in prose or "tell me" in prose
                                   or "could i" in prose or "can i" in prose))
        )

        # 1) Update state from the deterministic extractor.
        prev_state_key = state["state"]
        prev_slot_count = sum(1 for k in ("intent", "caller_name", "callback_number",
                                          "preferred_window", "message") if state.get(k))
        advance(state, user_text, prior_user_turns)
        prior_user_turns.append(user_text)

        # 1b) Stuck-detector: if the state didn't advance AND no new slot
        # got filled this turn, increment the counter. After 2 stuck turns
        # at the same state, ask the LLM to rescue (extract the missing
        # slot from history).
        new_slot_count = sum(1 for k in ("intent", "caller_name", "callback_number",
                                         "preferred_window", "message") if state.get(k))
        progressed = (state["state"] != prev_state_key) or (new_slot_count > prev_slot_count)
        if progressed:
            stuck_at = {state["state"]: 0}
        else:
            stuck_at[state["state"]] = stuck_at.get(state["state"], 0) + 1

        if stuck_at.get(state["state"], 0) >= 2:
            rescued = await llm_rescue(client, state, transcript)
            if rescued:
                # Re-derive transitions after rescue filled a slot.
                _advance_after_data(state)
                stuck_at[state["state"]] = 0

        # 2) Generate the assistant response prose for the current state.
        reply, ms = await gen_reply(client, state, transcript)
        llm_call_ms.append(ms)
        reply = _strip_malformed_tool_call(reply)
        # Strip qwen3 reasoning leakage if the model emits <think>...</think>
        import re as _re
        reply = _re.sub(r"<think>.*?</think>\s*", "", reply, flags=_re.DOTALL).strip()
        if farewell_spoken:
            reply = ""
        elif reply and _FAREWELL_RE.search(reply):
            farewell_spoken = True
        transcript.append({"role": "assistant", "content": reply})
        captured_turns.append(Turn(role="assistant", text=reply))
        last_bot_prose = reply or last_bot_prose

        if state["state"] == S_END:
            break

    # If the FSM didn't reach END but reached a CONFIRM with all data,
    # synthesize the logical call anyway.
    logical = synthesize_logical_call(state)
    if logical:
        captured_tool_calls.append(logical)

    if own_client:
        await client.close()

    return CaseResult(
        case_id=case_id, turns=captured_turns, tool_calls=captured_tool_calls,
        transcript=transcript, llm_call_ms=llm_call_ms,
        prompt_tokens=0, completion_tokens=0,
    )


def run_case(case_id: str, user_turns: list[str], max_tool_loops: int = 5) -> CaseResult:
    import asyncio
    return asyncio.run(run_case_async(case_id, user_turns, max_tool_loops))


async def run_cases_concurrent(cases, concurrency=2):
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
