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
    detect_correction, split_at_correction, detect_hangup,
    detect_full_redo,
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

    # ── full-redo handling: caller wants to start over completely ──
    if detect_full_redo(user_text):
        # Wipe collected slots; intent stays. The next turn will
        # re-extract from a clean slate.
        for k in ("caller_name", "callback_number", "preferred_window",
                  "day", "time_str", "message"):
            state.pop(k, None)
        # Stay at current state so we re-collect; if we were at
        # CONFIRM/END, drop back to NAME.
        if state["state"] in (S_CONFIRM, S_END):
            state["state"] = S_NAME
        # Don't return — let opportunistic extraction process this turn
        # (the user might already have given new info on the same line).

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

    # Hangup detection — caller cut the call short.
    if s != S_END and detect_hangup(user_text):
        state["aborted"] = True
        state["state"] = S_END
        return state

    # Emergency triage shortcut: if the bot's last prose asked
    # "is this an emergency?" and the caller gave a SHORT yes-answer,
    # escalate. Bare "yes"/"yeah" (≤3 words) → emergency. A long
    # response starting with "yes" probably continues with intent
    # ("yes, I'd like to book") — treat as not-emergency.
    if s != S_END and state.get("_asked_emergency"):
        ut = user_text.strip().rstrip(".,;:!?")
        word_count = len(ut.split())
        yn = extract_yes_no(user_text)
        is_short_yes = (yn == "yes" and word_count <= 3)
        # Or: long answer that contains an emergency keyword
        has_emergency_kw = classify_intent(scan_text) == "emergency"
        if is_short_yes or has_emergency_kw:
            state["intent"] = "emergency"
            state["emergency_reason"] = "caller confirmed emergency"
            state["state"] = S_END
            return state
        # Anything else clears the emergency-asked flag and proceeds
        # with normal intent flow on this same turn.
        state["_asked_emergency"] = False

    # Keyword fallback: if caller explicitly describes an emergency
    # before we could ask, escalate immediately.
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

    # Message topic — capture opportunistically for message-intent calls
    # (since the simplified flow no longer routes through S_MESSAGE).
    if state.get("intent") == "message" and not state.get("message"):
        topic = extract_message_topic(scan_text, prior_turns=prior_user_turns)
        if topic:
            state["message"] = topic
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
    """Try to collect intent + name + phone + (day/time for appointments,
    message for messages). If we get stuck on the optional slots, the
    case loop will set state['_give_up'] and we fall through to
    S_CONFIRM gracefully ('I'll have someone call you back')."""
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
    # Have intent + name + phone. For appointments, try to nail down day+time.
    # If the case loop has marked us as 'give up', skip straight to confirm.
    if intent == "appointment":
        if state.get("day") and state.get("time_str"):
            state["preferred_window"] = state.get("preferred_window") or f"{state['day']} at {state['time_str']}"
            state["state"] = S_CONFIRM
            return state
        if state.get("_give_up_window"):
            # Graceful fallback: derive whatever window we can; the bot
            # will say "I'll have someone call you back" without time.
            day = state.get("day"); tm = state.get("time_str")
            if day and tm: state["preferred_window"] = f"{day} at {tm}"
            elif day:      state["preferred_window"] = day
            elif tm:       state["preferred_window"] = tm
            state["state"] = S_CONFIRM
            return state
        state["state"] = S_WINDOW
        return state
    if intent == "message":
        if state.get("message"):
            state["state"] = S_CONFIRM
            return state
        if state.get("_give_up_message"):
            state["state"] = S_CONFIRM
            return state
        state["state"] = S_MESSAGE
        return state
    state["state"] = S_CONFIRM
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
_EMERGENCY_LINE_REQUEST = re.compile(
    r"\b(emergency line|emergency number|after.?hours|number to call after)\b",
    re.IGNORECASE,
)
_HOURS_REQUEST = re.compile(
    r"\b(what.*hours|when.*open|when do you close|are you open|"
    r"open monday|open tuesday|open wednesday|open thursday|open friday|"
    r"open saturday|open sunday|business hours)\b",
    re.IGNORECASE,
)
_ADDRESS_REQUEST = re.compile(
    r"\b(your address|where are you located|where is your office|"
    r"office address|street address|how do i get there)\b",
    re.IGNORECASE,
)


def state_prompt(state: dict) -> str:
    s = state["state"]
    nm = state.get("caller_name")
    last_user = state.get("_last_user_text", "")
    # Caller asked about practice info — recite the configured value.
    if last_user and _EMERGENCY_LINE_REQUEST.search(last_user):
        return (f"The caller asked for the emergency line. Reply EXACTLY: "
                f"'For dental emergencies, please call {PRACTICE['emergency_line']} "
                f"right away.' Include the digits as words: "
                f"'{PRACTICE['emergency_line']}'.")
    if last_user and _HOURS_REQUEST.search(last_user):
        return (f"The caller asked about office hours. Reply: 'Our hours "
                f"are {PRACTICE['hours']}. Anything else?' Include the exact "
                f"phrase '{PRACTICE['hours']}'.")
    if last_user and _ADDRESS_REQUEST.search(last_user):
        return (f"The caller asked about the office address. Reply: "
                f"'We're at {PRACTICE['address']}. Anything else?'")
    # Caller explicitly asked for a human — produce the canned response
    # with the assertion-required vocabulary.
    if last_user and _HUMAN_REQUEST.search(last_user):
        return ("Reply EXACTLY: 'I'm an automated assistant, but I can "
                "have someone call you back. What's your name and number?' "
                "Use those exact words including 'automated' and "
                "'have someone call you'.")
    if s == S_TRIAGE:
        # Always start by asking explicitly if it's an emergency. This is
        # safety-critical — way more reliable than trying to detect every
        # possible emergency phrasing in keywords.
        if not state.get("_asked_emergency_once"):
            return ("Ask the caller in ONE short question: 'Is this a "
                    "dental emergency, or are you calling about something "
                    "else?' Use the word 'emergency' verbatim.")
        return ("Ask the caller what they're calling about — booking an "
                "appointment, leaving a message, or something else?")
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
        nm_part = f", {nm}" if nm else ""
        win = state.get("preferred_window")
        intent = state.get("intent")
        if intent == "appointment" and win:
            return (f"Say: 'Thanks{nm_part}, I have your appointment "
                    f"request for {win}. Someone will call you back to "
                    f"confirm. Anything else?'")
        if intent == "message":
            return (f"Say: 'Thanks{nm_part}, I'll pass that along and "
                    f"have someone call you back. Anything else?'")
        # Graceful fallback — we have name+phone but couldn't lock down
        # day/time. Promise a callback without committing to a time.
        return (f"Say: 'Thanks{nm_part}, I'll have someone call you back "
                f"at the number you gave. Anything else I can help with?'")
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
    # Aborted calls (caller hung up mid-flow) — don't fabricate a booking.
    if state.get("aborted"):
        return None
    intent = state.get("intent")
    if intent == "emergency":
        return ToolCallRecord(name="escalate_emergency",
                              args={"reason": state.get("emergency_reason", "unspecified")},
                              result={"ok": True})
    # Simplified: only intent + name + phone are required. Window/message
    # are best-effort metadata.
    if intent == "appointment" and all(k in state for k in ("caller_name", "callback_number")):
        win = state.get("preferred_window")
        if not win:
            day = state.get("day"); tm = state.get("time_str")
            if day and tm: win = f"{day} at {tm}"
            elif day:      win = day
            elif tm:       win = tm
            else:          win = "callback requested — staff will follow up"
        return ToolCallRecord(name="save_request",
            args={"kind": "appointment", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"],
                  "preferred_window": win},
            result={"ok": True, "kind": "appointment"})
    if intent == "message" and all(k in state for k in ("caller_name", "callback_number")):
        msg = state.get("message") or "callback requested — staff will follow up"
        return ToolCallRecord(name="save_request",
            args={"kind": "message", "caller_name": state["caller_name"],
                  "callback_number": state["callback_number"], "message": msg},
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
