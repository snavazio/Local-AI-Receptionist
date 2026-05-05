"""LLM tool definitions and handlers for the AI Receptionist.

Three tools are registered with the LLM:

``save_callback_request``
    Persists an appointment callback request (name, phone, preferred day and
    time) to ``call_logs/callback_<timestamp>.json``.  Returns a spoken
    confirmation that ``ForcedSpeechOverride`` delivers via TTS.

``save_message``
    Persists a free-form message for the dentist / office staff to
    ``call_logs/message_<timestamp>.json``.

``escalate_emergency``
    Records a dental-emergency event and instructs the caller to ring the
    emergency line directly.

Every tool also appends an entry to the call's running transcript stored on
``CallState``.

Call state
----------
All handlers receive the shared ``CallState`` instance via
``FunctionCallParams.tool_resources``.  They set
``call_state.forced_speech_text`` when they want ``ForcedSpeechOverride`` to
speak a specific string rather than letting the LLM paraphrase.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CALL_LOGS_DIR = Path(__file__).parent / "call_logs"
CALL_LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CallState — shared mutable state for a single call
# ---------------------------------------------------------------------------


class CallState:
    """Mutable state shared between the bot pipeline and tool handlers.

    Attributes:
        call_id: Unique identifier for this call (set at pipeline start).
        transcript: Running list of ``{"role": str, "text": str, "ts": str}``
            dicts — populated by the context observer in ``bot.py``.
        forced_speech_text: When set by a tool handler, ``ForcedSpeechOverride``
            will speak this text instead of the LLM's follow-up.
        call_ended: Set to True once the farewell latch fires or the pipeline
            ends — used to write the final call transcript.
        emergency_line: Phone number / string read to the caller when an
            emergency is escalated.
    """

    def __init__(self, *, call_id: str, emergency_line: str = "9-1-1") -> None:
        self.call_id: str = call_id
        self.transcript: list[dict] = []
        self.forced_speech_text: str | None = None
        self.call_ended: bool = False
        self.emergency_line: str = emergency_line


# ---------------------------------------------------------------------------
# Helper: persist a JSON file with a timestamped name
# ---------------------------------------------------------------------------


def _write_log(prefix: str, data: dict) -> Path:
    """Write *data* as JSON to ``call_logs/<prefix>_<iso-timestamp>.json``.

    Args:
        prefix: File name prefix (e.g. ``"callback"``, ``"message"``).
        data: Payload to serialise.

    Returns:
        Path of the written file.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CALL_LOGS_DIR / f"{prefix}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote call log: {path}")
    return path


# ---------------------------------------------------------------------------
# Helper: convert a digit string to individually-spoken words
# ---------------------------------------------------------------------------

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _speak_phone(phone: str) -> str:
    """Format *phone* as space-separated digit words.

    Non-digit characters (hyphens, spaces, parentheses) are stripped.

    Args:
        phone: Phone number string in any common format.

    Returns:
        E.g. "5551234567" → "five five five one two three four five six seven"
    """
    digits = "".join(ch for ch in phone if ch.isdigit())
    return " ".join(_DIGIT_WORDS[d] for d in digits)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def handle_save_callback_request(params: FunctionCallParams) -> None:
    """Handle the ``save_callback_request`` tool call.

    Saves the appointment callback details to disk and stores a spoken
    confirmation in ``call_state.forced_speech_text`` so that
    ``ForcedSpeechOverride`` can deliver it to the caller.

    Args:
        params: Pipecat function call parameters including ``arguments`` and
            ``tool_resources`` (a ``CallState`` instance).
    """
    state: CallState = params.tool_resources
    args = params.arguments

    name: str = args.get("caller_name", "")
    phone: str = args.get("callback_phone", "")
    day: str = args.get("preferred_day", "")
    time_pref: str = args.get("preferred_time", "")

    payload = {
        "type": "callback",
        "call_id": state.call_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caller_name": name,
        "callback_phone": phone,
        "preferred_day": day,
        "preferred_time": time_pref,
    }
    _write_log("callback", payload)

    spoken_phone = _speak_phone(phone)
    confirmation = (
        f"Got it, {name}. I've saved your callback request. "
        f"We'll call you at {spoken_phone} on {day} {time_pref} "
        f"to confirm your appointment. Is there anything else I can help you with?"
    )
    state.forced_speech_text = confirmation

    await params.result_callback(
        f"Callback request saved for {name}. Phone: {phone}. Day: {day}. Time: {time_pref}."
    )


async def handle_save_message(params: FunctionCallParams) -> None:
    """Handle the ``save_message`` tool call.

    Saves a free-form message for the dental office and stores a spoken
    confirmation in ``call_state.forced_speech_text``.

    Args:
        params: Pipecat function call parameters.
    """
    state: CallState = params.tool_resources
    args = params.arguments

    name: str = args.get("caller_name", "")
    message: str = args.get("message", "")

    payload = {
        "type": "message",
        "call_id": state.call_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caller_name": name,
        "message": message,
    }
    _write_log("message", payload)

    confirmation = (
        f"Thank you, {name}. I've passed your message on to the team. "
        f"Someone will follow up with you soon. Is there anything else I can help you with?"
    )
    state.forced_speech_text = confirmation

    await params.result_callback(f"Message saved for {name}.")


async def handle_escalate_emergency(params: FunctionCallParams) -> None:
    """Handle the ``escalate_emergency`` tool call.

    Records the emergency details, writes a log entry, and directs the caller
    to the emergency line immediately.

    Args:
        params: Pipecat function call parameters.
    """
    state: CallState = params.tool_resources
    args = params.arguments

    name: str = args.get("caller_name", "")
    situation: str = args.get("situation", "")

    payload = {
        "type": "emergency",
        "call_id": state.call_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caller_name": name,
        "situation": situation,
    }
    _write_log("emergency", payload)
    logger.warning(f"DENTAL EMERGENCY — {name}: {situation}")

    emergency_line = state.emergency_line
    confirmation = (
        f"This sounds like a dental emergency. Please call our emergency line at "
        f"{emergency_line} right now. "
        f"I've also logged your details so the team is aware. "
        f"Please call {emergency_line} immediately. I hope you feel better soon."
    )
    state.forced_speech_text = confirmation

    await params.result_callback(
        f"Emergency escalated for {name}. Situation: {situation}. "
        f"Caller directed to emergency line {emergency_line}."
    )


# ---------------------------------------------------------------------------
# Tool schema definitions (ToolsSchema used by LLMContext)
# ---------------------------------------------------------------------------


TOOLS_SCHEMA = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="save_callback_request",
            description=(
                "Save an appointment callback request. "
                "Call this once you have the caller's name, callback phone number, "
                "preferred day, and preferred time."
            ),
            properties={
                "caller_name": {
                    "type": "string",
                    "description": "Full name of the caller.",
                },
                "callback_phone": {
                    "type": "string",
                    "description": "Callback phone number provided by the caller.",
                },
                "preferred_day": {
                    "type": "string",
                    "description": "Preferred day for the appointment (e.g. 'Monday', 'next Tuesday').",
                },
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred time or period (e.g. 'morning', '2 PM', 'afternoon').",
                },
            },
            required=["caller_name", "callback_phone", "preferred_day", "preferred_time"],
        ),
        FunctionSchema(
            name="save_message",
            description=(
                "Save a message for the dentist or office staff. "
                "Call this when the caller wants to leave a message rather than request a callback."
            ),
            properties={
                "caller_name": {
                    "type": "string",
                    "description": "Full name of the caller.",
                },
                "message": {
                    "type": "string",
                    "description": "The message the caller wants to leave.",
                },
            },
            required=["caller_name", "message"],
        ),
        FunctionSchema(
            name="escalate_emergency",
            description=(
                "Escalate a dental emergency. "
                "Call this immediately if the caller describes severe tooth pain, swelling, "
                "uncontrolled bleeding, a knocked-out tooth, jaw injury, or any trauma to the mouth."
            ),
            properties={
                "caller_name": {
                    "type": "string",
                    "description": "Name of the caller (ask if not already known).",
                },
                "situation": {
                    "type": "string",
                    "description": "Brief description of the emergency situation.",
                },
            },
            required=["caller_name", "situation"],
        ),
    ]
)
