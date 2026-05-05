"""FSM-driven variant of bot.py using pipecat-ai-flows.

Same audio pipeline (WhisperSTT, Qwen, Piper) and same FrameProcessors
(MalformedToolCallStripper, FarewellDeduper, TextNormalizer,
TurnLatencyTracker), but conversation structure is driven by a `FlowManager`
state machine instead of free-form LLM tool-calling.

Why this exists: in the eval harness, the message-taking flow stuck at
4/10 even after prompt fixes and tool-merging — the LLM kept gathering
slots correctly then refusing to invoke the tool. With FlowManager, the
function call IS the state transition, so it cannot be skipped.

Run: `python bot_flows.py` (drop-in replacement for bot.py).
"""

import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from pipecat_flows import FlowManager, NodeConfig, flows_direct_function

# Reuse all the FrameProcessors and shared helpers from the original bot.
from bot import (
    BiasedWhisperSTT,
    FarewellDeduper,
    ForcedSpeechOverride,
    IncomingAudioLogger,
    MalformedToolCallStripper,
    ManualEnergyVAD,
    AudioRateLogger,
    TextNormalizer,
    TurnLatencyTracker,
    PRACTICE,
    LOG_DIR,
    extract_phone_digits,
    _looks_like_garbage_name,
)

load_dotenv(override=True)


GREETING = (
    "Thanks for calling Smith Family Dental. "
    "This is Sarah, the AI assistant. How can I help you today?"
)


# ---------- shared role context ----------
ROLE_MESSAGE = (
    f"You are Sarah, an AI phone receptionist for {PRACTICE['name']} (answering "
    f"for {PRACTICE['doctor']}). You speak in 1-2 short sentences per turn. "
    f"This is a phone call — no markdown, no quotes, speak numbers naturally. "
    f"Phone numbers MUST be spelled as words ('two zero one, three eight eight, "
    f"two one four nine'), never with dashes. You do NOT have access to the "
    f"office calendar; you only collect a *requested* day/time as a callback. "
    f"Office hours: {PRACTICE['hours']}. Address: {PRACTICE['address']}. "
    f"Emergency line: {PRACTICE['emergency_line']}."
)


# ---------- flow handlers (each is also a transition) ----------
@flows_direct_function()
async def set_intent(flow_manager: FlowManager, intent: str):
    """Record what the caller wants. intent must be one of:
    'appointment' (book a visit), 'message' (leave a note for the doctor),
    or 'emergency' (severe pain/swelling/bleeding/trauma)."""
    intent = (intent or "").strip().lower()
    if intent not in {"appointment", "message", "emergency"}:
        return {"status": "unclear"}, triage_node()
    flow_manager.state["intent"] = intent
    if intent == "emergency":
        return {"status": "ok"}, emergency_node()
    return {"status": "ok"}, collect_name_node()


@flows_direct_function()
async def set_name(flow_manager: FlowManager, caller_name: str):
    """Record the caller's name as they spoke it. Never invent or guess —
    if you didn't actually hear a name, do not call this."""
    caller_name = (caller_name or "").strip()
    if not caller_name or len(caller_name) < 2 or _looks_like_garbage_name(caller_name):
        return {"status": "bad_name"}, collect_name_node()
    flow_manager.state["caller_name"] = caller_name
    return {"status": "ok"}, collect_phone_node()


@flows_direct_function()
async def set_phone(flow_manager: FlowManager, callback_number: str):
    """Record the caller's callback phone number. Pass exactly what the caller
    spoke — either digit form or spelled-out word form is fine; do not invent
    digits the caller did not say."""
    digits = extract_phone_digits(callback_number or "")
    if digits is None or len(digits) < 7:
        return {"status": "bad_phone"}, collect_phone_node()
    flow_manager.state["callback_number"] = digits
    intent = flow_manager.state.get("intent")
    if intent == "appointment":
        return {"status": "ok"}, collect_window_node()
    return {"status": "ok"}, collect_message_node()


@flows_direct_function()
async def set_window(flow_manager: FlowManager, preferred_window: str):
    """Record the day/time the caller asked for. Must be a specific day
    (Mon-Fri) plus a time of day. If the caller said something vague like
    'afternoon', do not call this — keep asking until you have a real time."""
    preferred_window = (preferred_window or "").strip()
    if not preferred_window or len(preferred_window) < 4:
        return {"status": "vague"}, collect_window_node()
    flow_manager.state["preferred_window"] = preferred_window
    return _save_and_advance(flow_manager, kind="appointment")


@flows_direct_function()
async def set_message(flow_manager: FlowManager, message: str):
    """Record the message the caller wants to leave for the doctor."""
    message = (message or "").strip()
    if not message:
        return {"status": "missing"}, collect_message_node()
    flow_manager.state["message"] = message
    return _save_and_advance(flow_manager, kind="message")


@flows_direct_function()
async def acknowledge_emergency(flow_manager: FlowManager, reason: str):
    """Acknowledge a dental emergency and direct the caller to the
    emergency line. Call this once the caller has described severe pain,
    swelling, bleeding, knocked-out tooth, or trauma."""
    reason = (reason or "unspecified").strip()
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "kind": "emergency",
        "reason": reason,
        **{k: v for k, v in flow_manager.state.items() if k != "intent"},
    }
    fn = LOG_DIR / f"escalation_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    fn.write_text(json.dumps(rec, indent=2, default=str))
    logger.warning(f"Emergency escalation -> {fn}")
    return {"status": "ok"}, end_node()


def _save_and_advance(flow_manager: FlowManager, kind: str):
    """Side effect: persist the request to call_logs/ and move to the
    confirmation node."""
    rec = {
        "ts": datetime.datetime.now().isoformat(),
        "kind": kind,
        **{k: v for k, v in flow_manager.state.items() if k != "intent"},
    }
    name = "callback" if kind == "appointment" else "message"
    fn = LOG_DIR / f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    fn.write_text(json.dumps(rec, indent=2, default=str))
    logger.info(f"{kind} saved -> {fn}")
    return {"status": "ok"}, confirm_node()


# ---------- node factories ----------
def triage_node() -> NodeConfig:
    return {
        "name": "triage",
        "role_messages": [{"role": "system", "content": ROLE_MESSAGE}],
        "task_messages": [{"role": "system", "content": (
            "Listen to what the caller wants. As soon as you can tell, call set_intent "
            "with one of: 'appointment' (book a visit), 'message' (leave a note for "
            "the doctor), or 'emergency' (severe pain, swelling, bleeding, trauma). "
            "If unclear, ask one short clarifying question — don't guess."
        )}],
        "functions": [set_intent],
    }


def collect_name_node() -> NodeConfig:
    return {
        "name": "collect_name",
        "task_messages": [{"role": "system", "content": (
            "Ask for the caller's name in one short sentence. When they answer, "
            "call set_name with their actual name. Never invent or use placeholders "
            "like 'John Doe'. If you didn't hear a name, ask again."
        )}],
        "functions": [set_name],
    }


def collect_phone_node() -> NodeConfig:
    return {
        "name": "collect_phone",
        "task_messages": [{"role": "system", "content": (
            "Ask for the caller's callback phone number in one short sentence. "
            "When they say it, repeat the digits back as words ('two zero one, "
            "three eight eight, two one four nine'). Then call set_phone with the "
            "number you heard."
        )}],
        "functions": [set_phone],
    }


def collect_window_node() -> NodeConfig:
    return {
        "name": "collect_window",
        "task_messages": [{"role": "system", "content": (
            "Ask what day and time they would prefer for the appointment. You do "
            "NOT have access to the calendar — never propose specific slots. When "
            "they give you a real day + time (e.g. 'Tuesday at 2 PM'), call "
            "set_window. If they say something vague like 'afternoon', ask for a "
            "specific time before calling."
        )}],
        "functions": [set_window],
    }


def collect_message_node() -> NodeConfig:
    return {
        "name": "collect_message",
        "task_messages": [{"role": "system", "content": (
            "Ask what message they want to leave for the doctor. When they tell "
            "you, call set_message with their words. Do not paraphrase or summarize."
        )}],
        "functions": [set_message],
    }


def emergency_node() -> NodeConfig:
    return {
        "name": "emergency",
        "task_messages": [{"role": "system", "content": (
            "Acknowledge the emergency briefly and direct the caller to the "
            "emergency line: five five five, one two three four. Call "
            "acknowledge_emergency with a short reason describing what they said."
        )}],
        "functions": [acknowledge_emergency],
    }


def confirm_node() -> NodeConfig:
    return {
        "name": "confirm",
        "task_messages": [{"role": "system", "content": (
            "Tell the caller their request is saved and ask if there's anything "
            "else. One short sentence."
        )}],
        "functions": [],
    }


def end_node() -> NodeConfig:
    return {
        "name": "end",
        "task_messages": [{"role": "system", "content": (
            "Say one short goodbye. 'Take care!' or 'Goodbye!' or 'Have a great "
            "day!' — pick exactly one. No 'thanks for calling' on top."
        )}],
        "functions": [],
    }


# ---------- pipeline ----------
async def main():
    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=8765,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=22050,
            add_wav_header=False,
            vad_analyzer=None,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    stt = BiasedWhisperSTT(
        settings=BiasedWhisperSTT.Settings(model="distil-large-v3"),
        device="cuda",
        compute_type="float16",
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model="qwen2.5:14b",
            temperature=0,
        ),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice="en_US-lessac-medium"),
    )

    context = LLMContext(messages=[], tools=ToolsSchema(standard_tools=[]))
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        ManualEnergyVAD(),
        IncomingAudioLogger(),
        stt,
        context_aggregator.user(),
        llm,
        ForcedSpeechOverride(),
        MalformedToolCallStripper(),
        FarewellDeduper(),
        TextNormalizer(),
        tts,
        AudioRateLogger(),
        TurnLatencyTracker(),
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=22050,
            allow_interruptions=True,
        ),
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — speaking hardcoded greeting + initializing flow")
        await task.queue_frames([TTSSpeakFrame(text=GREETING)])
        await flow_manager.initialize(triage_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended.")

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
