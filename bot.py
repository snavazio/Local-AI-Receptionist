"""Local AI dental receptionist - lower temp, anti-loop, deterministic flow."""

import os
import re
import json
import datetime
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    LLMContextFrame, OutputAudioRawFrame, InputAudioRawFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
    TranscriptionFrame, StartFrame, Frame,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService

from pipecat.transports.websocket.server import (
    WebsocketServerParams, WebsocketServerTransport,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer

load_dotenv(override=True)

PRACTICE = {
    "name": "Smith Family Dental",
    "doctor": "Dr. Smith",
    "hours": "Monday through Friday, eight to five",
    "address": "one two three Main Street",
    "emergency_line": "five five five, one two three four",
}

PIPER_VOICE = os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx")
LOG_DIR = Path("./call_logs")
LOG_DIR.mkdir(exist_ok=True)

# Track booking state per call so we can stop calling tools after completion
class CallState:
    def __init__(self):
        self.booking_complete = False
        self.message_complete = False
        self.last_user_text = ""

CALL_STATE = CallState()


# ---------- Manual VAD ----------
class ManualEnergyVAD(FrameProcessor):
    RMS_THRESHOLD = 800.0
    START_FRAMES = 5
    STOP_FRAMES = 55

    def __init__(self):
        super().__init__()
        self._loud_count = 0
        self._silent_count = 0
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            audio = np.frombuffer(frame.audio, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
            if rms >= self.RMS_THRESHOLD:
                self._loud_count += 1
                self._silent_count = 0
                if not self._is_speaking and self._loud_count >= self.START_FRAMES:
                    self._is_speaking = True
                    logger.warning(f">>> ManualVAD: STARTED (rms={rms:.0f}) <<<")
                    await self.push_frame(VADUserStartedSpeakingFrame(), direction)
                    await self.push_frame(UserStartedSpeakingFrame(), direction)
            else:
                self._silent_count += 1
                self._loud_count = 0
                if self._is_speaking and self._silent_count >= self.STOP_FRAMES:
                    self._is_speaking = False
                    logger.warning(">>> ManualVAD: STOPPED <<<")
                    await self.push_frame(VADUserStoppedSpeakingFrame(), direction)
                    await self.push_frame(UserStoppedSpeakingFrame(), direction)
        await self.push_frame(frame, direction)


class AudioRateLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._logged = 0
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and self._logged < 3:
            logger.warning(f"AUDIO_DEBUG: rate={frame.sample_rate} bytes={len(frame.audio)}")
            self._logged += 1
        await self.push_frame(frame, direction)


class IncomingAudioLogger(FrameProcessor):
    """Logs whisper output AND captures it as last_user_text for tool gating."""
    def __init__(self):
        super().__init__()
        self._first_audio_logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and not self._first_audio_logged:
            self._first_audio_logged = True
            logger.warning(f">>> First InputAudio: rate={frame.sample_rate} bytes={len(frame.audio)} <<<")
        elif isinstance(frame, TranscriptionFrame):
            txt = (frame.text or "").strip()
            CALL_STATE.last_user_text = txt
            logger.warning(f">>> WHISPER: {txt!r} <<<")
        await self.push_frame(frame, direction)


class LlamaTokenStripper(FrameProcessor):
    LEAKED_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                     "<|begin_of_text|>", "<|end_of_text|>"]
    LEADING_ROLE = re.compile(r"^\s*assistant\b[\s:.\-]*", re.IGNORECASE)
    WRAPPING_QUOTES = re.compile(r'^\s*"(.*)"\s*$', re.DOTALL)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if hasattr(frame, "text"):
            txt = getattr(frame, "text", None)
            if isinstance(txt, str):
                cleaned = txt
                for tok in self.LEAKED_TOKENS:
                    cleaned = cleaned.replace(tok, "")
                cleaned = self.LEADING_ROLE.sub("", cleaned)
                m = self.WRAPPING_QUOTES.match(cleaned)
                if m:
                    cleaned = m.group(1)
                if cleaned != txt:
                    try:
                        setattr(frame, "text", cleaned)
                    except Exception:
                        pass
        await self.push_frame(frame, direction)


def _save_record(kind: str, data: dict) -> Path:
    fn = LOG_DIR / f"{kind}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    fn.write_text(json.dumps(data, indent=2, default=str))
    return fn


# ---------- Tool gating helpers ----------
PLACEHOLDER_VALUES = {"", "unknown", "none", "null", "n/a", "na", "tbd", "to be determined", "string"}

# Affirmative / negative caller responses that should NOT be treated as messages
NO_RESPONSES = {"no", "no thanks", "no thank you", "nope", "nah", "no thats it", "thats all",
                "that's it", "that's all", "im good", "i'm good", "all good", "okay bye",
                "ok bye", "bye", "goodbye", "thanks bye"}


def _missing(args: dict, *fields) -> list:
    out = []
    for f in fields:
        v = args.get(f)
        if v is None:
            out.append(f); continue
        if isinstance(v, str) and v.strip().lower() in PLACEHOLDER_VALUES:
            out.append(f)
    return out


def _looks_like_assistant_question(s: str) -> bool:
    """Detect when LLM tries to save its own question as the user's message."""
    if not isinstance(s, str):
        return False
    low = s.lower()
    tells = [
        "would you like", "is there anything", "can i get", "could i get",
        "do you have", "what day", "what time", "what's the best", "anything else",
    ]
    return s.strip().endswith("?") or any(t in low for t in tells)


def _is_caller_declining(s: str) -> bool:
    norm = re.sub(r"[^\w\s']", "", s.lower()).strip()
    return norm in NO_RESPONSES


async def _reject(params, missing_fields: list, friendly: str):
    msg = (
        f"VALIDATION ERROR: missing or invalid {missing_fields}. "
        f"Do NOT call this tool again until the caller has actually given you these. "
        f"Ask the caller now. Suggested: {friendly!r}"
    )
    logger.warning(f"Tool gating rejected: missing={missing_fields}")
    await params.result_callback({"ok": False, "error": msg, "spoken_response": friendly})


async def book_appointment_callback(params):
    if CALL_STATE.booking_complete:
        logger.warning("book_appointment_callback called after booking complete - ignoring")
        await params.result_callback({
            "ok": False,
            "error": "Booking is already complete for this call. Do not book again. Just say goodbye if the caller is done.",
            "spoken_response": "We've already got your callback scheduled. Anything else?",
        })
        return

    args = params.arguments or {}
    missing = _missing(args, "caller_name", "callback_number", "preferred_window")
    if missing:
        prompts = {
            "caller_name": "Could I get your name?",
            "callback_number": "What's the best phone number to call you back on?",
            "preferred_window": "What day and time works best for you?",
        }
        ask = " ".join(prompts[f] for f in missing)
        await _reject(params, missing, ask)
        return

    rec = {"ts": datetime.datetime.now().isoformat(), **args}
    fn = _save_record("callback", rec)
    CALL_STATE.booking_complete = True
    logger.info(f"Booking callback queued -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Got it. Someone from the office will call you back shortly to confirm the time. Anything else I can help with?",
    })


async def take_message(params):
    args = params.arguments or {}

    # Block 1: caller said "no" / "bye" - that's not a message to save
    if _is_caller_declining(CALL_STATE.last_user_text):
        logger.warning(f"take_message blocked: caller declined ('{CALL_STATE.last_user_text}')")
        await params.result_callback({
            "ok": False,
            "error": "The caller said no/bye, they don't want to leave a message. Do not call take_message. Just say goodbye warmly.",
            "spoken_response": "Alright, take care!",
        })
        return

    # Block 2: LLM trying to save its own prior question as the message
    msg = (args.get("message") or "").strip()
    if _looks_like_assistant_question(msg):
        logger.warning(f"take_message blocked: message looks like assistant's own question ({msg!r})")
        await params.result_callback({
            "ok": False,
            "error": "The 'message' field looks like YOUR question, not the caller's words. Use what the caller actually said, or do not call this tool.",
            "spoken_response": "What message should I pass along?",
        })
        return

    missing = _missing(args, "caller_name", "callback_number", "message")
    if missing:
        prompts = {
            "caller_name": "Could I get your name?",
            "callback_number": "What's the best callback number?",
            "message": "What message should I pass along?",
        }
        ask = " ".join(prompts[f] for f in missing)
        await _reject(params, missing, ask)
        return

    rec = {"ts": datetime.datetime.now().isoformat(), **args}
    fn = _save_record("message", rec)
    CALL_STATE.message_complete = True
    logger.info(f"Message saved -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Message saved. The office will reach out soon. Anything else?",
    })


async def escalate_emergency(params):
    args = params.arguments or {}
    rec = {"ts": datetime.datetime.now().isoformat(), **args}
    fn = _save_record("escalation", rec)
    logger.warning(f"Emergency escalation -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": f"For dental emergencies please hang up and call {PRACTICE['emergency_line']} immediately.",
    })


tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="book_appointment_callback",
        description=(
            "Save a callback request after the caller has given you their name, phone number, AND preferred day/time. "
            "Only call this once per call."
        ),
        properties={
            "caller_name": {"type": "string"},
            "callback_number": {"type": "string"},
            "preferred_window": {"type": "string"},
            "reason": {"type": "string"},
        },
        required=["caller_name", "callback_number", "preferred_window"],
    ),
    FunctionSchema(
        name="take_message",
        description=(
            "Save a message ONLY when the caller explicitly asks to leave one for the doctor or staff. "
            "Do NOT call this when the caller says 'no', 'bye', or declines. "
            "Do NOT call this for off-topic questions — answer those directly."
        ),
        properties={
            "caller_name": {"type": "string"},
            "callback_number": {"type": "string"},
            "message": {"type": "string"},
        },
        required=["caller_name", "callback_number", "message"],
    ),
    FunctionSchema(
        name="escalate_emergency",
        description=(
            "Use ONLY for medical dental emergencies the caller describes: severe pain, swelling, "
            "knocked-out tooth, bleeding, trauma. Do NOT call proactively — wait for the caller to mention symptoms."
        ),
        properties={"reason": {"type": "string"}},
        required=["reason"],
    ),
])


SYSTEM_PROMPT = f"""You are the receptionist at {PRACTICE['name']}, answering for {PRACTICE['doctor']}.

YOUR JOB: Help one caller at a time, briefly and warmly. Most turns need NO tool — just talk.

FORMAT:
- Phone call. Spoken aloud. 1-2 short sentences. No quotes, no markdown.
- Speak numbers naturally.

RULES:
- Do NOT proactively ask "is this an emergency?" — wait for the caller to bring up symptoms.
- Do NOT make up facts about the office (address, hours) unless asked. Use only the KNOWN INFO below.
- After a successful tool call, the caller's task is DONE. Ask if there's anything else, then say goodbye if not.
- When caller says "no", "bye", "thanks", "all good": say a warm goodbye. Do NOT call any tool.
- Never invent a message to save. Only save what the caller actually said.

APPOINTMENT FLOW (ask one question at a time, then book):
1. Caller wants an appointment → ask their name.
2. Got name → ask callback number.
3. Got number → ask preferred day and time.
4. Got all three → call book_appointment_callback.

KNOWN INFO (use only when asked):
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

GREETING (first turn only): "Thanks for calling {PRACTICE['name']}, how can I help?"

If asked for a human: "I'm an automated assistant, but I can take your information and have someone call you right back."
"""


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

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model="distil-large-v3"),
        device="cuda",
        compute_type="float16",
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model="receptionist-llama",
            temperature=0.1,   # was 0.4 - much more deterministic
        ),
    )
    llm.register_function("book_appointment_callback", book_appointment_callback)
    llm.register_function("take_message", take_message)
    llm.register_function("escalate_emergency", escalate_emergency)

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice="en_US-lessac-medium"),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        ManualEnergyVAD(),
        IncomingAudioLogger(),
        stt,
        context_aggregator.user(),
        llm,
        LlamaTokenStripper(),
        tts,
        AudioRateLogger(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - greeting caller")
        # Reset per-call state
        CALL_STATE.booking_complete = False
        CALL_STATE.message_complete = False
        CALL_STATE.last_user_text = ""
        context.set_messages([{"role": "system", "content": SYSTEM_PROMPT}])
        context.add_message({"role": "system", "content": "Greet the caller now."})
        await task.queue_frames([LLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        transcript = context.get_messages()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"call_{ts}.json"
        log_path.write_text(json.dumps(transcript, indent=2, default=str))
        logger.info(f"Call ended. Transcript -> {log_path}")

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
