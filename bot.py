"""Local AI dental receptionist - tightened tool gating and turn timing."""

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


# ---------- Manual VAD with longer stop window for natural turn-taking ----------
class ManualEnergyVAD(FrameProcessor):
    """Inline energy-based VAD.

    Pipecat 1.x WebsocketServerInputTransport doesn't invoke VAD analyzers, so
    we synthesize speaking frames ourselves. We emit BOTH:
      - UserStartedSpeakingFrame / UserStoppedSpeakingFrame  (for UI / RTVI clients)
      - VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame  (for SegmentedSTTService)
    """

    RMS_THRESHOLD = 800.0       # int16 RMS for "speech"
    START_FRAMES = 5            # ~100ms loud => speaking
    STOP_FRAMES = 55            # ~1.1s silence => stopped (was 30 = 600ms)

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
    def __init__(self):
        super().__init__()
        self._first_audio_logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and not self._first_audio_logged:
            self._first_audio_logged = True
            logger.warning(
                f">>> First InputAudio: rate={frame.sample_rate} channels={frame.num_channels} bytes={len(frame.audio)} <<<"
            )
        elif isinstance(frame, TranscriptionFrame):
            logger.warning(f">>> WHISPER: {frame.text!r} <<<")
        await self.push_frame(frame, direction)


class LlamaTokenStripper(FrameProcessor):
    LEAKED_TOKENS = [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<|begin_of_text|>", "<|end_of_text|>",
    ]
    LEADING_ROLE = re.compile(r"^\s*assistant\b[\s:.\-]*", re.IGNORECASE)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if hasattr(frame, "text"):
            txt = getattr(frame, "text", None)
            if isinstance(txt, str):
                cleaned = txt
                for tok in self.LEAKED_TOKENS:
                    cleaned = cleaned.replace(tok, "")
                cleaned = self.LEADING_ROLE.sub("", cleaned)
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


async def book_appointment_callback(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("callback", rec)
    logger.info(f"Booking callback queued -> {fn}")
    await params.result_callback({"ok": True, "spoken_response": "Got it. Someone will call you back shortly to confirm the time."})


async def take_message(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("message", rec)
    logger.info(f"Message saved -> {fn}")
    await params.result_callback({"ok": True, "spoken_response": "Message saved. The office will reach out soon."})


async def escalate_emergency(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("escalation", rec)
    logger.warning(f"Emergency escalation -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": f"For dental emergencies please hang up and call {PRACTICE['emergency_line']} immediately."
    })


tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="book_appointment_callback",
        description=(
            "Call this ONLY after you have collected ALL THREE required fields from the caller: "
            "their name, their callback phone number, and their preferred appointment window "
            "(e.g. 'Tuesday morning'). Do NOT call this with empty or 'unknown' values. "
            "If any field is missing, ask the caller for it instead of calling this tool."
        ),
        properties={
            "caller_name": {"type": "string", "description": "Caller's full name as they stated it."},
            "callback_number": {"type": "string", "description": "Phone number caller gave for callback."},
            "preferred_window": {"type": "string", "description": "Day and time-of-day preference."},
            "reason": {"type": "string", "description": "Reason for visit, e.g. 'cleaning', 'toothache'."},
        },
        required=["caller_name", "callback_number", "preferred_window"],
    ),
    FunctionSchema(
        name="take_message",
        description=(
            "Call this ONLY when the caller EXPLICITLY asks to leave a message for the staff or doctor "
            "(e.g. 'Can you tell Dr. Smith...', 'Please leave a message saying...', 'I want to leave a note'). "
            "Do NOT call this for general questions, off-topic questions, social chat, or anything you can answer "
            "directly. Just answer those in your own words."
        ),
        properties={
            "caller_name": {"type": "string"},
            "callback_number": {"type": "string"},
            "message": {"type": "string", "description": "The exact message the caller asked to relay."},
        },
        required=["caller_name", "callback_number", "message"],
    ),
    FunctionSchema(
        name="escalate_emergency",
        description=(
            "Call this ONLY for ACTUAL DENTAL EMERGENCIES with clear medical urgency: "
            "severe tooth pain, facial swelling, knocked-out tooth, uncontrolled bleeding, trauma to mouth/jaw. "
            "Do NOT call this for: requests to speak with a manager, general complaints, scheduling questions, "
            "or non-medical issues. For those, use take_message or just answer directly."
        ),
        properties={
            "reason": {"type": "string", "description": "Specific emergency symptom described by caller."},
        },
        required=["reason"],
    ),
])


SYSTEM_PROMPT = f"""You are the receptionist for {PRACTICE['name']}, answering for {PRACTICE['doctor']}.

VOICE FORMAT:
- Phone call. Output is spoken aloud.
- 1-2 short sentences. No markdown.
- Speak numbers naturally.

CRITICAL TOOL RULES:
- Most caller turns DO NOT need a tool. Just answer in your own words.
- Tools are for SAVING DATA, not for replying. The reply happens after the tool returns.
- For off-topic questions ("do you sell cake?", "can you fix my car?", "can you be my friend?"):
  Just answer politely in your own words. DO NOT call take_message.
- For requests to speak to a human/manager: politely say you're an automated assistant and offer
  to take a message OR have someone call them back. DO NOT call escalate_emergency.
- escalate_emergency is ONLY for medical dental emergencies (severe pain, swelling, bleeding, trauma).

APPOINTMENT FLOW:
1. Caller says they want an appointment.
2. Ask for their name.
3. Ask for their callback number.
4. Ask for their preferred day and time-of-day window.
5. Optionally ask reason for visit.
6. ONLY THEN call book_appointment_callback with all fields filled in.
7. Confirm: "Got it, someone will call you back shortly to confirm."
NEVER call book_appointment_callback with blank or 'unknown' fields. Ask first.

KNOWN INFO YOU CAN ANSWER DIRECTLY:
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

GREETING:
"Thanks for calling {PRACTICE['name']}, how can I help?"

NEVER quote prices, confirm exact times, or give medical advice.
If caller asks for a human: "I'm an automated assistant, but I can take your information and have someone call you right back."

Be warm, brief, competent.
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
        settings=OLLamaLLMService.Settings(model="receptionist-llama", temperature=0.4),
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
