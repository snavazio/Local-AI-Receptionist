"""Local AI dental receptionist - manual VAD via FrameProcessor.

The Pipecat 1.x WebsocketServerInputTransport doesn't invoke vad_analyzer on
incoming audio, so we synthesize UserStartedSpeakingFrame / UserStoppedSpeakingFrame
ourselves based on energy. Downstream STT/STT segmentation listens to those events.
"""

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


# ---------- Manual VAD: emits speaking start/stop frames based on RMS ----------
class ManualEnergyVAD(FrameProcessor):
    """Inline energy VAD. Looks at every InputAudioRawFrame, tracks RMS state,
    emits UserStartedSpeakingFrame when sustained loud frames detected, and
    UserStoppedSpeakingFrame after sustained silence. All audio frames are
    passed through unchanged."""

    RMS_THRESHOLD = 800.0      # int16 RMS for "speech"
    START_FRAMES = 5           # ~100ms loud => speaking
    STOP_FRAMES = 30           # ~600ms silence => stopped

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
                    await self.push_frame(UserStartedSpeakingFrame(), direction)
            else:
                self._silent_count += 1
                self._loud_count = 0
                if self._is_speaking and self._silent_count >= self.STOP_FRAMES:
                    self._is_speaking = False
                    logger.warning(">>> ManualVAD: STOPPED <<<")
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


async def transfer_to_human(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("escalation", rec)
    logger.warning(f"Escalation -> {fn}")
    await params.result_callback({"ok": True, "spoken_response": f"For dental emergencies please call {PRACTICE['emergency_line']}."})


tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="book_appointment_callback",
        description="Use when caller wants to schedule, reschedule, or cancel an appointment.",
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
        description="Use when caller wants to leave a non-appointment message.",
        properties={
            "caller_name": {"type": "string"},
            "callback_number": {"type": "string"},
            "message": {"type": "string"},
        },
        required=["caller_name", "callback_number", "message"],
    ),
    FunctionSchema(
        name="transfer_to_human",
        description="Use ONLY for dental emergencies.",
        properties={"reason": {"type": "string"}},
        required=["reason"],
    ),
])


SYSTEM_PROMPT = f"""You are the receptionist for {PRACTICE['name']}, answering for {PRACTICE['doctor']}.

VOICE FORMAT:
- Phone call. Output is spoken aloud.
- 1-2 short sentences. No markdown.
- Speak numbers naturally.

FLOW:
1. Greet warmly: "Thanks for calling {PRACTICE['name']}, how can I help?"
2. Identify intent: appointment, question, message, emergency.
3. Appointments -> gather name, callback number, preferred window, reason -> call book_appointment_callback. Tell caller staff will call back. NEVER commit to specific times.
4. General questions you can answer (hours, address) -> answer directly.
5. Anything else -> take_message.
6. Emergencies (severe pain, swelling, trauma, bleeding) -> transfer_to_human immediately.

KNOWN INFO:
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

NEVER quote prices, confirm exact times, give medical advice. If asked human, say "I'm an automated assistant, but I can take your information and have someone call you right back."

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
            vad_analyzer=None,   # we do VAD inline below
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
    llm.register_function("transfer_to_human", transfer_to_human)

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
        ManualEnergyVAD(),         # synthesizes start/stop speaking events
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
