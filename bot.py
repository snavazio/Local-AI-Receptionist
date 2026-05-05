"""Local AI Receptionist — main bot pipeline.

Run with::

    python bot.py

The bot:
* Listens for WebSocket connections (default: 0.0.0.0:8765).
* On connect, speaks a hardcoded greeting (bypasses the LLM to guarantee
  the exact wording).
* Runs a Pipecat pipeline:
    WebSocket input
    → RMS-energy VAD
    → BiasedWhisperSTT   (faster-whisper distil-large-v3, dental bias prompt)
    → LLM user aggregator
    → OLLamaLLMService   (Qwen 2.5 14B via Ollama /v1)
    → ForcedSpeechOverride
    → FarewellDeduper
    → TextNormalizer
    → PiperTTSService    (en_US-lessac-medium)
    → WebSocket output
    → LLM assistant aggregator
* On disconnect, writes ``call_logs/call_<timestamp>.json`` with the full
  transcript.

Environment variables (copy ``.env.example`` to ``.env`` and adjust):
    WS_HOST, WS_PORT,
    OLLAMA_BASE_URL, LLM_MODEL,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    PIPER_VOICE, PIPER_DOWNLOAD_DIR,
    EMERGENCY_LINE, IDLE_TIMEOUT_SECS
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Pipecat imports
# ---------------------------------------------------------------------------
from pipecat.frames.frames import (
    EndFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from processors.biased_whisper import BiasedWhisperSTT
from processors.farewell import FarewellDeduper
from processors.forced_speech import ForcedSpeechOverride
from processors.text_normalizer import TextNormalizer
from processors.vad import RMSEnergyVAD
from tools import (
    TOOLS_SCHEMA,
    CallState,
    handle_escalate_emergency,
    handle_save_callback_request,
    handle_save_message,
)

# ---------------------------------------------------------------------------
# Configuration (from environment with sensible defaults)
# ---------------------------------------------------------------------------

WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT: int = int(os.getenv("WS_PORT", "8765"))

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:14b")

WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

PIPER_VOICE: str = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
PIPER_DOWNLOAD_DIR: Path = Path(os.getenv("PIPER_DOWNLOAD_DIR", "./voice_models"))

EMERGENCY_LINE: str = os.getenv("EMERGENCY_LINE", "9-1-1")
IDLE_TIMEOUT_SECS: float = float(os.getenv("IDLE_TIMEOUT_SECS", "120"))

CALL_LOGS_DIR: Path = Path("call_logs")
CALL_LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Sarah, the AI phone receptionist for Bright Smile Dental.
Your job is to help callers in a warm, friendly, and efficient manner.

You can help callers in three ways:
1. Collect an appointment callback request (name, callback phone, preferred day and time).
2. Take a message for the dentist or office staff.
3. Escalate a dental emergency to the emergency line.

Appointment callback requests:
- Ask for the caller's full name.
- Ask for their callback phone number, then read it back digit by digit to confirm.
- Ask for their preferred day (for example, Monday or next Tuesday).
- Ask for their preferred time or period (for example, morning, afternoon, or 2 PM).
- Ask for one piece of information at a time; do not ask multiple questions at once.
- Once you have all four pieces of information, call save_callback_request.
- You do not have access to the calendar and never mention specific available slots.
  A staff member will call back to confirm actual availability.

Taking a message:
- Ask for the caller's name and their message.
- Call save_message once you have both.

Dental emergencies:
- If the caller describes severe tooth pain, swelling, uncontrolled bleeding,
  a knocked-out tooth, jaw injury, or any trauma to the mouth, call escalate_emergency
  immediately with their name and a brief description of the situation.

General guidelines:
- Keep responses short and conversational — this is a phone call.
- Ask only one question at a time.
- Spell out phone numbers as individual digits when speaking them.
- Say goodbye at the end of the call.
- Do not invent calendar availability or make promises about scheduling.
"""

# ---------------------------------------------------------------------------
# Greeting (hardcoded — bypasses the LLM to guarantee exact wording)
# ---------------------------------------------------------------------------

GREETING = (
    "Thank you for calling Bright Smile Dental. "
    "This is Sarah, your AI assistant. "
    "How can I help you today?"
)

# ---------------------------------------------------------------------------
# Transcript writer
# ---------------------------------------------------------------------------


def _write_call_transcript(state: CallState) -> None:
    """Write the full call transcript to ``call_logs/call_<timestamp>.json``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CALL_LOGS_DIR / f"call_{ts}.json"
    payload = {
        "call_id": state.call_id,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "transcript": state.transcript,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Call transcript written: {path}")


# ---------------------------------------------------------------------------
# Bot factory — one pipeline per connection
# ---------------------------------------------------------------------------


async def run_bot() -> None:
    """Start the WebSocket server and run one pipeline instance.

    A single Pipecat pipeline is kept alive for the duration of the server
    process.  For a production deployment handling concurrent callers you
    would spawn one pipeline per WebSocket connection; for a single-workstation
    dental office one call at a time is the expected use case.
    """

    # ── Per-call shared state ──────────────────────────────────────────────
    call_state = CallState(call_id=str(uuid.uuid4()), emergency_line=EMERGENCY_LINE)

    # ── Transport ─────────────────────────────────────────────────────────
    transport = WebsocketServerTransport(
        host=WS_HOST,
        port=WS_PORT,
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=22_050,  # Piper lessac-medium native rate
        ),
    )

    # ── VAD ───────────────────────────────────────────────────────────────
    vad = VADProcessor(vad_analyzer=RMSEnergyVAD(sample_rate=16_000))

    # ── STT ───────────────────────────────────────────────────────────────
    stt = BiasedWhisperSTT(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        settings=BiasedWhisperSTT.Settings(
            model=WHISPER_MODEL,
            language=Language.EN,
            # Filter out segments where the model is more than 40% confident
            # there is no speech — reduces hallucinations on short utterances.
            no_speech_prob=0.4,
        ),
    )

    # ── LLM context ───────────────────────────────────────────────────────
    context = LLMContext()
    context.add_message({"role": "system", "content": SYSTEM_PROMPT})
    context.set_tools(TOOLS_SCHEMA)

    context_aggregators = LLMContextAggregatorPair(context)
    user_agg = context_aggregators.user()
    assistant_agg = context_aggregators.assistant()

    # ── LLM ───────────────────────────────────────────────────────────────
    llm = OLLamaLLMService(
        base_url=OLLAMA_BASE_URL,
        settings=OLLamaLLMService.Settings(model=LLM_MODEL),
    )

    # Register tool handlers
    llm.register_function("save_callback_request", handle_save_callback_request)
    llm.register_function("save_message", handle_save_message)
    llm.register_function("escalate_emergency", handle_escalate_emergency)

    # ── Custom processors ─────────────────────────────────────────────────
    forced_speech = ForcedSpeechOverride(call_state=call_state)
    farewell_deduper = FarewellDeduper()
    text_normalizer = TextNormalizer()

    # ── TTS ───────────────────────────────────────────────────────────────
    PIPER_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tts = PiperTTSService(
        download_dir=PIPER_DOWNLOAD_DIR,
        settings=PiperTTSService.Settings(voice=PIPER_VOICE),
    )

    # ── Pipeline ──────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_agg,
            llm,
            forced_speech,
            farewell_deduper,
            text_normalizer,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

    # ── Task ──────────────────────────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        tool_resources=call_state,
        idle_timeout_secs=IDLE_TIMEOUT_SECS,
    )

    # ── Event handlers ────────────────────────────────────────────────────

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        logger.info(f"Client connected: {websocket.remote_address}")
        # Reset per-call state for a fresh conversation.
        call_state.call_id = str(uuid.uuid4())
        call_state.transcript = []
        call_state.forced_speech_text = None
        call_state.call_ended = False
        # Hardcoded greeting — bypasses LLM to guarantee exact wording.
        await task.queue_frame(TTSSpeakFrame(text=GREETING))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        logger.info(f"Client disconnected: {websocket.remote_address}")
        _write_call_transcript(call_state)
        await task.queue_frame(EndFrame())

    # ── Transcript capture (uses aggregator events) ────────────────────────

    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        call_state.transcript.append(
            {
                "role": "user",
                "text": message.content,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if message.content:
            call_state.transcript.append(
                {
                    "role": "assistant",
                    "text": message.content,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )

    # ── Run ───────────────────────────────────────────────────────────────
    runner = PipelineRunner()
    logger.info(f"Starting WebSocket server on {WS_HOST}:{WS_PORT}")
    await runner.run(task)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_bot())
