"""Local AI dental receptionist - Pipecat 1.1.0.

Runs entirely on local infrastructure:
  - STT: faster-whisper (distil-large-v3 on CUDA)
  - LLM: Ollama (receptionist variant of phi3.5)
  - TTS: Piper (en_US-lessac-medium)
  - Transport: WebSocket (test client connects directly)

For session 1, customer audio is decoded locally; nothing leaves the box
except via Ollama's localhost API.
"""

import os
import json
import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import LLMContextFrame
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService

from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer

load_dotenv(override=True)

# ---------- Practice config (edit per client) ----------
PRACTICE = {
    "name": "Smith Family Dental",
    "doctor": "Dr. Smith",
    "hours": "Monday through Friday, eight to five",
    "address": "one two three Main Street",
    "emergency_line": "five five five, one two three four",
}

# ---------- Paths ----------
PIPER_VOICE = os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx")
LOG_DIR = Path("./call_logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------- Tool implementations (logging stubs for now) ----------
def _save_record(kind: str, data: dict) -> Path:
    fn = LOG_DIR / f"{kind}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    fn.write_text(json.dumps(data, indent=2, default=str))
    return fn


async def book_appointment_callback(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("callback", rec)
    logger.info(f"Booking callback queued -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Got it. Someone from the office will call you back shortly to confirm the time.",
    })


async def take_message(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("message", rec)
    logger.info(f"Message saved -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Message saved. The office will reach out soon.",
    })


async def transfer_to_human(params):
    rec = {"ts": datetime.datetime.now().isoformat(), **params.arguments}
    fn = _save_record("escalation", rec)
    logger.warning(f"Escalation -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": f"For dental emergencies please call {PRACTICE['emergency_line']}.",
    })


# ---------- Tool schemas ----------
tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="book_appointment_callback",
        description=(
            "Use when a caller wants to schedule, reschedule, or cancel an appointment. "
            "Collect their name, callback number, preferred day/window, and reason, then call this tool. "
            "Do NOT commit to a specific exact time on the call - staff will call back to confirm."
        ),
        properties={
            "caller_name": {"type": "string", "description": "Caller's full name"},
            "callback_number": {"type": "string", "description": "Best phone number for callback"},
            "preferred_window": {"type": "string", "description": "Preferred day/time window, e.g. 'Tuesday afternoon'"},
            "reason": {"type": "string", "description": "Brief reason: cleaning, pain, consultation, etc."},
        },
        required=["caller_name", "callback_number", "preferred_window"],
    ),
    FunctionSchema(
        name="take_message",
        description="Use when caller wants to leave a message that isn't an appointment - billing, asking the doctor a question, etc.",
        properties={
            "caller_name": {"type": "string"},
            "callback_number": {"type": "string"},
            "message": {"type": "string", "description": "What the caller wants the practice to know"},
        },
        required=["caller_name", "callback_number", "message"],
    ),
    FunctionSchema(
        name="transfer_to_human",
        description="Use ONLY for dental emergencies (severe pain, swelling, trauma, bleeding) or if caller explicitly insists on speaking to a person.",
        properties={
            "reason": {"type": "string", "description": "Why escalating"},
        },
        required=["reason"],
    ),
])


# ---------- System prompt ----------
SYSTEM_PROMPT = f"""You are the receptionist for {PRACTICE['name']}, answering for {PRACTICE['doctor']}.

VOICE FORMAT - critical:
- This is a phone call. Your output is spoken aloud.
- 1-2 short sentences per response. No markdown, no lists, no bullets.
- Speak numbers naturally ("five five five, one two three four").
- If you don't know something, say so plainly. Never invent hours, prices, or appointment times.

FLOW:
1. Greet warmly: "Thanks for calling {PRACTICE['name']}, how can I help?"
2. Identify intent: appointment, question, message, or emergency.
3. Appointments -> gather name, callback number, preferred window, reason -> call book_appointment_callback. Tell caller staff will call back to confirm exact time. NEVER commit to a specific time on this call.
4. General questions you can answer (hours, address) -> answer directly.
5. Anything else -> take_message.
6. Emergencies (severe pain, swelling, trauma, bleeding) -> transfer_to_human immediately.

KNOWN INFO YOU CAN SHARE:
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- After-hours emergency line: {PRACTICE['emergency_line']}

NEVER:
- Quote prices (always "staff will follow up on that").
- Confirm specific appointment times (always book a callback).
- Give medical or dental advice.
- Pretend to be human if directly asked - say "I'm an automated assistant, but I can take your information and have someone call you right back."

Be warm, brief, and competent. Patients are often anxious. Get them sorted quickly.
"""


# ---------- Bot entry point ----------
async def main():    
    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=8765,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model="distil-large-v3"),
        device="cuda",
        compute_type="float16",
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(model="receptionist", temperature=0.4),
    )
    # llm.register_function("book_appointment_callback", book_appointment_callback)
    # llm.register_function("take_message", take_message)
    # llm.register_function("transfer_to_human", transfer_to_human)

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice="en_US-lessac-medium"),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        # tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=22050,  # Piper lessac-medium native rate
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - greeting caller")
        context.add_message({"role": "system", "content": "Greet the caller now per your instructions."})
        await task.queue_frames([LLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        transcript = context.get_messages()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"call_{ts}.json"
        log_path.write_text(json.dumps(transcript, indent=2, default=str))
        logger.info(f"Call ended. Transcript -> {log_path}")
        # await task.cancel()  # disabled: keep server alive across reconnects

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
