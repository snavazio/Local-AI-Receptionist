"""Local AI dental receptionist - hardened against placeholder injection + over-eager tool calls."""

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
    TTSSpeakFrame,
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


class CallState:
    def __init__(self):
        self.booking_complete = False
        self.message_complete = False
        self.last_user_text = ""
        self.pending_phone = None
        self.confirmed_phone = None
        self.force_speak = None

CALL_STATE = CallState()


# ---------- Phone normalization ----------
DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

PHONE_PATTERN = re.compile(
    r"\(?\b\d{3}\)?[\s.\-]*\d{3}[\s.\-]*\d{4}\b"
    r"|\b\d{7,11}\b"
)


def speak_digits(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        parts = [digits[:3], digits[3:6], digits[6:]]
    elif len(digits) == 11 and digits[0] == "1":
        parts = [digits[0], digits[1:4], digits[4:7], digits[7:]]
    elif len(digits) == 7:
        parts = [digits[:3], digits[3:]]
    else:
        parts = [digits]
    return ", ".join(" ".join(DIGIT_WORDS[d] for d in p) for p in parts)


def normalize_for_tts(text: str) -> str:
    return PHONE_PATTERN.sub(lambda m: speak_digits(m.group(0)), text)


def extract_phone_digits(text: str) -> str | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if 7 <= len(digits) <= 11:
        return digits
    m = PHONE_PATTERN.search(text)
    if m:
        d = re.sub(r"\D", "", m.group(0))
        if 7 <= len(d) <= 11:
            return d
    return None


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


class ForcedSpeechOverride(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if CALL_STATE.force_speak and not self._spoken:
            if hasattr(frame, "text") and isinstance(getattr(frame, "text", None), str):
                forced = CALL_STATE.force_speak
                logger.warning(f">>> FORCING SPEECH: {forced!r} <<<")
                await self.push_frame(TTSSpeakFrame(text=forced), direction)
                self._spoken = True
                CALL_STATE.force_speak = None
                return

        if self._spoken and not CALL_STATE.force_speak:
            self._spoken = False

        await self.push_frame(frame, direction)


class TextNormalizer(FrameProcessor):
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
                cleaned = normalize_for_tts(cleaned)
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


# ---------- Hardened placeholder + intent detection ----------
PLACEHOLDER_VALUES = {
    "", "unknown", "none", "null", "n/a", "na", "tbd", "to be determined",
    "string", "name", "phone", "number", "callback_number", "caller_name",
    "message", "the caller", "caller", "user", "anonymous", "no name",
    "no number", "not provided", "not given", "nil",
}

NO_RESPONSES = {
    "no", "no thanks", "no thank you", "nope", "nah", "no thats it", "thats all",
    "that's it", "that's all", "im good", "i'm good", "all good", "okay bye",
    "ok bye", "bye", "goodbye", "thanks bye", "bye bye", "see ya",
}

YES_RESPONSES = {
    "yes", "yeah", "yep", "yup", "correct", "thats right", "that's right",
    "right", "thats it", "that's it", "yes thats right", "yes correct",
    "sounds right", "sounds good", "yes please", "uh huh", "mhm", "ya",
}

GREETING_NOISE = {
    "hi", "hello", "hey", "yo", "hiya", "howdy",
    "test", "testing", "testing testing", "can you hear me", "hello hello",
}


def _normalize_placeholder(s) -> str:
    """Strip angle brackets, square brackets, parens, curly braces from placeholder values.
    LLMs love wrapping placeholders in <unknown>, [name], (phone), {number}."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"^[<\[\(\{]+|[>\]\)\}]+$", "", s.strip()).lower()


def _is_short_chitchat(s: str) -> bool:
    """If user's last utterance is just a greeting/farewell, no tool should fire."""
    if not s:
        return True
    norm = re.sub(r"[^\w\s']", "", s.lower()).strip()
    if not norm:
        return True
    if norm in NO_RESPONSES or norm in GREETING_NOISE:
        return True
    if len(norm.split()) < 2 and norm in {"yes", "no", "ok", "okay", "sure"}:
        return True
    return False


def _missing(args: dict, *fields) -> list:
    out = []
    for f in fields:
        v = args.get(f)
        if v is None:
            out.append(f); continue
        if not isinstance(v, str):
            continue
        normalized = _normalize_placeholder(v)
        if normalized in PLACEHOLDER_VALUES:
            out.append(f)
            continue
        if f == "callback_number" and not re.search(r"\d", v):
            out.append(f)
            continue
        if f == "caller_name" and len(normalized) < 2:
            out.append(f)
    return out


def _looks_like_assistant_question(s: str) -> bool:
    if not isinstance(s, str):
        return False
    low = s.lower()
    tells = ["would you like", "is there anything", "can i get", "could i get",
             "do you have", "what day", "what time", "what's the best", "anything else"]
    return s.strip().endswith("?") or any(t in low for t in tells)


def _is_caller_declining(s: str) -> bool:
    norm = re.sub(r"[^\w\s']", "", s.lower()).strip()
    return norm in NO_RESPONSES


def _is_caller_affirming(s: str) -> bool:
    norm = re.sub(r"[^\w\s']", "", s.lower()).strip()
    return norm in YES_RESPONSES


def _looks_like_garbage_name(s: str) -> bool:
    if not s:
        return True
    low = s.lower().strip()
    bad_tokens = ["shirts", ".com", "http", "www.", "@", "<", ">"]
    return any(b in low for b in bad_tokens)


# ---------- Tool implementations ----------
async def book_appointment_callback(params):
    # Hard guard: if last user turn was just chitchat/farewell, refuse outright
    if _is_short_chitchat(CALL_STATE.last_user_text):
        msg = "Could I get your name to start?"
        CALL_STATE.force_speak = msg
        logger.warning(f"book_appointment_callback blocked: chitchat ({CALL_STATE.last_user_text!r})")
        await params.result_callback({
            "ok": False,
            "error": "User has not actually given any info yet. Ask for name.",
            "spoken_response": msg,
        })
        return

    if CALL_STATE.booking_complete:
        msg = "We've already got your callback scheduled. Anything else?"
        CALL_STATE.force_speak = msg
        logger.warning("book_appointment_callback called after booking complete - ignoring")
        await params.result_callback({"ok": False, "error": "Already booked", "spoken_response": msg})
        return

    args = params.arguments or {}

    name = (args.get("caller_name") or "").strip()
    if _looks_like_garbage_name(name):
        msg = "Sorry, I didn't catch your name clearly. Could you say it again?"
        CALL_STATE.force_speak = msg
        logger.warning(f"Tool gating: bogus name {name!r}")
        await params.result_callback({"ok": False, "error": f"Bad name {name!r}", "spoken_response": msg})
        return

    missing = _missing(args, "caller_name", "callback_number", "preferred_window")
    if missing:
        # Ask only for the FIRST missing field, one at a time
        prompts = {
            "caller_name": "Could I get your name?",
            "callback_number": "What's the best phone number to call you back on?",
            "preferred_window": "What day and time works best for you?",
        }
        ask = prompts[missing[0]]
        CALL_STATE.force_speak = ask
        logger.warning(f"Tool gating rejected: missing={missing}")
        await params.result_callback({"ok": False, "error": f"Missing {missing}", "spoken_response": ask})
        return

    raw_number = args.get("callback_number", "")
    digits = extract_phone_digits(raw_number)
    if digits is None or len(digits) < 7:
        msg = "I didn't catch your phone number clearly. Could you say it again, slowly?"
        CALL_STATE.force_speak = msg
        await params.result_callback({"ok": False, "error": "Bad phone", "spoken_response": msg})
        return

    if CALL_STATE.confirmed_phone != digits:
        CALL_STATE.pending_phone = digits
        spoken_back = speak_digits(digits)
        msg = f"Just to confirm, your number is {spoken_back}. Is that right?"
        CALL_STATE.force_speak = msg
        logger.warning(f"Phone needs confirmation: {digits}")
        await params.result_callback({"ok": False, "error": "Confirm phone", "spoken_response": msg})
        return

    rec = {"ts": datetime.datetime.now().isoformat(), **args, "callback_number": digits}
    fn = _save_record("callback", rec)
    CALL_STATE.booking_complete = True
    logger.info(f"Booking callback queued -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Got it. Someone from the office will call you back shortly to confirm the time. Anything else I can help with?",
    })


async def take_message(params):
    if _is_short_chitchat(CALL_STATE.last_user_text):
        msg = "Take care!" if _is_caller_declining(CALL_STATE.last_user_text) else "How can I help?"
        CALL_STATE.force_speak = msg
        logger.warning(f"take_message blocked: chitchat ({CALL_STATE.last_user_text!r})")
        await params.result_callback({"ok": False, "error": "Chitchat", "spoken_response": msg})
        return

    args = params.arguments or {}

    msgtxt = (args.get("message") or "").strip()
    if _looks_like_assistant_question(msgtxt):
        spoken = "What message should I pass along?"
        CALL_STATE.force_speak = spoken
        logger.warning(f"take_message blocked: self-question")
        await params.result_callback({"ok": False, "error": "Self-question", "spoken_response": spoken})
        return

    name = (args.get("caller_name") or "").strip()
    if _looks_like_garbage_name(name):
        spoken = "Sorry, I didn't catch your name clearly. Could you say it again?"
        CALL_STATE.force_speak = spoken
        await params.result_callback({"ok": False, "error": "Bad name", "spoken_response": spoken})
        return

    missing = _missing(args, "caller_name", "callback_number", "message")
    if missing:
        prompts = {
            "caller_name": "Could I get your name?",
            "callback_number": "What's the best callback number?",
            "message": "What message should I pass along?",
        }
        ask = prompts[missing[0]]
        CALL_STATE.force_speak = ask
        await params.result_callback({"ok": False, "error": f"Missing {missing}", "spoken_response": ask})
        return

    digits = extract_phone_digits(args.get("callback_number", ""))
    if digits is None:
        msg = "I didn't catch your phone number clearly. Could you say it again, slowly?"
        CALL_STATE.force_speak = msg
        await params.result_callback({"ok": False, "error": "Bad phone", "spoken_response": msg})
        return
    if CALL_STATE.confirmed_phone != digits:
        CALL_STATE.pending_phone = digits
        spoken_back = speak_digits(digits)
        msg = f"Just to confirm, your number is {spoken_back}. Is that right?"
        CALL_STATE.force_speak = msg
        await params.result_callback({"ok": False, "error": "Confirm phone", "spoken_response": msg})
        return

    rec = {"ts": datetime.datetime.now().isoformat(), **args, "callback_number": digits}
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
            "Save a callback request. ONLY call this AFTER the caller has personally "
            "told you their name, their phone number with digits, AND their preferred day/time "
            "in this conversation. Do NOT call with placeholder values."
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
            "Save a message ONLY when the caller has explicitly asked to leave a message "
            "for the doctor. Do NOT call this for greetings, declines, or off-topic chat. "
            "Do NOT call with placeholder values."
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
        description="Use ONLY when caller describes severe pain, swelling, bleeding, knocked-out tooth, or trauma.",
        properties={"reason": {"type": "string"}},
        required=["reason"],
    ),
])


SYSTEM_PROMPT = f"""You are the receptionist at {PRACTICE['name']}, answering for {PRACTICE['doctor']}.

JOB: Help one caller, briefly and warmly. Most turns need NO tool — just talk like a normal person.

FORMAT:
- Phone call. Spoken aloud. 1-2 short sentences. No quotes, no markdown.
- Speak numbers naturally.

CRITICAL RULES:
- DO NOT call any tool until the caller has actually given you the required information IN THIS CONVERSATION.
- DO NOT use placeholder values like "unknown", "<unknown>", "[name]", "null", "string". If you don't have the info, ASK the caller — do not call the tool.
- DO NOT proactively ask "is this an emergency?" — wait for caller to mention symptoms.
- DO NOT make up office facts. Use only KNOWN INFO below.
- DO NOT apologize for "mistakes" — just ask the next question politely.
- When caller says "no", "bye", "thanks", "all good", "hi": just chat back. NO tool call.
- Greetings and farewells need NO tool call. Just respond conversationally.

APPOINTMENT FLOW (one question per turn, then book):
1. Caller wants appointment → ask "What's your name?" (no tool yet)
2. Got name → ask "What's the best callback number?" (no tool yet)
3. Got number → repeat back, ask "Is that right?" (no tool yet)
4. Confirmed → ask "What day and time works?" (no tool yet)
5. Got all three pieces → NOW call book_appointment_callback with the real values.

KNOWN INFO (only when asked):
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

GREETING (first turn): "Thanks for calling {PRACTICE['name']}, how can I help?"

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
            model="receptionist-llama",   # Back to llama3.2:3b - proven on the booking flow
            temperature=0.1,
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

    class ConfirmationInterceptor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and CALL_STATE.pending_phone:
                txt = (frame.text or "").strip()
                if _is_caller_affirming(txt):
                    CALL_STATE.confirmed_phone = CALL_STATE.pending_phone
                    CALL_STATE.pending_phone = None
                    logger.warning(f">>> Phone confirmed: {CALL_STATE.confirmed_phone} <<<")
                elif _is_caller_declining(txt) or "wrong" in txt.lower():
                    logger.warning(f">>> Phone rejected, clearing pending <<<")
                    CALL_STATE.pending_phone = None
                    CALL_STATE.confirmed_phone = None
            await self.push_frame(frame, direction)

    pipeline = Pipeline([
        transport.input(),
        ManualEnergyVAD(),
        IncomingAudioLogger(),
        stt,
        ConfirmationInterceptor(),
        context_aggregator.user(),
        llm,
        ForcedSpeechOverride(),
        TextNormalizer(),
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
        CALL_STATE.booking_complete = False
        CALL_STATE.message_complete = False
        CALL_STATE.last_user_text = ""
        CALL_STATE.pending_phone = None
        CALL_STATE.confirmed_phone = None
        CALL_STATE.force_speak = None
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
