"""Local AI dental receptionist - Hermes3:8b + fixed chitchat guard."""

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
from pipecat.services.settings import assert_given
from pipecat.frames.frames import ErrorFrame
from pipecat.utils.time import time_now_iso8601
import asyncio

from pipecat.transports.websocket.server import (
    WebsocketServerParams, WebsocketServerTransport,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer

load_dotenv(override=True)


WHISPER_BIAS_PROMPT = (
    "Phone call to a dental office. The caller may say short replies like "
    "yes, no, two, four, eight, ten, AM, PM, Monday, Tuesday, Wednesday, "
    "Thursday, Friday. Phone numbers are spoken as digits."
)


class BiasedWhisperSTT(WhisperSTTService):
    """Whisper service that passes an initial_prompt to bias decoding toward
    digits, times, and short affirmatives — reduces single-syllable hallucinations
    like '2' -> 'True.'"""

    async def run_stt(self, audio):
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        language = assert_given(self._settings.language)
        segments, _ = await asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=language,
            initial_prompt=WHISPER_BIAS_PROMPT,
            vad_filter=True,
        )

        text = ""
        no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
        for segment in segments:
            if (
                no_speech_prob_threshold is not None
                and segment.no_speech_prob < no_speech_prob_threshold
            ):
                text += f"{segment.text} "

        await self.stop_processing_metrics()

        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language
            )

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


_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _words_to_digits(text: str) -> str:
    """Convert any digit-words in text to digits in place. 'two zero one' -> '201'.
    Other tokens are dropped — we only return digit characters."""
    out = []
    for tok in re.findall(r"[a-zA-Z]+|\d", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in _WORD_TO_DIGIT:
            out.append(_WORD_TO_DIGIT[tok])
    return "".join(out)


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
    # Fallback: model passed phone as spelled-out words ("two zero one...").
    word_digits = _words_to_digits(text)
    if 7 <= len(word_digits) <= 11:
        return word_digits
    return None


# ---------- Manual VAD ----------
class ManualEnergyVAD(FrameProcessor):
    RMS_THRESHOLD = 800.0
    START_FRAMES = 5
    STOP_FRAMES = 25

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
    """When CALL_STATE.force_speak is set, emit it once via TTSSpeakFrame and
    suppress every assistant text frame until the user speaks again.

    Without the suppression-window, the LLM's multi-sentence stream after a
    failed tool call leaks through and Piper speaks the same confirmation
    several times back-to-back."""

    def __init__(self):
        super().__init__()
        self._suppressing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # User starting a new turn → reopen the gate.
        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._suppressing = False

        if CALL_STATE.force_speak and not self._suppressing:
            forced = CALL_STATE.force_speak
            logger.warning(f">>> FORCING SPEECH: {forced!r} <<<")
            await self.push_frame(TTSSpeakFrame(text=forced), direction)
            self._suppressing = True
            CALL_STATE.force_speak = None
            # Drop the current frame too if it carries assistant text.
            if hasattr(frame, "text") and isinstance(getattr(frame, "text", None), str):
                return

        # While suppressing, swallow any further assistant text frames.
        if self._suppressing and hasattr(frame, "text") and isinstance(getattr(frame, "text", None), str):
            return

        await self.push_frame(frame, direction)


class MalformedToolCallStripper(FrameProcessor):
    """Catch tool-calls that the LLM emitted as plain text instead of via the
    structured function-calling API, and drop them before TTS speaks them.

    Local 8-14B models (Hermes, Qwen, Llama) sometimes hallucinate corrupted
    fragments of their own tool-call template — things like:
        <tool_call>{"name": "escalate_emergency", ...}</tool_call>
        _icall_{"name": "...", "arguments": {...}}
         iNdEx_icall_{...}
    If those reach Piper, the caller hears literal gibberish ("eye-call-name-
    quote-escalate-emergency..."). The eval surfaced this most dangerously in
    the emergency category — exactly the path where it must not fail.

    Strategy: detect the pattern in any text-bearing frame; if found, strip
    the malformed segment. If nothing legible is left, drop the frame entirely.
    Don't try to re-issue it as a real tool call here — that's harder than it
    sounds, and the LLM will usually emit a proper call on the next turn."""

    # Markers that indicate text-form tool-call leakage. Order matters: the
    # broadest patterns last.
    PATTERNS = [
        re.compile(r"<\s*tool_call\s*>.*?</\s*tool_call\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<\s*tool_call\s*>.*", re.DOTALL | re.IGNORECASE),  # unterminated
        re.compile(r"</\s*tool_call\s*>", re.IGNORECASE),
        re.compile(r"_icall_[^\s]*", re.IGNORECASE),
        re.compile(r"\biNdEx[^\s]*", re.IGNORECASE),
        # Bare JSON tool-call object, e.g. {"name": "foo", "arguments": {...}}
        re.compile(
            r'\{\s*"name"\s*:\s*"[A-Za-z_][\w]*"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',
            re.DOTALL,
        ),
    ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if hasattr(frame, "text"):
            txt = getattr(frame, "text", None)
            if isinstance(txt, str) and txt.strip():
                cleaned = txt
                hit = False
                for pat in self.PATTERNS:
                    new = pat.sub(" ", cleaned)
                    if new != cleaned:
                        hit = True
                        cleaned = new
                if hit:
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    if not cleaned:
                        logger.warning(f">>> MalformedToolCallStripper: dropped frame {txt!r}")
                        return
                    logger.warning(
                        f">>> MalformedToolCallStripper: cleaned {txt!r} -> {cleaned!r}"
                    )
                    try:
                        setattr(frame, "text", cleaned)
                    except Exception:
                        pass

        await self.push_frame(frame, direction)


class FarewellDeduper(FrameProcessor):
    """Suppress every farewell-shaped sentence after the first in an assistant turn.

    Without this the model strings together "Thanks for calling X. Have a great day.
    Take care!" — three goodbyes in one turn. We let the first farewell phrase
    through and drop subsequent ones until the caller speaks again."""

    FAREWELL_PATTERNS = re.compile(
        r"\b(take care|have a (great|good|nice|wonderful|lovely) day"
        r"|good ?bye|bye now|see you|we look forward|talk to you (soon|later))\b",
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__()
        self._farewell_spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Once a farewell has been said, drop ALL subsequent assistant text for
        # the rest of the call. Don't reset on user turns — the caller saying
        # "goodbye" back shouldn't reopen the gate and trigger another farewell.

        txt = getattr(frame, "text", None) if hasattr(frame, "text") else None
        if isinstance(txt, str) and txt.strip():
            if self._farewell_spoken:
                logger.warning(f">>> Post-farewell silence — dropping: {txt!r} <<<")
                return
            if self.FAREWELL_PATTERNS.search(txt):
                self._farewell_spoken = True

        await self.push_frame(frame, direction)


class TextNormalizer(FrameProcessor):
    LEAKED_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                     "<|begin_of_text|>", "<|end_of_text|>",
                     "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
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
    "john doe", "jane doe", "john smith", "jane smith", "test", "test user",
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
    if not isinstance(s, str):
        return ""
    return re.sub(r"^[<\[\(\{]+|[>\]\)\}]+$", "", s.strip()).lower()


def _is_known_chitchat(s: str) -> bool:
    """Block tool calls only when user CLEARLY said a chitchat phrase.
    Empty/missing input does NOT count as chitchat (changed from previous version)."""
    if not s or not s.strip():
        return False  # Don't block on empty - the LLM may have valid info from earlier turns
    norm = re.sub(r"[^\w\s']", "", s.lower()).strip()
    if not norm:
        return False
    if norm in NO_RESPONSES or norm in GREETING_NOISE:
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
        if f == "callback_number" and extract_phone_digits(v) is None:
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
    # Block ONLY if user just said a clear chitchat phrase like "hi" or "bye"
    if _is_known_chitchat(CALL_STATE.last_user_text):
        msg = "How can I help?" if CALL_STATE.last_user_text.lower() in {"hi", "hello", "hey"} else "Take care!"
        CALL_STATE.force_speak = msg
        logger.warning(f"book_appointment_callback blocked: chitchat ({CALL_STATE.last_user_text!r})")
        await params.result_callback({
            "ok": False,
            "error": "User just said a greeting/farewell, no booking needed.",
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
    if name and _looks_like_garbage_name(name):
        msg = "Sorry, I didn't catch your name clearly. Could you say it again?"
        CALL_STATE.force_speak = msg
        logger.warning(f"Tool gating: bogus name {name!r}")
        await params.result_callback({"ok": False, "error": f"Bad name {name!r}", "spoken_response": msg})
        return

    missing = _missing(args, "caller_name", "callback_number", "preferred_window")
    if missing:
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

    rec = {"ts": datetime.datetime.now().isoformat(), **args, "callback_number": digits}
    fn = _save_record("callback", rec)
    CALL_STATE.booking_complete = True
    logger.info(f"Booking callback queued -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Got it, your callback is saved.",
    })


async def take_message(params):
    if _is_known_chitchat(CALL_STATE.last_user_text):
        msg = "Take care!" if _is_caller_declining(CALL_STATE.last_user_text) else "How can I help?"
        CALL_STATE.force_speak = msg
        logger.warning(f"take_message blocked: chitchat ({CALL_STATE.last_user_text!r})")
        await params.result_callback({"ok": False, "error": "Chitchat", "spoken_response": msg})
        return

    args = params.arguments or {}

    msgtxt = (args.get("message") or "").strip()
    if msgtxt and _looks_like_assistant_question(msgtxt):
        spoken = "What message should I pass along?"
        CALL_STATE.force_speak = spoken
        logger.warning(f"take_message blocked: self-question")
        await params.result_callback({"ok": False, "error": "Self-question", "spoken_response": spoken})
        return

    name = (args.get("caller_name") or "").strip()
    if name and _looks_like_garbage_name(name):
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
    rec = {"ts": datetime.datetime.now().isoformat(), **args, "callback_number": digits}
    fn = _save_record("message", rec)
    CALL_STATE.message_complete = True
    logger.info(f"Message saved -> {fn}")
    await params.result_callback({
        "ok": True,
        "spoken_response": "Message saved.",
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
            "Save a callback request. ONLY call AFTER the caller has personally told you "
            "their name, their phone number with digits, AND their preferred day/time "
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
            "Save a message ONLY when caller has explicitly asked to leave a message "
            "for the doctor. Do NOT call for greetings, declines, or off-topic chat. "
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


SYSTEM_PROMPT = f"""You are Sarah, the AI assistant for {PRACTICE['name']} (answering for {PRACTICE['doctor']}).

On the first turn, greet the caller with EXACTLY this sentence and nothing else: "Thanks for calling Smith Family Dental. This is Sarah, the AI assistant. How can I help you today?" If a caller asks whether you are a person or a bot, answer honestly that you are an AI assistant.

Speak in 1-2 short sentences per turn. This is a phone call — no markdown, no quotes, speak numbers naturally.

Phone numbers MUST be spelled out as words, never as digits with dashes. Correct: "two zero one, three eight eight, two one four nine". Wrong: "201-388-2149" or "2013882149". Apply this rule whenever you read a phone number aloud.

Closing the call: end the call with EXACTLY ONE short goodbye sentence and nothing else. Pick one of: "Take care!" / "Have a great day!" / "Goodbye!" Never combine two farewells. Specifically: never say "Thanks for calling..." and "Take care" in the same turn — pick one. After a booking succeeds, do not preemptively say goodbye; ask "Anything else I can help with?" and wait.

You do NOT have access to the office calendar or the schedule. You cannot see available slots, propose specific appointment times, or confirm a booking. Your job is to collect the caller's REQUESTED day and time as a callback request — a staff member will call them back to confirm actual availability. Never say things like "I have a slot at 2 PM" or "your appointment is booked."

To take a callback request, gather these slots one at a time:
1. Caller's name
2. Their callback phone number (read it back digit-by-digit and ask "Is that right?")
3. The day and time they would prefer

If you only hear a vague answer like "afternoon" or a single unclear word, ask them to be more specific (e.g. "What time in the afternoon works best?"). If a reply doesn't sound like a real time or day, ask them to repeat it — don't guess.

Once you have name + phone + a specific preferred day/time, call book_appointment_callback with the real values the caller gave you.

When you invoke any tool, you MUST use the structured function-calling API. Never write a tool call as plain text, JSON, XML, or pseudo-code in your spoken reply (no `<tool_call>` tags, no `_icall_...`, no `{{"name": ..., "arguments": ...}}` text). The caller is on a phone — they would hear that as gibberish.

After the booking tool returns ok:true, the request IS saved — confirm briefly and ask "Anything else I can help with?". Never apologize for a "glitch", never claim something went wrong, and never offer to redo the booking unless the caller explicitly says some specific detail (name/phone/time) is wrong.

If the caller wants to leave a message instead, gather name + callback number + message, then call take_message. NEVER invent a name (no "John Doe", no placeholders). If the caller has not said their name in this conversation, ASK for it before calling take_message. As soon as you have all three slots, call take_message immediately — do not announce that you're about to save it; just call the tool.
If the caller describes severe pain, swelling, bleeding, or trauma, call escalate_emergency.

Office info (share when asked):
- Hours: {PRACTICE['hours']}
- Address: {PRACTICE['address']}
- Emergency line: {PRACTICE['emergency_line']}

If asked to speak to a human: "I'm an automated assistant, but I can take your information and have someone call you right back."
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

    stt = BiasedWhisperSTT(
        settings=WhisperSTTService.Settings(model="distil-large-v3"),
        device="cuda",
        compute_type="float16",
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model="qwen2.5:14b",
            temperature=0,
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
        MalformedToolCallStripper(),
        farewell_deduper := FarewellDeduper(),
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

    GREETING = (
        "Thanks for calling Smith Family Dental. "
        "This is Sarah, the AI assistant. How can I help you today?"
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
        farewell_deduper._farewell_spoken = False
        # Hardcode the greeting — bypass the LLM so it can't drop a word ("Sarah, the AI.").
        context.set_messages([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": GREETING},
        ])
        await task.queue_frames([TTSSpeakFrame(text=GREETING)])

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
