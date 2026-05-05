"""Whisper STT with a dental-office bias prompt.

Without an ``initial_prompt``, distil-large-v3 frequently hallucinates on
short utterances (e.g. the caller saying "2" is transcribed as "True.").
Seeding the model with a domain-relevant sentence dramatically improves
accuracy for digits, days of the week, and yes/no responses.

Usage::

    stt = BiasedWhisperSTT(
        settings=BiasedWhisperSTT.Settings(
            model="distil-large-v3",
            language=Language.EN,
            no_speech_prob=0.4,
        ),
        device="cuda",
        compute_type="float16",
    )
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.settings import assert_given
except ImportError as exc:
    raise ImportError(
        "WhisperSTTService is not available. "
        "Install the whisper extra: pip install pipecat-ai[whisper] faster-whisper"
    ) from exc


# ---------------------------------------------------------------------------
# Dental-office bias prompt
# ---------------------------------------------------------------------------

_DENTAL_BIAS_PROMPT = (
    "Appointment callback for dental office. "
    "Name, phone number, day of the week, time of day. "
    "Zero, one, two, three, four, five, six, seven, eight, nine. "
    "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday. "
    "Morning, afternoon, evening. "
    "Yes, no, please, thank you."
)


class BiasedWhisperSTT(WhisperSTTService):
    """``WhisperSTTService`` subclass that injects a dental-office bias prompt.

    The prompt steers the model toward the vocabulary and phonetic patterns
    that appear in phone calls to a dental office: digits read individually,
    day names, time-of-day phrases, and common courtesy words.

    All constructor arguments are forwarded to ``WhisperSTTService``.  The
    only difference in behaviour is that ``initial_prompt`` is automatically
    passed to ``WhisperModel.transcribe``.

    Args:
        initial_prompt: Override the default dental-office prompt.  Pass
            ``None`` to disable bias entirely (falls back to standard
            ``WhisperSTTService`` behaviour).
        **kwargs: Forwarded verbatim to ``WhisperSTTService.__init__``.
    """

    def __init__(self, *, initial_prompt: str | None = _DENTAL_BIAS_PROMPT, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initial_prompt = initial_prompt

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe *audio* using the injected bias prompt.

        Overrides ``WhisperSTTService.run_stt`` only to add ``initial_prompt``
        to the ``WhisperModel.transcribe`` call.  All other logic (metrics,
        ``no_speech_prob`` filtering, frame emission) is reproduced faithfully.

        Args:
            audio: Raw audio bytes in 16-bit signed PCM format.

        Yields:
            ``TranscriptionFrame`` on success, ``ErrorFrame`` on failure.
        """
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        language = assert_given(self._settings.language)
        kwargs: dict = dict(language=language)
        if self._initial_prompt:
            kwargs["initial_prompt"] = self._initial_prompt

        segments, _ = await asyncio.to_thread(self._model.transcribe, audio_float, **kwargs)

        text = ""
        no_speech_threshold = assert_given(self._settings.no_speech_prob)
        for segment in segments:
            if no_speech_threshold is not None and segment.no_speech_prob < no_speech_threshold:
                text += f"{segment.text} "

        await self.stop_processing_metrics()

        if text.strip():
            logger.debug(f"BiasedWhisperSTT transcription: [{text.strip()}]")
            yield TranscriptionFrame(
                text.strip(),
                self._user_id,
                time_now_iso8601(),
                language,
            )
