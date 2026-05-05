"""ForcedSpeechOverride — speak tool confirmations, suppress the LLM follow-up.

Problem
-------
When a tool (e.g. ``save_callback_request``) returns a deterministic
confirmation string, the LLM is still called with the tool result and
generates its own follow-up.  Because Piper TTS is sequential the caller
would hear:

1. The confirmation from the tool result (spoken via TTSSpeakFrame).
2. The LLM paraphrasing the same confirmation yet again.

This processor sits **between the LLM and TTS** and:

* Reads a *forced-speech* text that the tool handler stores in the shared
  ``CallState`` object (passed as ``tool_resources`` on ``PipelineTask``).
* When the next LLM response starts (``LLMFullResponseStartFrame``), if a
  forced-speech text is pending it:
    1. Emits ``TTSSpeakFrame`` with that text (spoken immediately by Piper).
    2. Suppresses all ``LLMTextFrame`` tokens for that turn.
    3. Still forwards the ``LLMFullResponseEndFrame`` so downstream
       aggregators stay in sync.

If no forced-speech text is pending the response flows through unchanged.
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ForcedSpeechOverride(FrameProcessor):
    """Intercept LLM follow-ups after tool calls and replace them with a
    pre-computed spoken confirmation.

    Args:
        call_state: Shared ``CallState`` instance.  The tool handler sets
            ``call_state.forced_speech_text`` before returning; this
            processor reads and clears it.
    """

    def __init__(self, call_state, **kwargs) -> None:
        super().__init__(**kwargs)
        self._call_state = call_state
        self._suppressing: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Only intercept downstream frames (LLM → TTS direction).
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            speak_text = self._call_state.forced_speech_text
            if speak_text:
                logger.debug(f"ForcedSpeechOverride: speaking forced text, suppressing LLM turn")
                self._call_state.forced_speech_text = None
                self._suppressing = True
                # Speak the deterministic confirmation instead of the LLM's version.
                await self.push_frame(TTSSpeakFrame(text=speak_text))
                # Do NOT forward LLMFullResponseStartFrame — suppressing this turn.
                return
            # No override pending — pass through normally.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            was_suppressing = self._suppressing
            self._suppressing = False
            if was_suppressing:
                # Forward the end frame so aggregators stay consistent.
                await self.push_frame(frame, direction)
                return
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and self._suppressing:
            # Drop the LLM token — the TTSSpeakFrame already covers this turn.
            return

        await self.push_frame(frame, direction)
