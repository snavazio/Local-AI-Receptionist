"""FarewellDeduper — latch on the first goodbye and drop all subsequent ones.

Problem
-------
LLMs tend to be polite.  After the caller says "goodbye", the bot says
"goodbye", which the caller echoes back, which triggers another goodbye from
the bot — sometimes 3-4 times.

This processor sits **between the LLM and TTS**.  Once any LLM response
contains a farewell phrase, a latch is set:

* The response that *triggered* the latch is forwarded in full.
* Every subsequent LLM response is silently dropped.

The bot therefore says goodbye exactly once.
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# Phrases (lower-cased) whose presence in an LLM response marks it as a farewell.
_FAREWELL_PHRASES: tuple[str, ...] = (
    "goodbye",
    "good-bye",
    "bye ",
    "bye!",
    "farewell",
    "take care",
    "have a great day",
    "have a good day",
    "have a wonderful day",
)


class FarewellDeduper(FrameProcessor):
    """Drop duplicate bot farewells.

    Allows the *first* farewell response through unchanged.  All subsequent
    LLM responses are suppressed once the latch is set.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Set after the first farewell response completes.
        self._farewell_latch: bool = False
        # True while we are accumulating text for the current response.
        self._in_response: bool = False
        # Accumulated text for the response currently being processed.
        self._response_buffer: str = ""
        # True when the *current* response is being suppressed.
        self._suppress_current: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            if self._farewell_latch:
                logger.debug("FarewellDeduper: latch active — suppressing LLM response")
                self._suppress_current = True
                # Do not forward the start frame.
                return
            self._in_response = True
            self._response_buffer = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            # Check whether this response contained a farewell.
            if self._in_response and self._contains_farewell(self._response_buffer):
                logger.debug("FarewellDeduper: farewell detected — setting latch")
                self._farewell_latch = True
            self._in_response = False
            self._response_buffer = ""
            if self._suppress_current:
                self._suppress_current = False
                return  # Drop the end frame of the suppressed response.
            await self.push_frame(frame, direction)
            return

        if self._suppress_current:
            # Drop everything inside a suppressed response.
            return

        if isinstance(frame, LLMTextFrame) and self._in_response:
            self._response_buffer += frame.text

        await self.push_frame(frame, direction)

    @staticmethod
    def _contains_farewell(text: str) -> bool:
        """Return True if *text* contains any farewell phrase."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in _FAREWELL_PHRASES)
