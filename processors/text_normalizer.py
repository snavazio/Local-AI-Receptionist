"""TextNormalizer — convert digit-form numbers to Piper-friendly spoken words.

Problem
-------
Piper pronounces "555-1234" as "five hundred fifty-five minus one thousand two
hundred thirty-four" instead of "five five five, one two three four".  The
same issue affects any standalone digit string or phone-number pattern the LLM
emits despite instructions to spell them out.

This processor sits **between FarewellDeduper and TTS** and rewrites
``LLMTextFrame`` text so that:

* Phone-number patterns → space-separated digit words
  ("555-867-5309" → "five five five eight six seven five three zero nine")
* Isolated digit sequences → digit words
  ("2 PM" → "two PM")

The rewrite is intentionally conservative: only sequences that look like
phone numbers or isolated digits are transformed.  Regular prose numbers
(prices, years, counts) are left alone.
"""

import re

from pipecat.frames.frames import Frame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# ---------------------------------------------------------------------------
# Digit → word mapping
# ---------------------------------------------------------------------------

_DIGIT_WORDS: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# ---------------------------------------------------------------------------
# Patterns (order matters — more-specific patterns first)
# ---------------------------------------------------------------------------

# North-American style: (NNN) NNN-NNNN  or  NNN-NNN-NNNN  or  NNN.NNN.NNNN
_PHONE_FULL = re.compile(
    r"\(?\b(\d{3})\)?[\s.\-]?(\d{3})[\s.\-](\d{4})\b"
)

# Shorter digit runs that look like phone fragments: 7–10 consecutive digits
_DIGIT_RUN = re.compile(r"\b\d{7,10}\b")

# Isolated single digits surrounded by non-digits (e.g. "Press 2 for billing")
_ISOLATED_DIGIT = re.compile(r"(?<!\d)(\d)(?!\d)")


def _digits_to_words(digits: str) -> str:
    """Convert a string of digits to space-separated digit words.

    Args:
        digits: String containing only digit characters.

    Returns:
        Space-separated spoken-digit words.
    """
    return " ".join(_DIGIT_WORDS[d] for d in digits if d.isdigit())


def _replace_phone(match: re.Match) -> str:
    """Replace a full phone-number match with spoken digits."""
    digits = re.sub(r"\D", "", match.group(0))
    return _digits_to_words(digits)


def normalize_text(text: str) -> str:
    """Apply all normalisation rules to *text*.

    Args:
        text: Raw text from an LLM token or sentence.

    Returns:
        Text with phone-number and digit patterns replaced by spoken words.
    """
    # 1. Full phone-number patterns
    text = _PHONE_FULL.sub(_replace_phone, text)

    # 2. Remaining digit runs (7-10 digits)
    text = _DIGIT_RUN.sub(lambda m: _digits_to_words(m.group(0)), text)

    # 3. Isolated single digits
    text = _ISOLATED_DIGIT.sub(lambda m: _DIGIT_WORDS[m.group(1)], text)

    return text


class TextNormalizer(FrameProcessor):
    """Rewrite ``LLMTextFrame`` tokens that contain digit-form phone numbers.

    Passes all other frame types through unchanged.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            normalized = normalize_text(frame.text)
            if normalized != frame.text:
                frame = LLMTextFrame(text=normalized)

        await self.push_frame(frame, direction)
