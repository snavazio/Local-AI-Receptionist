"""Tests for the FrameProcessor classes in bot.py.

We don't run the async process_frame methods (that needs a Pipecat
pipeline runtime). What we do test:
  - The static regex patterns compile and match what we expect them to.
  - The class-level constants are sane.
  - The text-cleaning logic on TextNormalizer can be tested by directly
    calling its inner functions / regex.

This is less coverage than a full-pipeline integration test but it
catches the most common regression: someone tweaks a regex and breaks
the pattern match for a leak/farewell phrase the eval harness doesn't
exercise via this exact path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


# ---------- MalformedToolCallStripper.PATTERNS ----------

class TestMalformedToolCallPatterns:
    """The regex patterns are also mirrored in eval/harness.py
    (_TOOLCALL_LEAK_PATTERNS). These tests pin the bot.py side."""

    def _strip(self, text):
        cleaned = text
        for pat in bot.MalformedToolCallStripper.PATTERNS:
            cleaned = pat.sub(" ", cleaned)
        # Mirror the production processor's whitespace collapse
        import re
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def test_xml_tool_call(self):
        s = '<tool_call>{"name": "foo"}</tool_call>'
        assert self._strip(s) == ""

    def test_unterminated_xml(self):
        s = '<tool_call>{"name": "foo", broken'
        assert self._strip(s) == ""

    def test_icall_token(self):
        s = '_icall_{"name":"foo"}'
        assert self._strip(s) == ""

    def test_iNdEx_garbage(self):
        s = ' iNdEx_icism garbage'
        # iNdEx_icism is exactly the Qwen template-leak token we observed
        out = self._strip(s)
        assert "iNdEx" not in out

    def test_bare_json_tool_call(self):
        s = '{"name": "escalate_emergency", "arguments": {"reason": "pain"}}'
        assert self._strip(s) == ""

    def test_clean_text_preserved(self):
        s = "Got it, your callback is saved."
        assert self._strip(s) == s


# ---------- FarewellDeduper.FAREWELL_PATTERNS ----------

class TestFarewellPatterns:
    def _matches(self, text):
        return bool(bot.FarewellDeduper.FAREWELL_PATTERNS.search(text))

    def test_take_care(self):
        assert self._matches("Take care!")
        assert self._matches("Please take care today")

    def test_have_a_x_day(self):
        assert self._matches("Have a great day")
        assert self._matches("Have a wonderful day")
        assert self._matches("Have a nice day")

    def test_goodbye(self):
        assert self._matches("Goodbye!")
        assert self._matches("Bye now")
        assert self._matches("good bye")

    def test_see_you(self):
        assert self._matches("See you later")

    def test_thanks_for_calling_NOT_matched(self):
        # Critical: thanks-for-calling is a GREETING phrase. Must not trigger
        # the deduper or the bot will go silent immediately.
        assert not self._matches("Thanks for calling Smith Family Dental")

    def test_assistant_question_NOT_matched(self):
        assert not self._matches("What's your name?")

    def test_message_saved_NOT_matched(self):
        assert not self._matches("Got it, your callback is saved.")


# ---------- TextNormalizer.LEAKED_TOKENS ----------

class TestTextNormalizerTokens:
    def test_known_chat_template_tokens_present(self):
        # If someone deletes a token, leaked control tokens will show up
        # in TTS as garbled audio. Pin the list.
        leaked = bot.TextNormalizer.LEAKED_TOKENS
        for required in ("<|im_start|>", "<|im_end|>", "<|eot_id|>"):
            assert required in leaked, f"missing {required!r}"


# ---------- BiasedWhisperSTT bias prompt ----------

class TestWhisperBiasPrompt:
    def test_prompt_includes_digits(self):
        # Critical: the bias prompt is what stops "2" -> "True" hallucinations.
        p = bot.WHISPER_BIAS_PROMPT.lower()
        for digit_word in ("two", "four", "eight", "ten"):
            assert digit_word in p, f"bias prompt missing {digit_word!r}"

    def test_prompt_includes_yes_no(self):
        p = bot.WHISPER_BIAS_PROMPT.lower()
        assert "yes" in p and "no" in p

    def test_prompt_mentions_phone_call(self):
        # The "this is a phone call" framing matters for Whisper's prior.
        p = bot.WHISPER_BIAS_PROMPT.lower()
        assert "phone" in p


# ---------- SYSTEM_PROMPT contract ----------

class TestSystemPromptContract:
    """Lock in essential SYSTEM_PROMPT properties so prompt edits don't
    silently drop a critical rule."""

    def test_mentions_practice(self):
        assert "Smith Family Dental" in bot.SYSTEM_PROMPT

    def test_phone_words_rule_present(self):
        # Essential: prevents '201-388-2149' from streaming through TTS
        # in chunks the regex can't match. Also forbids the model from
        # defaulting to an example number (a regression we observed).
        s = bot.SYSTEM_PROMPT.lower()
        assert "spell each digit" in s and "as a separate word" in s
        assert "never invent or default to a phone number" in s

    def test_no_calendar_rule_present(self):
        # Essential: prevents the model from inventing slot times.
        s = bot.SYSTEM_PROMPT.lower()
        assert "do not have access to the office calendar" in s

    def test_save_request_tool_named(self):
        # Catches tool-name drift if someone renames it.
        assert "save_request" in bot.SYSTEM_PROMPT

    def test_no_text_tool_call_rule(self):
        # The MalformedToolCallStripper is the safety net; this prompt
        # rule is the first line of defense.
        s = bot.SYSTEM_PROMPT
        assert "tool_call" in s.lower()
        assert "structured function-calling" in s.lower()
