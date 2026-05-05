"""Unit tests for the pure-Python helpers in bot.py.

These don't need Ollama, audio, or the Pipecat pipeline — they test the
small functions that the eval would otherwise cover only indirectly. They
catch regressions in number/name/text normalization quickly (~50 ms total)
so a pre-commit hook can run them.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so 'import bot' works from tests/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


# ---------- extract_phone_digits ----------

class TestExtractPhoneDigits:
    def test_dashed_format(self):
        assert bot.extract_phone_digits("201-388-2149") == "2013882149"

    def test_dotted_format(self):
        assert bot.extract_phone_digits("415.555.0188") == "4155550188"

    def test_paren_area_code(self):
        assert bot.extract_phone_digits("(212) 555-0144") == "2125550144"

    def test_no_separators(self):
        assert bot.extract_phone_digits("5125550144") == "5125550144"

    def test_country_code(self):
        # 11 digits with leading 1 should still extract; downstream gates strip if needed
        assert bot.extract_phone_digits("+1 415-555-0102") == "14155550102"

    def test_word_form_simple(self):
        assert bot.extract_phone_digits(
            "two zero one, three eight eight, two one four nine"
        ) == "2013882149"

    def test_word_form_no_commas(self):
        assert bot.extract_phone_digits(
            "two zero one three eight eight two one four nine"
        ) == "2013882149"

    def test_word_form_oh_for_zero(self):
        assert bot.extract_phone_digits(
            "five oh four, five five five, oh one oh two"
        ) == "5045550102"

    def test_mixed_words_and_digits(self):
        assert bot.extract_phone_digits(
            "Two oh one, 388, two one four nine"
        ) == "2013882149"

    def test_too_short(self):
        assert bot.extract_phone_digits("555-0144") == "5550144"  # 7 digits is allowed

    def test_way_too_short(self):
        assert bot.extract_phone_digits("123") is None

    def test_garbage_returns_none(self):
        assert bot.extract_phone_digits("hello world") is None

    def test_empty_returns_none(self):
        assert bot.extract_phone_digits("") is None
        assert bot.extract_phone_digits(None) is None  # type: ignore[arg-type]


# ---------- _looks_like_garbage_name ----------

class TestLooksLikeGarbageName:
    def test_real_name_passes(self):
        assert bot._looks_like_garbage_name("Steve") is False
        assert bot._looks_like_garbage_name("Maria Lopez") is False

    def test_url_artifact_caught(self):
        assert bot._looks_like_garbage_name("steve.com") is True
        assert bot._looks_like_garbage_name("http://foo") is True
        assert bot._looks_like_garbage_name("www.example") is True

    def test_email_caught(self):
        assert bot._looks_like_garbage_name("steve@dental") is True

    def test_xml_artifact_caught(self):
        assert bot._looks_like_garbage_name("<unknown>") is True


# ---------- _is_caller_affirming / declining ----------

class TestAffirmingDeclining:
    def test_affirming_yes(self):
        assert bot._is_caller_affirming("Yes")
        assert bot._is_caller_affirming("yeah")
        assert bot._is_caller_affirming("yep")
        assert bot._is_caller_affirming("that's right")
        assert bot._is_caller_affirming("correct")

    def test_declining_no(self):
        assert bot._is_caller_declining("No")
        assert bot._is_caller_declining("nope")
        assert bot._is_caller_declining("bye")
        assert bot._is_caller_declining("goodbye")

    def test_uncertain_neither(self):
        assert not bot._is_caller_affirming("hmm")
        assert not bot._is_caller_declining("hmm")


# ---------- _missing (placeholder + slot validation) ----------

class TestMissing:
    def test_all_present(self):
        args = {
            "caller_name": "Steve",
            "callback_number": "201-388-2149",
            "preferred_window": "Tuesday 2pm",
        }
        assert bot._missing(args, "caller_name", "callback_number", "preferred_window") == []

    def test_placeholder_name_caught(self):
        args = {"caller_name": "John Doe", "callback_number": "201-388-2149"}
        m = bot._missing(args, "caller_name", "callback_number")
        assert "caller_name" in m

    def test_placeholder_unknown_caught(self):
        args = {"caller_name": "<unknown>", "callback_number": "201-388-2149"}
        assert "caller_name" in bot._missing(args, "caller_name", "callback_number")

    def test_word_form_phone_accepted(self):
        # Critical regression test: word-form numbers should not trip _missing's
        # callback_number check now that it uses extract_phone_digits.
        args = {
            "caller_name": "Steve",
            "callback_number": "two zero one, three eight eight, two one four nine",
        }
        assert "callback_number" not in bot._missing(args, "caller_name", "callback_number")

    def test_short_name_caught(self):
        args = {"caller_name": "S"}
        assert "caller_name" in bot._missing(args, "caller_name")

    def test_missing_field(self):
        args = {"caller_name": "Steve"}
        assert "callback_number" in bot._missing(args, "caller_name", "callback_number")


# ---------- normalize_for_tts (phone-string -> spoken digits) ----------

class TestNormalizeForTTS:
    def test_full_number_normalized(self):
        out = bot.normalize_for_tts("Your number is 201-388-2149.")
        assert "two zero one" in out
        assert "three eight eight" in out
        assert "201-388-2149" not in out

    def test_no_phone_unchanged(self):
        s = "Hello, this is Sarah."
        assert bot.normalize_for_tts(s) == s


# ---------- speak_digits (low-level digit-to-words) ----------

class TestSpeakDigits:
    def test_ten_digits(self):
        assert bot.speak_digits("2013882149") == (
            "two zero one, three eight eight, two one four nine"
        )

    def test_eleven_digits_us(self):
        assert bot.speak_digits("12013882149").startswith("one,")

    def test_seven_digits(self):
        assert bot.speak_digits("3882149") == "three eight eight, two one four nine"


# ---------- chitchat detection ----------

class TestChitchat:
    def test_hi_is_chitchat(self):
        assert bot._is_known_chitchat("hi")
        assert bot._is_known_chitchat("hello")
        assert bot._is_known_chitchat("bye")

    def test_real_request_is_not(self):
        assert not bot._is_known_chitchat("I want to book an appointment")

    def test_empty_is_not(self):
        # Critical: empty input should NOT count as chitchat — that was a
        # past bug where empty strings blocked legitimate tool calls.
        assert not bot._is_known_chitchat("")


# ---------- placeholder normalization ----------

class TestPlaceholderNormalize:
    def test_strips_brackets(self):
        assert bot._normalize_placeholder("<unknown>") == "unknown"
        assert bot._normalize_placeholder("[name]") == "name"
        assert bot._normalize_placeholder("(steve)") == "steve"

    def test_lowercases(self):
        assert bot._normalize_placeholder("STEVE") == "steve"
