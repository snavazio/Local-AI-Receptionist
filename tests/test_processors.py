"""Unit tests for the Local AI Receptionist custom processors and tools.

These tests exercise the pure-Python / pure-logic parts of the codebase
without requiring CUDA, Ollama, Piper, or a live WebSocket connection.

Run with::

    pytest tests/
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Execute a coroutine in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# processors/vad.py
# ---------------------------------------------------------------------------


class TestRMSEnergyVAD:
    def _make_vad(self, threshold=600.0, sample_rate=16_000):
        from processors.vad import RMSEnergyVAD

        vad = RMSEnergyVAD(rms_threshold=threshold, sample_rate=sample_rate)
        vad.set_sample_rate(sample_rate)
        return vad

    def _silence_buffer(self, n_samples=320):
        """Return a buffer of pure silence (all zeros)."""
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    def _speech_buffer(self, amplitude=3000, n_samples=320):
        """Return a buffer whose RMS clearly exceeds the default threshold."""
        samples = (np.ones(n_samples, dtype=np.float32) * amplitude).astype(np.int16)
        return samples.tobytes()

    def test_silence_returns_zero_confidence(self):
        vad = self._make_vad()
        confidence = vad.voice_confidence(self._silence_buffer())
        assert confidence == 0.0

    def test_speech_returns_one_confidence(self):
        vad = self._make_vad()
        confidence = vad.voice_confidence(self._speech_buffer(amplitude=3000))
        assert confidence == 1.0

    def test_num_frames_required_is_positive(self):
        vad = self._make_vad()
        assert vad.num_frames_required() > 0

    def test_custom_threshold(self):
        vad_low = self._make_vad(threshold=100.0)
        vad_high = self._make_vad(threshold=10_000.0)
        buf = self._speech_buffer(amplitude=500)
        assert vad_low.voice_confidence(buf) == 1.0
        assert vad_high.voice_confidence(buf) == 0.0


# ---------------------------------------------------------------------------
# processors/text_normalizer.py
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def setup_method(self):
        from processors.text_normalizer import normalize_text

        self.normalize = normalize_text

    def test_plain_text_unchanged(self):
        assert self.normalize("Hello, how are you?") == "Hello, how are you?"

    def test_full_phone_with_dashes(self):
        result = self.normalize("555-867-5309")
        assert "five" in result
        assert "eight" in result
        assert "-" not in result

    def test_full_phone_with_dots(self):
        result = self.normalize("555.867.5309")
        assert "five" in result
        assert "." not in result

    def test_digit_run_replaced(self):
        result = self.normalize("5551234567")
        assert "five five five" in result

    def test_short_number_not_replaced(self):
        # 3-digit numbers like area codes alone should NOT be replaced
        # by the digit-run rule (only 7+ digit runs are replaced).
        result = self.normalize("555")
        # The short run is not matched by _DIGIT_RUN (requires 7-10 digits).
        assert result == "555"

    def test_isolated_digit_replaced(self):
        result = self.normalize("Press 2 for billing")
        assert "two" in result
        assert "2" not in result

    def test_isolated_digit_returns_string(self):
        result = self.normalize("2nd floor")
        assert isinstance(result, str)  # just check it doesn't raise


# ---------------------------------------------------------------------------
# processors/farewell.py
# ---------------------------------------------------------------------------


class TestFarewellDeduper:
    def _make_deduper(self):
        from processors.farewell import FarewellDeduper

        return FarewellDeduper()

    def test_contains_farewell_detects_goodbye(self):
        from processors.farewell import FarewellDeduper

        assert FarewellDeduper._contains_farewell("Goodbye, have a great day!")
        assert FarewellDeduper._contains_farewell("take care now")
        assert not FarewellDeduper._contains_farewell("Your appointment is confirmed.")

    def _simulate_llm_response(self, deduper, text):
        """Simulate a full LLM response (start → text tokens → end) and return
        the list of frames that were forwarded downstream."""
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection

        forwarded = []

        async def _run():
            # Monkey-patch push_frame to capture outputs.
            async def mock_push(frame, direction=FrameDirection.DOWNSTREAM):
                if direction == FrameDirection.DOWNSTREAM:
                    forwarded.append(frame)

            deduper.push_frame = mock_push

            await deduper.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            for token in text.split():
                await deduper.process_frame(
                    LLMTextFrame(text=token + " "), FrameDirection.DOWNSTREAM
                )
            await deduper.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        asyncio.get_event_loop().run_until_complete(_run())
        return forwarded

    def test_first_farewell_passes_through(self):
        deduper = self._make_deduper()
        frames = self._simulate_llm_response(deduper, "Goodbye have a great day")
        from pipecat.frames.frames import LLMTextFrame

        text_frames = [f for f in frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) > 0

    def test_second_farewell_suppressed(self):
        deduper = self._make_deduper()
        # First goodbye
        self._simulate_llm_response(deduper, "Goodbye have a great day")
        assert deduper._farewell_latch is True
        # Second goodbye (caller echoing back) — should be suppressed
        frames = self._simulate_llm_response(deduper, "Goodbye to you too")
        from pipecat.frames.frames import LLMTextFrame

        text_frames = [f for f in frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) == 0

    def test_normal_response_after_no_farewell_passes(self):
        deduper = self._make_deduper()
        frames = self._simulate_llm_response(deduper, "Your appointment is confirmed")
        from pipecat.frames.frames import LLMTextFrame

        text_frames = [f for f in frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) > 0
        assert deduper._farewell_latch is False


# ---------------------------------------------------------------------------
# processors/forced_speech.py
# ---------------------------------------------------------------------------


class TestForcedSpeechOverride:
    def _make_processor(self, state=None):
        from processors.forced_speech import ForcedSpeechOverride
        from tools import CallState

        if state is None:
            state = CallState(call_id="test-call")
        return ForcedSpeechOverride(call_state=state), state

    def _simulate_response(self, processor, state, forced_text=None):
        """Simulate an LLM response with optional forced_speech_text set on state."""
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            TTSSpeakFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection

        if forced_text:
            state.forced_speech_text = forced_text

        forwarded = []

        async def _run():
            async def mock_push(frame, direction=FrameDirection.DOWNSTREAM):
                if direction == FrameDirection.DOWNSTREAM:
                    forwarded.append(frame)

            processor.push_frame = mock_push

            await processor.process_frame(
                LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM
            )
            await processor.process_frame(
                LLMTextFrame(text="This is the LLM response. "), FrameDirection.DOWNSTREAM
            )
            await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        asyncio.get_event_loop().run_until_complete(_run())
        return forwarded

    def test_no_forced_speech_passes_through(self):
        proc, state = self._make_processor()
        frames = self._simulate_response(proc, state, forced_text=None)
        from pipecat.frames.frames import LLMTextFrame

        text_frames = [f for f in frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) > 0

    def test_forced_speech_emits_speak_frame(self):
        proc, state = self._make_processor()
        frames = self._simulate_response(proc, state, forced_text="We'll call you back!")
        from pipecat.frames.frames import TTSSpeakFrame

        speak_frames = [f for f in frames if isinstance(f, TTSSpeakFrame)]
        assert len(speak_frames) == 1
        assert speak_frames[0].text == "We'll call you back!"

    def test_forced_speech_suppresses_llm_text(self):
        proc, state = self._make_processor()
        frames = self._simulate_response(proc, state, forced_text="We'll call you back!")
        from pipecat.frames.frames import LLMTextFrame

        text_frames = [f for f in frames if isinstance(f, LLMTextFrame)]
        assert len(text_frames) == 0

    def test_forced_speech_text_cleared_after_use(self):
        proc, state = self._make_processor()
        self._simulate_response(proc, state, forced_text="Confirmed!")
        assert state.forced_speech_text is None


# ---------------------------------------------------------------------------
# tools.py — CallState and _speak_phone
# ---------------------------------------------------------------------------


class TestCallState:
    def test_initial_state(self):
        from tools import CallState

        cs = CallState(call_id="abc123", emergency_line="5-5-5-9-1-1")
        assert cs.call_id == "abc123"
        assert cs.emergency_line == "5-5-5-9-1-1"
        assert cs.forced_speech_text is None
        assert cs.transcript == []
        assert cs.call_ended is False


class TestSpeakPhone:
    def test_basic_phone(self):
        from tools import _speak_phone

        result = _speak_phone("5551234567")
        assert result == "five five five one two three four five six seven"

    def test_phone_with_dashes(self):
        from tools import _speak_phone

        result = _speak_phone("555-867-5309")
        assert result == "five five five eight six seven five three zero nine"

    def test_phone_with_parens(self):
        from tools import _speak_phone

        result = _speak_phone("(555) 867-5309")
        assert result == "five five five eight six seven five three zero nine"


class TestToolHandlers:
    """Smoke-test each tool handler: verify file creation and state mutations."""

    def _make_params(self, function_name, arguments, state):
        """Build a minimal FunctionCallParams mock."""
        params = MagicMock()
        params.function_name = function_name
        params.arguments = arguments
        params.tool_resources = state
        params.result_callback = AsyncMock()
        return params

    def test_save_callback_request_writes_file(self, tmp_path, monkeypatch):
        from tools import CallState, handle_save_callback_request

        monkeypatch.setattr("tools.CALL_LOGS_DIR", tmp_path)

        state = CallState(call_id="test-cb")
        params = self._make_params(
            "save_callback_request",
            {
                "caller_name": "Jane Smith",
                "callback_phone": "5551234567",
                "preferred_day": "Monday",
                "preferred_time": "morning",
            },
            state,
        )
        _run(handle_save_callback_request(params))

        files = list(tmp_path.glob("callback_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["caller_name"] == "Jane Smith"
        assert data["type"] == "callback"
        assert state.forced_speech_text is not None
        assert "Jane" in state.forced_speech_text

    def test_save_message_writes_file(self, tmp_path, monkeypatch):
        from tools import CallState, handle_save_message

        monkeypatch.setattr("tools.CALL_LOGS_DIR", tmp_path)

        state = CallState(call_id="test-msg")
        params = self._make_params(
            "save_message",
            {"caller_name": "John Doe", "message": "Please call me back."},
            state,
        )
        _run(handle_save_message(params))

        files = list(tmp_path.glob("message_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["caller_name"] == "John Doe"
        assert state.forced_speech_text is not None

    def test_escalate_emergency_writes_file(self, tmp_path, monkeypatch):
        from tools import CallState, handle_escalate_emergency

        monkeypatch.setattr("tools.CALL_LOGS_DIR", tmp_path)

        state = CallState(call_id="test-emrg", emergency_line="5-5-5-9-1-1")
        params = self._make_params(
            "escalate_emergency",
            {"caller_name": "Alex Brown", "situation": "Severe swelling"},
            state,
        )
        _run(handle_escalate_emergency(params))

        files = list(tmp_path.glob("emergency_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["situation"] == "Severe swelling"
        assert "5-5-5-9-1-1" in state.forced_speech_text
