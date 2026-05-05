"""Custom RMS-energy Voice Activity Detector.

Silero's model was not reliably firing for short phone-quality utterances in
testing (possibly due to sample-rate or chunk-size mismatches). This simpler
detector uses root-mean-square energy to decide whether audio is speech.

It is a drop-in replacement for SileroVADAnalyzer — pass it to
``VADProcessor(vad_analyzer=RMSEnergyVAD())`` in the pipeline.
"""

import numpy as np

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams


# ---------------------------------------------------------------------------
# Default tuning knobs
# ---------------------------------------------------------------------------

# Frame duration targeted by the VAD (milliseconds).  20 ms is a common
# choice: short enough to be responsive, long enough for reliable energy
# estimation at 16 kHz.
_FRAME_MS = 20

# RMS amplitude threshold below which audio is considered silence.  The
# value is expressed on the int16 PCM scale [0, 32768].  ~600 corresponds
# to roughly –34 dBFS — well above thermal noise but below typical speech.
_DEFAULT_RMS_THRESHOLD = 600.0


class RMSEnergyVAD(VADAnalyzer):
    """Voice activity detector based on root-mean-square frame energy.

    Computes per-frame RMS and compares it against a configurable threshold.
    The Pipecat base class (``VADAnalyzer``) handles the state machine
    (QUIET → STARTING → SPEAKING → STOPPING) and the hysteresis timers;
    this subclass only needs to supply a ``voice_confidence`` score.

    Args:
        rms_threshold: Int16-scale amplitude at or above which a frame is
            considered voiced.  Increase to reduce false positives in noisy
            environments; decrease for quiet callers.
        sample_rate: Expected audio sample rate in Hz.  Defaults to 16 000 Hz.
        params: Pipecat ``VADParams`` (start/stop hysteresis, min_volume,
            confidence threshold).
    """

    def __init__(
        self,
        *,
        rms_threshold: float = _DEFAULT_RMS_THRESHOLD,
        sample_rate: int = 16_000,
        params: VADParams | None = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, params=params or VADParams())
        self._rms_threshold = rms_threshold

    # ------------------------------------------------------------------
    # VADAnalyzer interface
    # ------------------------------------------------------------------

    def num_frames_required(self) -> int:
        """Return the number of PCM samples required per analysis chunk.

        We target ``_FRAME_MS`` milliseconds worth of mono 16-bit samples.
        """
        return int(self._sample_rate * _FRAME_MS / 1000)

    def voice_confidence(self, buffer: bytes) -> float:
        """Return a [0, 1] voiced-speech confidence score for *buffer*.

        The score is 1.0 when RMS ≥ ``rms_threshold``, and 0.0 otherwise.
        The Pipecat base class compares this against ``params.confidence``
        (default 0.7) to decide whether the frame counts as voiced.

        Args:
            buffer: Raw audio bytes — mono, 16-bit signed PCM, little-endian.

        Returns:
            1.0 if the frame energy exceeds the threshold, else 0.0.
        """
        samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))
        return 1.0 if rms >= self._rms_threshold else 0.0
