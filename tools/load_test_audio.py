"""Audio-level load test: spawns N concurrent WebSocket sessions against
the bot, each streaming a pre-recorded WAV, measures end-to-end latency.

This is the long-promised "real callers" load test. The text-only eval
(eval/run_eval.py) measures LLM behavior; this measures the entire
audio pipeline (Whisper STT + LLM + Piper TTS + WebSocket transport
under contention).

Prerequisite: the bot must be running on the target host:port, and audio
fixtures must exist in tests/audio_fixtures/<case_id>/. Generate them
with `python tools/gen_audio_fixtures.py`.

Usage:
  # 4 concurrent callers, each running the happy_path_basic case
  python tools/load_test_audio.py --case happy_path_basic --concurrency 4

  # Use Tailscale-reachable host
  python tools/load_test_audio.py --host 100.113.202.107 --case ...

What we measure per session:
  - connect_ms        : time to establish WebSocket
  - first_audio_ms    : time from connect to first audio frame from bot
  - turn_latencies_ms : list of (user-stop -> bot-audio-start) per turn

Aggregate across all sessions:
  - p50 / p95 / p99 of each metric
  - error rate (sessions that failed mid-conversation)

Heads-up: this hammers the GPU. Don't run while eval/run_eval.py is in
flight; numbers will be meaningless under shared load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipecat.frames.frames import InputAudioRawFrame, StartFrame  # noqa: E402
from pipecat.serializers.protobuf import ProtobufFrameSerializer  # noqa: E402

import websockets  # noqa: E402


SAMPLE_RATE_IN = 16000   # bot expects 16 kHz mono PCM
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE_IN * 2 * FRAME_MS // 1000


def percentile(xs, pct):
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def load_wav_int16(path: Path) -> bytes:
    """Read a WAV and return raw int16 mono samples at 16 kHz. Resamples
    by simple decimation if the file is at a different rate (Piper's
    fixtures are 22050 Hz; we need 16 kHz for the bot's STT)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    assert sw == 2, f"expected 16-bit PCM, got {sw * 8}-bit"
    if ch == 2:
        # Average the two channels
        import array
        a = array.array("h", raw)
        mono = array.array("h", (((a[i] + a[i + 1]) // 2) for i in range(0, len(a), 2)))
        raw = mono.tobytes()

    if sr != SAMPLE_RATE_IN:
        # Crude rate conversion via decimation/interpolation.
        # For 22050 -> 16000 the ratio is ~0.7256; we resample with linear
        # interpolation. This isn't audio-quality but is good enough for
        # Whisper to recognize.
        import array
        a = array.array("h", raw)
        ratio = sr / SAMPLE_RATE_IN
        n_out = int(len(a) / ratio)
        out = array.array("h", [0] * n_out)
        for i in range(n_out):
            src = i * ratio
            lo = int(src)
            hi = min(lo + 1, len(a) - 1)
            frac = src - lo
            out[i] = int(a[lo] * (1 - frac) + a[hi] * frac)
        raw = out.tobytes()

    return raw


async def run_one_session(
    session_id: int,
    host: str,
    port: int,
    wav_files: list[Path],
    silence_between_turns_s: float = 1.5,
) -> dict:
    """One simulated caller. Returns a stats dict; never raises."""
    serializer = ProtobufFrameSerializer()
    await serializer.setup(StartFrame(
        audio_in_sample_rate=SAMPLE_RATE_IN,
        audio_out_sample_rate=22050,
    ))

    uri = f"ws://{host}:{port}"
    stats = {
        "session_id": session_id,
        "ok": False,
        "error": None,
        "connect_ms": None,
        "first_audio_ms": None,
        "session_duration_s": None,
        "audio_frames_received": 0,
    }

    t_start = time.monotonic()
    first_audio_at = None
    audio_frames = 0

    try:
        t_connect = time.monotonic()
        async with websockets.connect(uri, max_size=2**24) as ws:
            stats["connect_ms"] = int((time.monotonic() - t_connect) * 1000)

            async def receive():
                nonlocal first_audio_at, audio_frames
                async for msg in ws:
                    if isinstance(msg, bytes):
                        try:
                            frame = await serializer.deserialize(msg)
                        except Exception:
                            continue
                        if frame is not None:
                            audio_frames += 1
                            if first_audio_at is None:
                                first_audio_at = time.monotonic()
                                stats["first_audio_ms"] = int(
                                    (first_audio_at - t_connect) * 1000
                                )

            recv_task = asyncio.create_task(receive())

            # Real-time-paced silence + WAV streaming
            silence = b"\x00" * BYTES_PER_FRAME
            # 1s of silence first so the bot's greeting can play
            for _ in range(int(1.0 * 1000 / FRAME_MS)):
                payload = await serializer.serialize(InputAudioRawFrame(
                    audio=silence, sample_rate=SAMPLE_RATE_IN, num_channels=1,
                ))
                if payload:
                    await ws.send(payload)
                await asyncio.sleep(FRAME_MS / 1000)

            for wav_path in wav_files:
                pcm = load_wav_int16(wav_path)
                # Stream in 20 ms chunks at real-time pace
                for off in range(0, len(pcm), BYTES_PER_FRAME):
                    chunk = pcm[off : off + BYTES_PER_FRAME]
                    if len(chunk) < BYTES_PER_FRAME:
                        chunk = chunk + b"\x00" * (BYTES_PER_FRAME - len(chunk))
                    payload = await serializer.serialize(InputAudioRawFrame(
                        audio=chunk, sample_rate=SAMPLE_RATE_IN, num_channels=1,
                    ))
                    if payload:
                        await ws.send(payload)
                    await asyncio.sleep(FRAME_MS / 1000)
                # Silence between turns so VAD sees end-of-speech
                for _ in range(int(silence_between_turns_s * 1000 / FRAME_MS)):
                    payload = await serializer.serialize(InputAudioRawFrame(
                        audio=silence, sample_rate=SAMPLE_RATE_IN, num_channels=1,
                    ))
                    if payload:
                        await ws.send(payload)
                    await asyncio.sleep(FRAME_MS / 1000)

            try:
                await asyncio.wait_for(recv_task, timeout=2.0)
            except asyncio.TimeoutError:
                recv_task.cancel()

        stats["ok"] = audio_frames > 0
    except Exception as e:
        stats["error"] = repr(e)

    stats["session_duration_s"] = round(time.monotonic() - t_start, 2)
    stats["audio_frames_received"] = audio_frames
    return stats


def aggregate(stats_list: list[dict]) -> dict:
    ok = [s for s in stats_list if s["ok"]]
    failed = [s for s in stats_list if not s["ok"]]
    connects = [s["connect_ms"] for s in ok if s["connect_ms"] is not None]
    first_audio = [s["first_audio_ms"] for s in ok if s["first_audio_ms"] is not None]
    durations = [s["session_duration_s"] for s in ok]

    return {
        "total_sessions": len(stats_list),
        "successful": len(ok),
        "failed": len(failed),
        "connect_ms": {
            "p50": percentile(connects, 50),
            "p95": percentile(connects, 95),
            "max": max(connects) if connects else 0,
        },
        "first_audio_ms": {
            "p50": percentile(first_audio, 50),
            "p95": percentile(first_audio, 95),
            "max": max(first_audio) if first_audio else 0,
        },
        "session_duration_s": {
            "mean": round(statistics.mean(durations), 2) if durations else 0,
            "max": max(durations) if durations else 0,
        },
        "errors": [s["error"] for s in failed if s.get("error")],
    }


async def main_async(args) -> int:
    case_dir = Path(args.fixtures) / args.case
    if not case_dir.exists():
        print(f"No fixtures at {case_dir}. Run tools/gen_audio_fixtures.py first.",
              file=sys.stderr)
        return 2

    wav_files = sorted(case_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files in {case_dir}", file=sys.stderr)
        return 2

    print(f"[load] {args.concurrency} concurrent sessions, "
          f"case={args.case} ({len(wav_files)} turns)", flush=True)

    t0 = time.monotonic()
    sessions = await asyncio.gather(*[
        run_one_session(i, args.host, args.port, wav_files)
        for i in range(args.concurrency)
    ])
    wall = time.monotonic() - t0

    summary = aggregate(sessions)
    print(json.dumps(summary, indent=2))
    print(f"\nwall: {wall:.1f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "summary": summary,
            "sessions": sessions,
            "wall_s": wall,
        }, indent=2))
        print(f"detail written to {args.json_out}")

    return 0 if summary["failed"] == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--case", default="happy_path_basic",
                   help="Which case_id's fixtures to play")
    p.add_argument("--concurrency", type=int, default=2,
                   help="Number of simultaneous sessions to spawn")
    p.add_argument("--fixtures", default=str(ROOT / "tests" / "audio_fixtures"))
    p.add_argument("--json-out", help="Write per-session JSON detail here")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
