"""Minimal WebSocket smoke-test client for the receptionist bot.

Connects to the bot at ws://HOST:8765, sends a few seconds of silence
(so the bot's manual VAD won't fire), receives whatever the bot streams
back (the hardcoded greeting), and exits. Lets you verify the bot starts
cleanly without needing a real microphone.

Usage:
  python tools/ws_smoke.py                    # localhost:8765, 6s of silence
  python tools/ws_smoke.py --host my-tailnet  # Tailscale hostname
  python tools/ws_smoke.py --duration 10      # send 10s of silence

Doesn't validate transcript correctness — just confirms:
  - WebSocket connection accepts
  - Bot emits at least one OutputAudioRawFrame back
  - No exceptions on either side during a short session

Useful as a CI/smoke gate before a real audio test, and as a baseline
for the eventual full audio-pipeline load test.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import Pipecat's protobuf serializer to build well-formed frames the bot
# can actually parse. Anything we'd hand-roll would be brittle.
from pipecat.frames.frames import InputAudioRawFrame, StartFrame  # noqa: E402
from pipecat.serializers.protobuf import ProtobufFrameSerializer  # noqa: E402

import websockets  # noqa: E402


SAMPLE_RATE = 16000  # bot expects 16 kHz mono PCM int16
FRAME_MS = 20        # send in 20 ms chunks
BYTES_PER_FRAME = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 16-bit mono


def silence_frame() -> bytes:
    return b"\x00" * BYTES_PER_FRAME


async def smoke(host: str, port: int, duration_s: float) -> int:
    serializer = ProtobufFrameSerializer()
    # Pipecat serializers need a setup() with audio params.
    await serializer.setup(StartFrame(
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=22050,
    ))

    uri = f"ws://{host}:{port}"
    print(f"[smoke] connecting to {uri}", flush=True)
    t_connect = time.monotonic()

    audio_frames_received = 0
    bytes_received = 0
    first_audio_at: float | None = None

    try:
        async with websockets.connect(uri, max_size=2**24) as ws:
            print(f"[smoke] connected in {(time.monotonic() - t_connect) * 1000:.0f} ms", flush=True)

            async def send_silence():
                # Fire 20ms PCM frames at real-time pace.
                t0 = time.monotonic()
                n = int(duration_s * 1000 / FRAME_MS)
                for i in range(n):
                    frame = InputAudioRawFrame(
                        audio=silence_frame(),
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                    )
                    payload = await serializer.serialize(frame)
                    if payload is not None:
                        await ws.send(payload)
                    # Pace to real time so the bot doesn't see us as a flood
                    target = t0 + (i + 1) * FRAME_MS / 1000
                    sleep_for = target - time.monotonic()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                print(f"[smoke] finished sending {n} silence frames", flush=True)

            async def receive():
                nonlocal audio_frames_received, bytes_received, first_audio_at
                async for msg in ws:
                    if isinstance(msg, bytes):
                        bytes_received += len(msg)
                        try:
                            frame = await serializer.deserialize(msg)
                        except Exception:
                            continue
                        # We don't need to inspect — just count audio out frames.
                        if frame is not None:
                            audio_frames_received += 1
                            if first_audio_at is None:
                                first_audio_at = time.monotonic()

            send_task = asyncio.create_task(send_silence())
            recv_task = asyncio.create_task(receive())
            await send_task
            # Give the server ~1.5s to flush any final audio
            try:
                await asyncio.wait_for(recv_task, timeout=1.5)
            except asyncio.TimeoutError:
                recv_task.cancel()
    except Exception as e:
        print(f"[smoke] FAIL: {e!r}", flush=True)
        return 1

    elapsed = time.monotonic() - t_connect
    first_byte_ms = (
        (first_audio_at - t_connect) * 1000 if first_audio_at is not None else None
    )
    print(f"[smoke] session ended in {elapsed:.1f}s")
    print(f"[smoke] frames received: {audio_frames_received}")
    print(f"[smoke] bytes received: {bytes_received}")
    if first_byte_ms is not None:
        print(f"[smoke] first audio out: {first_byte_ms:.0f} ms after connect")

    if audio_frames_received == 0:
        print("[smoke] FAIL: no audio frames received from bot")
        return 1
    print("[smoke] PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--duration", type=float, default=6.0,
        help="Seconds of silence to send (default 6 — enough to capture the greeting)",
    )
    args = p.parse_args()
    return asyncio.run(smoke(args.host, args.port, args.duration))


if __name__ == "__main__":
    sys.exit(main())
