"""Generate WAV fixtures for the audio-level eval.

Uses the Piper voice that's already part of the bot stack to synthesize
each user-turn utterance from eval/cases.yaml into a WAV file. This gives
us a deterministic audio dataset for testing the full audio pipeline
(STT + LLM + TTS) end-to-end.

Output: tests/audio_fixtures/<case_id>/<turn_idx>.wav
        tests/audio_fixtures/<case_id>/<turn_idx>.txt   (the source text)

These fixtures are gitignored — regenerate locally on first checkout.

Usage:
  python tools/gen_audio_fixtures.py                 # all cases
  python tools/gen_audio_fixtures.py --case CASE_ID  # single case
  python tools/gen_audio_fixtures.py --dry-run       # report what would be generated
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--out", default=str(ROOT / "tests" / "audio_fixtures"))
    parser.add_argument("--case", help="Generate only this case_id")
    parser.add_argument("--voice-onnx", default=str(ROOT / "en_US-lessac-medium.onnx"))
    parser.add_argument("--voice-json", default=str(ROOT / "en_US-lessac-medium.onnx.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = yaml.safe_load(Path(args.cases).read_text())
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case {args.case!r}", file=sys.stderr)
            return 2

    out_root = Path(args.out)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    total_turns = sum(len(c.get("user_turns", [])) for c in cases)
    print(f"[gen] {len(cases)} cases, {total_turns} total turns")

    if args.dry_run:
        print("[gen] dry-run — no audio generated")
        return 0

    # Lazy-load Piper so dry-run doesn't pay the import cost.
    from piper import PiperVoice
    print(f"[gen] loading voice {args.voice_onnx}...")
    voice = PiperVoice.load(args.voice_onnx, config_path=args.voice_json)
    sr = voice.config.sample_rate
    print(f"[gen] voice loaded; sample rate {sr} Hz")

    written = 0
    for case in cases:
        case_id = case["id"]
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for i, turn in enumerate(case.get("user_turns", [])):
            wav_path = case_dir / f"{i:02d}.wav"
            txt_path = case_dir / f"{i:02d}.txt"
            if wav_path.exists():
                continue
            txt_path.write_text(turn)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                # synthesize() yields AudioChunk objects
                for chunk in voice.synthesize(turn):
                    wf.writeframes(chunk.audio_int16_bytes)
            written += 1
            if written % 25 == 0:
                print(f"[gen] {written} files written...", flush=True)

    print(f"[gen] done — {written} WAV files written under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
