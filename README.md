# Local AI Receptionist

A fully local AI receptionist for a dental office, built on [Pipecat](https://github.com/pipecat-ai/pipecat). It answers calls, holds a natural conversation, collects appointment-callback requests, takes messages, and escalates emergencies — all running on a single workstation with **no cloud APIs**. This is a path for regulated industries to have required control over the data. This can be useful for Legal, Financial and Health Care industries(HIPPA).

## Why this exists

Voice-agent demos are easy when you're allowed to lean on cloud APIs (OpenAI, Deepgram, ElevenLabs). The interesting question is whether a *local-only* stack — running on hardware a small business could actually own — is good enough for a real-world front-desk job.

This project is the working answer: yes, with the right model choices and some defensive scaffolding.

## What it does

A caller dials in over WebSocket. Sarah, the AI receptionist, picks up:

- Greets the caller and identifies as an AI assistant.
- Collects name, callback phone number, and preferred day/time for an appointment, one slot at a time.
- Reads the phone number back digit-by-digit for confirmation.
- Saves the request as a JSON file to `call_logs/` — a real human staff member calls back to confirm actual scheduling. The bot never claims to see a calendar.
- Can also take messages for the doctor or escalate dental emergencies (severe pain, swelling, bleeding, trauma) to the office's emergency line.
- Says goodbye exactly once.

Every successful call leaves three artifacts: a `callback_*.json` (or `message_*.json`), and a `call_*.json` with the full transcript.

## Stack

All local, runs on a 24 GB GPU today; future testing will target a 16 GB GPU.

| Layer     | Component                                            |
| --------- | ---------------------------------------------------- |
| LLM       | **Qwen 2.5 14B** via Ollama (excellent tool-caller)  |
| STT       | **faster-whisper distil-large-v3** on CUDA fp16      |
| TTS       | **Piper en_US-lessac-medium** @ 22 kHz               |
| Framework | **Pipecat 1.1**                                      |
| Transport | WebSocket + Protobuf serializer                      |
| VAD       | Custom RMS-energy detector (Silero wasn't firing)    |

## Hardware tested

- Linux (Ubuntu 24.04), RTX 5090 mobile, 24 GB VRAM
- Cross-continent test client over Tailscale (~190 ms RTT) — works fine
- Models tried before settling on Qwen 2.5 14B: Llama 3.1 / 3.2 (too tool-trigger-happy, calls tools with placeholder args), Hermes 3 8B (great prose, refused to call tools under negative-gating prompts)

## Capacity & latency

Measured via the eval harness (`eval/run_eval.py`) on the hardware above:

| Concurrent inference slots | LLM-call p50 | LLM-call p95 | Throughput |
| --- | --- | --- | --- |
| 1 | 663 ms | 2.7 s | 0.94 calls/s |
| 2 | 1.6 s | 3.4 s | 1.07 calls/s |
| 4 | 2.9 s | 5.3 s | 1.14 calls/s |
| 8 | 5.8 s | 11.6 s | 1.15 calls/s |

Throughput plateaus around concurrency 2 — the GPU is compute-bound, not request-bound. Per-call latency scales roughly linearly with concurrency.

A real caller's turn includes ~600 ms of VAD silence wait, ~150 ms Whisper, and ~80 ms Piper on top of the LLM call, so add ~800 ms to each row to get end-to-end turn latency.

A caller is bursty — they only need the GPU during their *own* turn. Practical capacity on this hardware:

- **~2 inference slots ≈ 4-6 simultaneous callers** (comfortable latency)
- **~4 slots ≈ 8-10 callers** (workable but slower)
- **More than ~10 callers** in flight will exceed acceptable phone latency.

If you need more capacity, swap to Qwen 2.5 7B (~5 GB VRAM) for ~2x the parallel slots in the same VRAM budget — at the cost of ~10-15 percentage points on the eval pass rate.

## Eval / regression suite

`eval/` is a text-only harness that drives Qwen via Ollama's OpenAI-compat endpoint and asserts on per-case behavior across 100 scripted scenarios in 10 categories (happy_path, slot_order, vague_response, correction, cancel, message, emergency, adversarial, phone_variant, office_qa). It mirrors the production-side post-processors (`MalformedToolCallStripper`, `FarewellDeduper`) so assertions reflect what a caller actually hears.

```bash
# 100 cases at concurrency 4 (~7 min on the hardware above)
python eval/run_eval.py --concurrency 4

# One category, single-threaded
python eval/run_eval.py --category emergency

# Try a different local model
python eval/run_eval.py --concurrency 4 --model qwen2.5:7b
```

Current baseline: **80/100** on Qwen 2.5 14B Q4. The remaining gap is concentrated in the message-taking flow (~6/10 fail) where the LLM gathers slots correctly but drifts away from invoking the tool — this is what `bot_flows.py` exists to fix.

### Regression watcher

`eval/watch.py` runs the eval, compares against the previous accepted run, and emits a focused diff report. Cron-friendly: it exits non-zero when a case newly fails or LLM p95 spikes by >1s, so a scheduler can flag the owner.

```bash
# First run: establishes the baseline.
python eval/watch.py

# Subsequent runs: compares to baseline, updates only if no regression.
python eval/watch.py

# Force a new baseline after an intentional change.
python eval/watch.py --update-baseline

# Compare without overwriting (e.g. CI).
python eval/watch.py --no-update
```

State files (gitignored):

- `eval/baseline.json` — last accepted run
- `eval/history.jsonl` — append-only trend log
- `eval/regression_report.md` — most recent diff report

After a few runs, glance at the trend with `python eval/trend.py`:

```
Pass rate (%):
   ▂▂▄▄▆▆█  range [70, 85]
  current: 85%   first shown: 70%

LLM p50 (ms):
   ▁▂▃▅▆▇█  range [1500, 1710]
  current: 1710 ms   first shown: 1500 ms
```

Per-category sparklines and a tabular last-10 view help spot drift.

Schedule it from cron / systemd timer / GitHub Actions. Example crontab line for a nightly 2 AM run that posts the report on regression:

```cron
0 2 * * * cd /path/to/Local-AI-Receptionist && .venv/bin/python eval/watch.py >> eval/cron.log 2>&1 || mail -s "Receptionist regression" you@example.com < eval/regression_report.md
```

## Two bot variants

- **`bot.py`** — the live, free-form-LLM-with-tool-calling implementation. What the README setup steps run.
- **`bot_flows.py`** — same audio pipeline, but conversation is driven by [pipecat-ai-flows](https://github.com/pipecat-ai/pipecat-flows) as an explicit FSM (`triage → collect_name → collect_phone → collect_window | collect_message → save → confirm/end`). The function call *is* the state transition, so the model can't "talk without acting." Use this when message-flow reliability matters more than conversational latitude. Same Ollama / Whisper / Piper stack — just a different driver.

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/snavazio/Local-AI-Receptionist.git
cd Local-AI-Receptionist
uv sync   # or: python -m venv .venv && pip install -e .
```

### 2. Install and pull the LLM via Ollama

```bash
# https://ollama.com/download
ollama pull qwen2.5:14b
```

### 3. Download the Piper voice weights

The Piper TTS voice (~60 MB) is **not** committed to this repo. Download it once into the project root:

```bash
curl -L -o en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Other Piper voices live at <https://huggingface.co/rhasspy/piper-voices>. If you swap, update the voice name in `bot.py` (`PiperTTSService.Settings(voice=...)`).

### 4. Run

```bash
python bot.py
```

The bot starts a WebSocket server on `0.0.0.0:8765`. Connect from any Pipecat-compatible WebSocket client (the upstream [pipecat-quickstart-phone-bot](https://github.com/pipecat-ai/pipecat-quickstart-phone-bot) ships a reference client).

## Engineering notes

The stack itself is unsurprising. The interesting work was making local models behave well enough on a phone:

- **Whisper bias prompt + VAD filter.** Without an `initial_prompt` biasing toward digits / days / yes-no, distil-large-v3 hallucinates short utterances ("2" → "True."). The `BiasedWhisperSTT` subclass passes both.
- **Positive-voice system prompt.** Heavy "MUST / DO NOT / FAILURE" language broke tool-calling on Hermes and made Qwen overly cautious. Plain slot-filling instructions work.
- **Hardcoded greeting.** Even with "say EXACTLY this" instructions, the LLM sometimes dropped words from the greeting (`"this is Sarah, the AI."`). Bypassing the LLM with a `TTSSpeakFrame` on connect removed the variable.
- **`ForcedSpeechOverride`.** When a tool returns a deterministic confirmation prompt, this processor speaks it once and suppresses the LLM's chatty multi-sentence follow-up — otherwise the caller hears the same confirmation 3-4 times.
- **`FarewellDeduper`.** Latches on the first farewell phrase and drops everything after. The caller saying "goodbye" back doesn't trigger another bot farewell.
- **Phone-number rules.** The LLM is instructed to spell phone numbers as words, and a `TextNormalizer` converts any digit-form numbers that slip through into Piper-friendly spoken digits.
- **No fake calendar.** The system prompt explicitly forbids the model from inventing time slots ("I have 2 PM and 4 PM available") — it collects a *requested* time and a human handles availability.

## What this is not

- **Not connected to the PSTN.** This bot speaks over WebSocket. To take real phone calls you'd bridge it through Twilio Programmable Voice, FreeSWITCH, or similar — Pipecat has integrations for that.
- **Not a calendar system.** Bookings are written to JSON; a human (or your existing scheduling system) does the actual confirmation.
- **Not a state machine.** It's still LLM-driven turn taking with tool calls, layered with defensive validators. A more robust production version would use [pipecat-flows](https://github.com/pipecat-ai/pipecat-flows) for explicit FSM-driven slot filling.

## License & attribution

Forked from the [Pipecat phone-bot quickstart](https://github.com/pipecat-ai/pipecat-quickstart-phone-bot).
