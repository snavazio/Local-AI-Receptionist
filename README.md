# Local AI Phone Receptionist

A fully local AI receptionist for a dental office, built on [Pipecat 1.1](https://github.com/pipecat-ai/pipecat).  
It answers calls, holds a natural conversation, collects appointment-callback requests, takes messages, and escalates emergencies — **all running on a single workstation with no cloud APIs**.

---

## Why this exists

Voice-agent demos are easy when you're allowed to lean on cloud APIs (OpenAI, Deepgram, ElevenLabs).  The interesting question is whether a *local-only* stack — running on hardware a small business could actually own — is good enough for a real-world front-desk job.

This project is the working answer: **yes**, with the right model choices and some defensive scaffolding.

---

## What it does

A caller dials in over WebSocket.  **Sarah**, the AI receptionist, picks up:

1. Greets the caller and identifies as an AI assistant.
2. Collects name, callback phone number, and preferred day/time for an appointment, **one slot at a time**.
3. Reads the phone number back **digit-by-digit** for confirmation.
4. Saves the request as a JSON file to `call_logs/` — a real human staff member calls back to confirm actual scheduling.  The bot **never claims to see a calendar**.
5. Can also take messages for the doctor or escalate dental emergencies (severe pain, swelling, bleeding, trauma) to the office's emergency line.
6. Says goodbye **exactly once**.

Every successful call leaves up to three artifacts: a `callback_*.json` (or `message_*.json` / `emergency_*.json`), and a `call_*.json` with the full transcript.

---

## Stack

All local, all runs on a 24 GB GPU:

| Layer | Component |
|---|---|
| LLM | Qwen 2.5 14B via [Ollama](https://ollama.ai) (excellent tool-caller) |
| STT | faster-whisper `distil-large-v3` on CUDA fp16 |
| TTS | Piper `en_US-lessac-medium` @ 22 kHz |
| Framework | Pipecat 1.1 |
| Transport | WebSocket + Protobuf serialiser |
| VAD | Custom RMS-energy detector (Silero wasn't firing) |

### Hardware tested

- Linux (Ubuntu 24.04), RTX 5090 mobile, 24 GB VRAM
- Cross-continent test client over Tailscale (~190 ms RTT) — works fine

### Models tried before settling on Qwen 2.5 14B

| Model | Issue |
|---|---|
| Llama 3.1/3.2 | Too tool-trigger-happy; calls tools with placeholder args |
| Hermes 3 8B | Great prose; refused to call tools under negative-gating prompts |

---

## Engineering notes

The stack itself is unsurprising.  The interesting work was making local models behave well enough on a phone:

### `BiasedWhisperSTT`

Without an `initial_prompt` biasing toward digits / days / yes-no, `distil-large-v3` hallucinates short utterances ("2" → "True.").  
`BiasedWhisperSTT` (in `processors/biased_whisper.py`) is a thin subclass of `WhisperSTTService` that passes a dental-office-flavoured prompt on every transcription call.

### Positive-voice system prompt

Heavy "MUST / DO NOT / FAILURE" language broke tool-calling on Hermes and made Qwen overly cautious.  Plain slot-filling instructions work.

### Hardcoded greeting

Even with "say EXACTLY this" instructions, the LLM dropped words from the greeting ("this is Sarah, the AI.").  
Bypassing the LLM with a `TTSSpeakFrame` on connect removed the variability.

### `ForcedSpeechOverride`

When a tool returns a deterministic confirmation prompt, this processor speaks it once and suppresses the LLM's chatty multi-sentence follow-up — otherwise the caller hears the same confirmation 3–4 times.

Implementation: tool handlers set `call_state.forced_speech_text`; `ForcedSpeechOverride` detects the next `LLMFullResponseStartFrame`, emits a `TTSSpeakFrame` with that text, and drops all `LLMTextFrame` tokens for that turn.

### `FarewellDeduper`

Latches on the first farewell phrase and drops everything after.  The caller saying "goodbye" back doesn't trigger another bot farewell.

### `TextNormalizer`

The LLM is instructed to spell phone numbers as words, but some digit-form numbers slip through.  A regex-based processor converts patterns like `555-867-5309` → `five five five eight six seven five three zero nine` before the text reaches Piper.

### No fake calendar

The system prompt explicitly forbids the model from inventing time slots ("I have 2 PM and 4 PM available") — it collects a *requested* time and a human handles actual availability.

---

## What this is not

- **Not connected to the PSTN.**  This bot speaks over WebSocket.  To take real phone calls you'd bridge it through Twilio Programmable Voice, FreeSWITCH, or similar — Pipecat has integrations for that.
- **Not a calendar system.**  Bookings are written to JSON; a human (or your existing scheduling system) does the actual confirmation.
- **Not a state machine.**  It's still LLM-driven turn-taking with tool calls, layered with defensive validators.  A more robust production version would use [pipecat-flows](https://github.com/pipecat-ai/pipecat-flows) for explicit FSM-driven slot filling.

---

## Setup

### 1 — Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| [Ollama](https://ollama.ai) | `ollama pull qwen2.5:14b` |
| CUDA GPU (24 GB recommended) | CPU fallback works but is slow |
| Piper voice model files | Downloaded automatically on first run |

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure

```bash
cp .env.example .env
# Edit .env — at minimum set WHISPER_DEVICE and PIPER_VOICE
```

### 4 — Run

```bash
python bot.py
```

The server starts on `0.0.0.0:8765` (configurable via `WS_PORT`).

### 5 — Test client

Use any WebSocket client that sends/receives Pipecat Protobuf frames.  
The official [Pipecat client SDK](https://github.com/pipecat-ai/pipecat-client-web) works out of the box.

---

## Project layout

```
.
├── bot.py                   # Main pipeline and WebSocket server
├── tools.py                 # Tool definitions + handlers (callback / message / emergency)
├── processors/
│   ├── vad.py               # RMSEnergyVAD — RMS-energy voice activity detector
│   ├── biased_whisper.py    # BiasedWhisperSTT — Whisper + dental bias prompt
│   ├── forced_speech.py     # ForcedSpeechOverride — speak tool result, suppress LLM follow-up
│   ├── farewell.py          # FarewellDeduper — prevent duplicate goodbyes
│   └── text_normalizer.py   # TextNormalizer — digit strings → spoken words
├── call_logs/               # Runtime JSON logs (git-ignored)
├── voice_models/            # Piper model files (git-ignored, downloaded on first run)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Call log format

### `call_logs/callback_<ts>.json`

```json
{
  "type": "callback",
  "call_id": "...",
  "created_at": "2025-01-01T12:00:00+00:00",
  "caller_name": "Jane Smith",
  "callback_phone": "555-867-5309",
  "preferred_day": "Tuesday",
  "preferred_time": "morning"
}
```

### `call_logs/message_<ts>.json`

```json
{
  "type": "message",
  "call_id": "...",
  "created_at": "2025-01-01T12:05:00+00:00",
  "caller_name": "John Doe",
  "message": "Please call me back about my prescription."
}
```

### `call_logs/emergency_<ts>.json`

```json
{
  "type": "emergency",
  "call_id": "...",
  "created_at": "2025-01-01T12:10:00+00:00",
  "caller_name": "Alex Brown",
  "situation": "Severe tooth pain and swelling since this morning"
}
```

### `call_logs/call_<ts>.json`

```json
{
  "call_id": "...",
  "ended_at": "2025-01-01T12:15:00+00:00",
  "transcript": [
    {"role": "user", "text": "Hi I need to make an appointment", "ts": "..."},
    {"role": "assistant", "text": "Of course! May I have your name?", "ts": "..."}
  ]
}
```

---

## License & attribution

MIT — see [LICENSE](LICENSE).

Forked from the [Pipecat phone-bot quickstart](https://github.com/pipecat-ai/pipecat).
