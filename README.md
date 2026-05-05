Local AI Phone Receptionist

A fully local AI receptionist for a dental office, built on Pipecat. It answers calls, holds a natural conversation, collects appointment-callback requests, takes messages, and escalates emergencies — all running on a single workstation with no cloud APIs.

Why this exists

Voice-agent demos are easy when you're allowed to lean on cloud APIs (OpenAI, Deepgram, ElevenLabs). The interesting question is whether a local-only stack — running on hardware a small business could actually own — is good enough for a real-world front-desk job.

This project is the working answer: yes, with the right model choices and some defensive scaffolding.

What it does..

A caller dials in over WebSocket. Sarah, the AI receptionist, picks up:

Greets the caller and identifies as an AI assistant from a dental office.


Collects name, callback phone number, and preferred day/time for an appointment, one slot at a time.
Reads the phone number back digit-by-digit for confirmation.
Saves the request as a JSON file to call_logs/ — a real human staff member calls back to confirm actual scheduling. The bot never claims to see a calendar.
Can also take messages for the doctor or escalate dental emergencies (severe pain, swelling, bleeding, trauma) to the office's emergency line.
Says goodbye exactly once.
Every successful call leaves three artifacts: a callback_*.json (or message_*.json), and a call_*.json with the full transcript.

Stack
All local, all runs on a 24GB GPU, future tests will include running it on a 16GB GPU:

Layer	Component

LLM	Qwen 2.5 14B via Ollama (excellent tool-caller)

STT	faster-whisper distil-large-v3 on CUDA fp16

TTS	Piper en_US-lessac-medium @ 22kHz

Framework	Pipecat 1.1

Transport	WebSocket + Protobuf serializer

VAD	Custom RMS-energy detector (Silero wasn't firing)

Hardware tested

Linux (Ubuntu 24.04), RTX 5090 mobile, 24GB VRAM

Cross-continent test client over Tailscale (~190ms RTT) — works fine

Models tried before settling on Qwen 2.5 14B: Llama 3.1/3.2 (too tool-trigger-happy, calls tools with placeholder args), Hermes 3 8B (great prose, refused to call tools under negative-gating prompts)

Engineering notes

The stack itself is unsurprising. 
The interesting work was making local models behave well enough on a phone:

Whisper bias prompt + VAD filter. Without an initial_prompt biasing toward digits / days / yes-no, distil-large-v3 hallucinates short utterances ("2" → "True."). The BiasedWhisperSTT subclass passes both.
Positive-voice system prompt. Heavy "MUST / DO NOT / FAILURE" language broke tool-calling on Hermes and made Qwen overly cautious. Plain slot-filling instructions work.
Hardcoded greeting. Even with "say EXACTLY this" instructions, the LLM dropped words from the greeting ("this is Sarah, the AI."). Bypassing the LLM with a TTSSpeakFrame on connect removed the variable.
ForcedSpeechOverride. When a tool returns a deterministic confirmation prompt, this processor speaks it once and suppresses the LLM's chatty multi-sentence follow-up — otherwise the caller hears the same confirmation 3-4 times.
FarewellDeduper. Latches on the first farewell phrase and drops everything after. The caller saying "goodbye" back doesn't trigger another bot farewell.
Phone-number rules. The LLM is instructed to spell phone numbers as words, and a TextNormalizer converts any digit-form numbers that slip through into Piper-friendly spoken digits.
No fake calendar. The system prompt explicitly forbids the model from inventing time slots ("I have 2 PM and 4 PM available") — it collects a requested time and a human handles availability.

What this is not

Not connected to the PSTN. This bot speaks over WebSocket. To take real phone calls you'd bridge it through Twilio Programmable Voice, FreeSWITCH, or similar — Pipecat has integrations for that.
Not a calendar system. Bookings are written to JSON; a human (or your existing scheduling system) does the actual confirmation.
Not a state machine. It's still LLM-driven turn taking with tool calls, layered with defensive validators. A more robust production version would use pipecat-flows for explicit FSM-driven slot filling.

License & attribution

Forked from the Pipecat phone-bot quickstart.
