# Runbook

Operational guidance for keeping the bot maturing. Pair with the README's
setup section.

## Daily routine

```bash
# 1. Run the regression watcher (~25 min).
make watch

# 2. Glance at the trend.
make trend

# 3. If watcher exited non-zero: read eval/regression_report.md.
cat eval/regression_report.md
```

Anything in 🔴 Regressions is a real failure: a case that was passing
yesterday and is failing today. The watcher will refuse to update the
baseline so the next run still compares against the last known-good state.

## "What's a normal pass rate?"

| Concurrency | Expected range | Notes |
| --- | --- | --- |
| 1 (deterministic, slow) | 76-82 / 100 | Use this for regression gating |
| 2 | ~10pp lower | Ollama context-cache thrashing — DON'T use for gating |
| 4 | varies / sometimes hangs | Throughput tests only |

If pass rate drops below 75 at concurrency 1, something real broke. Check
the most recent commits to `bot.py`, `eval/harness.py`, and the prompt.

## "The watcher says 71/100 but I didn't change anything"

The watcher defaults to concurrency 1 specifically because c≥2 has a ~10
percentage point noise band. If you ran it with `--concurrency 2`, the
71 is that noise, not a regression. Re-run with the default.

## "An eval is hung"

Symptoms: `pgrep -af run_eval` shows children alive but the log hasn't
moved in >5 min, GPU is busy.

Likely cause: Ollama context-cache thrashing under concurrent load
(especially with 7B / Hermes models — see commit history).

```bash
# Kill everything cleanly
pkill -f "watch.py|run_eval"
sleep 2
pgrep -af "run_eval"  # should show nothing

# Force-unload models from VRAM
curl -s -X POST http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b","keep_alive":0,"prompt":""}'

# Re-run at concurrency 1
make watch
```

## "I changed the prompt — is it better?"

```bash
# Single category turnaround (~2-3 min)
.venv/bin/python eval/run_eval.py --category message
```

Compare to the per-category numbers in `eval/baseline.json`. If the
target category improved AND no other category regressed, run the full
watcher to update the baseline.

## "I want to test a different LLM"

```bash
# Once-off (saves to a separate report path)
.venv/bin/python eval/run_eval.py --model qwen2.5:7b --report eval/report_qwen7b.md

# Don't update the main baseline with a non-default model — start a
# separate baseline file for that model:
.venv/bin/python eval/watch.py --model qwen2.5:7b \
  --update-baseline --report eval/regression_qwen7b.md
```

## "Ollama is loaded with multiple models, VRAM is full"

```bash
# List loaded
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# Force-unload one
curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"NAME","keep_alive":0,"prompt":""}'
```

## "Tests fail after I edit something"

`make test` is the unit-test gate (1.6s, no Ollama). It catches:

- Broken regex in `MalformedToolCallStripper` / `FarewellDeduper`
- Removed essential rules from `SYSTEM_PROMPT` (tests pin the contract)
- Bad refactors in `extract_phone_digits`, `_missing`, etc.

If a test fails, the message will tell you which contract you broke.
Most fixes are a one-line edit; prefer fixing the bug to changing the
test unless the test itself is wrong.

## Knobs worth knowing about

| Where | Knob | Effect |
| --- | --- | --- |
| `bot.py` | `ManualEnergyVAD.STOP_FRAMES` | How long of silence triggers end-of-turn (currently 25 → ~600 ms). Lower = snappier but cuts users off. |
| `bot.py` | `OLLamaLLMService.Settings.temperature` | 0 = deterministic. Don't raise this — the eval baseline assumes 0. |
| `bot.py` | `WHISPER_BIAS_PROMPT` | If Whisper starts mishearing a new vocabulary item, add it here before debugging the LLM. |
| `eval/cases.yaml` | per-case YAML | Add new cases here when you encounter a real-world failure. Re-baseline after. |

## Live calls

Track real call latency via `call_logs/latency_*.jsonl`:

```bash
# Most recent file
ls -t call_logs/latency_*.jsonl | head -1 | xargs cat | python3 -c "
import json, sys, statistics
rows = [json.loads(l) for l in sys.stdin if l.strip()]
for k in ('stt_ms', 'llm_ms', 'tts_ms', 'total_ms'):
    vals = [r.get(k) for r in rows if r.get(k) is not None]
    if vals:
        print(f'{k}: median={statistics.median(vals):.0f} p95={sorted(vals)[int(len(vals)*0.95)]:.0f}')
"
```

If real-call p95 wildly exceeds the eval's LLM p95, something else in
the audio pipeline (STT, TTS, or VAD) is the bottleneck.
