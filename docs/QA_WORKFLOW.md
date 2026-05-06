# QA Workflow for LLM-driven Applications

A project-agnostic guide to building, running, and improving an automated
QA suite for any LLM-driven application. Distilled from lessons learned
building the eval suite for this voice-receptionist project. Copy this file
and the `eval/` directory into a new project to bootstrap QA in ~an hour.

---

## Table of contents

1. [Three layers of testing](#1-three-layers-of-testing) — when to use which
2. [Case design](#2-case-design) — what a good test case looks like
3. [Naming and documentation](#3-naming-and-documentation) — so failures are self-explanatory
4. [The iteration loop](#4-the-iteration-loop) — failure → diagnosis → fix → re-test
5. [Noise hygiene](#5-noise-hygiene) — LLM evals are noisy; here's how to cope
6. [Failure-mode taxonomy](#6-failure-mode-taxonomy) — known patterns and their fix shapes
7. [Artifacts](#7-artifacts) — what each file does, with templates
8. [Bootstrapping a new project](#8-bootstrapping-a-new-project) — checklist

---

## 1. Three layers of testing

LLM applications have three distinct testing surfaces. Each layer catches
different bugs and runs at different speeds. You want **all three**, but
they serve different purposes.

### Layer 1 — Unit tests (deterministic, fast)

Pure-Python tests of helper functions, data validators, post-processors,
and any deterministic code. **No LLM calls.** Run in ~1-3 seconds.

These verify *the building blocks work*: when invoked with known inputs,
they produce known outputs.

Example: in this project, we have 121 unit tests for things like:
- `extract_phone_digits("two zero one three eight eight two one four nine")` → `"2013882149"`
- `_strip_malformed_tool_call("<tool_call>{...}</tool_call>")` → `""`
- `_FAREWELL_RE.search("Thanks for calling")` → `None` (greeting, not farewell)

**They don't measure LLM behavior.** A 100% green unit test suite tells
you nothing about whether the LLM works — it tells you the supporting
deterministic code does. That's still important; bugs in the helpers are
silent killers.

### Layer 2 — Conversation eval (LLM-driven, scripted)

Scripted multi-turn conversations against the live LLM, asserting on
tool calls and assistant text. **Hits the LLM but skips audio.** Each
case takes seconds; full suite of 100-300 cases takes minutes.

This is where most of your QA effort goes. It tests the system the user
actually experiences, minus the audio layer:
- Does the LLM call the right tool with the right arguments?
- Does it follow the conversation flow you specified?
- Does it handle edge-case inputs (corrections, multi-intent, dense info)?

### Layer 3 — End-to-end audio eval (real audio, slow)

Pre-recorded audio fed to the live system through whatever transport
the bot uses (WebSocket, SIP, etc.). Catches STT failures, audio-level
latency issues, and pipeline-state bugs that the text harness can't see.

Slow (each conversation = real-time playback) and infrastructure-heavy.
Worth building once Layer 2 is mature.

### When to use which

| Symptom you're hunting | Layer | Why |
|---|---|---|
| Helper function returns wrong value | 1 | Deterministic, fast, isolates the bug |
| LLM doesn't call the tool when it should | 2 | Layer 1 can't see this |
| LLM hallucinates argument values | 2 | Same |
| Production sounds weird but logs look right | 3 | Audio-level only |
| Whisper mistranscribes specific phrasing | 3 | Same |
| Tool fires but downstream JSON is malformed | 1 + 2 | 1 catches schema, 2 catches usage |

---

## 2. Case design

A test case has one job: **prove or disprove a single hypothesis about
behavior under a specific input pattern**.

### One failure mode per case

Bad case: "happy path with phone, name, time, and the caller is rude
about being on hold and also corrects their phone twice."

Good cases (split into three):
- `happy_path_basic` — straight booking
- `correction_phone_typo` — caller corrects phone last digit
- `polite_decline_on_hold` — caller is annoyed; bot stays professional

When a single case mixes patterns, a failure tells you nothing — you
don't know which pattern broke it.

### Each case must have

- **`id`** — unique, follows naming convention (see §3)
- **`category`** — for grouping in reports
- **`description`** — what *behavior* this case is testing, not what the
  caller says. (See §3 for examples.)
- **`user_turns`** — list of caller utterances, in order
- **`expect`** — assertions:
  - `tool_called` — name of tool that must fire
  - `tool_args_contain` — substring-match on tool args (with smart phone
    normalization)
  - `tool_must_not_be_called` — list of tools that must NOT fire
  - `assistant_must_say_any` — at least one of these substrings must
    appear in the bot's text
  - `assistant_must_not_say` — none of these may appear
  - `max_assistant_farewells` — usually 1; cancel/end-of-call enforcement

### Anti-patterns to avoid

- **No `description`.** Future-you (or your teammate) gets a failure with no
  idea why the case was written. Diagnosis becomes archaeology.
- **Asserting on bot prose.** The bot says "Got it" today, "Saved" tomorrow,
  "Thanks" the day after. Assert on tool calls and key facts (digits,
  named days), not phrasing.
- **Cases that depend on each other.** Each case starts with a fresh
  conversation. Don't write `case_b` that assumes `case_a` "set up" state.
- **One caller turn that contains everything.** Real conversations are
  multi-turn. If you only test single-turn inputs, you'll miss the
  flow-control bugs that dominate failures.
- **Asserting on case-sensitive substrings without `.lower()`.** Make the
  harness compare case-insensitively (we do).

---

## 3. Naming and documentation

Failures should explain themselves at a glance.

### ID convention

```
<category>_<scenario>_<variant?>
```

- **`category`** — high-level grouping. ~10-15 categories total.
  Examples: `happy_path`, `correction`, `cancel`, `emergency`,
  `phone_variant`, `name_edge`, `false_emergency`, `multi_intent`,
  `out_of_scope`, `disfluent`.
- **`scenario`** — what specific situation this case covers.
  Examples: `extension`, `obrien`, `cold_sensitivity`, `book_plus_message`.
- **`variant`** — only when you have two near-cousins testing the same
  scenario with different phrasings or values. Use `_a` / `_b` /
  meaningful suffix (`_terse`, `_polite`, `_corrected`).

Real examples:
```
happy_path_basic
happy_path_basic_terse           # variant: terse caller
phone_variant_extension          # base
phone_variant_extension_voip     # variant
name_edge_apostrophe_obrien      # name with apostrophe
false_emergency_cold_sensitivity # sounds urgent, shouldn't escalate
multi_intent_book_plus_message   # two intents in one call
```

### `description` field — write it for the failure

Bad: `"Caller says they want a cleaning."`
Good: `"Verifies the bot doesn't escalate to emergency when the caller mentions sensitivity to cold drinks (sounds urgent, isn't)."`

The description should answer: **"why was this test written?"** Future-you
reads the failing case and immediately knows what behavior is broken.

### Common bad descriptions and their fixes

| Bad | Good |
|---|---|
| "Test phone number" | "Verifies extract_phone_digits handles the +1 country code" |
| "Caller cancels" | "Caller cancels after giving full info; tool must NOT fire" |
| "Name with apostrophe" | "Verifies _looks_like_garbage_name doesn't reject 'O'Brien'" |

---

## 4. The iteration loop

When the eval reports failures, follow this loop. Don't skip steps —
each prevents a class of mistake.

```
   [1] Run the eval; pick the worst category or biggest regression.
                          ↓
   [2] Open ONE failing case's full transcript.
                          ↓
   [3] Classify the failure mode (see §6).
                          ↓
   [4] Hypothesize ONE fix (prompt edit, code regex, tool change).
                          ↓
   [5] Apply the fix; re-run that category only.
                          ↓
   [6] Did it improve? Did anything else regress?
        ├── improved, no regression  →  keep, log it
        ├── improved, but regression  →  revert, try a narrower fix
        └── no change                 →  noise; run again or revert
                          ↓
   [7] Update the iteration log so future-you understands what worked.
```

### Why this works

**Step 2 (read the transcript)** is the step everyone wants to skip.
Don't. The pass/fail bit tells you nothing; the actual LLM output tells
you the failure mode. 80% of fixes are obvious once you've read the
transcript.

**Step 4 (ONE fix)** is the most-violated rule. If you change the prompt
AND the schema AND the post-processor in one iteration, you can't tell
which change moved the score. Your future iterations on this category
will be guesswork.

**Step 5 (category-only)** keeps iteration cycles fast. Full eval is
expensive (minutes); category eval is cheap (~1 minute). Use the cheap
loop until the category looks good, then run the full eval to check for
ripple effects.

**Step 7 (log it)** is what separates "QA process" from "guess and check".
A real iteration log lets you answer "have we tried this before?"

### What to log per iteration

In `docs/ITERATION_LOG.md`, newest first:

```
## YYYY-MM-DD — <category>: <before>/<n> → <after>/<n>

### iter<N> — <one-line label>
**Observed:** specific failing transcript snippet, what was wrong.
**Cause:** your hypothesis.
**Change:** the actual diff (prompt sentence added, regex pattern, etc).
**Result:** new pass count, any other categories affected.
```

Even when an iteration fails, log the negative result. Future-you will
re-derive the same failed idea otherwise.

---

## 5. Noise hygiene

LLM evals are non-deterministic even at `temperature=0`. On this
project, the same prompt + same code can produce a 6-8/10 score on the
same category across two runs.

### Why
- Ollama scheduler / KV-cache state varies
- Floating-point non-associativity in matmul
- Some tokens are tied at the top probability; tie-break is non-deterministic

### What to do about it

1. **Don't declare a fix "real" off a single run.** Run 2-3 times. Keep the
   fix only if the median score moves clearly.
2. **Build noise tolerance into your watcher.** Our `watch.py` only marks
   a real regression when:
   - A specific case that previously passed is now failing (per-case
     check, not category-level), AND
   - The drop persists across re-runs (manual judgment for now).
3. **Big latency jumps are signals too.** If p95 LLM-call latency jumps
   500ms+ between runs without a model swap, something else changed
   (background process, GPU contention).

### Acceptable noise band

For a category of 10 cases at temp=0:
- ±1 case across re-runs: probably noise
- ±2-3 cases: borderline; rerun to denoise
- ±4+ cases: real signal

Across the full 100-case suite:
- ±2-3 cases: probably noise
- ±5+ cases: real

---

## 6. Failure-mode taxonomy

Classify every failure into one of these. Each has a known fix shape.

### Talks-without-acting
The model says "your request is saved" / "got it" / "I'll book that"
without actually invoking the tool. Slot data is lost.

**Fix shape (in order of cheapness):**
1. Add an explicit prompt rule: "the next thing you produce must be the
   tool call, not text."
2. Move the rule earlier in the prompt (transformer attention bias).
3. If still flaky, switch the flow to an FSM (stateful processor that
   invokes the tool itself when slots are filled — the LLM only phrases
   the asks).

### Hallucinated argument values
The model passes `caller_name="John Doe"` or
`callback_number="2013882149"` even though the caller said something
else. Often caused by **example values in the prompt** — the model
treats the example as canonical default.

**Fix shape:**
1. Search the prompt for example values; remove or genericize.
2. Add `placeholder_values` (a hard-coded set including common defaults
   like "John Doe", "555-1234") to the tool's gating logic so it
   rejects placeholders before saving.

### Language drift
The model responds in a non-target language (Chinese, Spanish, etc.)
mid-conversation. Common with multilingual models like Qwen.

**Fix shape:**
1. Add an "ALWAYS respond in <language> only" rule near the top of
   the prompt.
2. Don't rely on this alone — also add a regex check: if the assistant's
   text contains characters outside the target alphabet, drop the frame
   and re-prompt.

### Leaked chat-template tokens
The model emits raw scaffolding like `<|im_start|>`, `<tool_call>`,
`TokenName:`, `_icall_`, `<|endoftext|>` as plain text. These reach
TTS and confuse callers.

**Fix shape:**
1. Add a `MalformedToolCallStripper` post-processor (regex-based).
2. Maintain a list of known patterns; new ones get added when observed.
3. Drop the frame entirely if cleaning leaves nothing legible.

### Slot skip
The model never asks for a required slot, or asks for it but never
captures the answer.

**Fix shape:**
1. Re-order the slot list in the prompt. Earlier mentions get more
   weight.
2. Make required slots more prominent (numbered list, bold "REQUIRED").
3. If the model captures-but-doesn't-use, the issue is usually the
   tool call coming before all slots are filled. Add a rule: "do not
   call the tool until ALL of [name, phone, time] are in your record."

### Hallucinated multi-turn dialogue
The model writes both sides of a conversation in a single response,
then commits fabricated values to the tool. Looks like:
> "What's your name? TokenName: Steve TokenName: My name is Steve.
> TokenName: And your phone? TokenName: 555-1234..."

**Fix shape:**
1. Prompt: "Generate exactly ONE assistant turn at a time. Never write
   the caller's reply yourself."
2. Stripper for `TokenName:`, `<|im_start|>user`, similar patterns.

### Apocalyptic-rule paralysis
You added a strict rule like "FORBIDDEN... if you do X you have failed
your job." The model becomes so risk-averse it does nothing — hangs
up early or asks endless clarifying questions to avoid action.

**Fix shape:**
1. Soften the wording. Calmer rules outperform threats.
2. Pair the prohibition with positive guidance: "instead, do Y."

### Format hallucination
The model emits the right intent but in the wrong format — a phone
number spelled in words when the tool expects digits, a date in a
non-standard form, etc.

**Fix shape:**
1. Code-side: extend your validators to accept more formats (we did
   this with `_words_to_digits` for phone numbers).
2. Tool description: be explicit about format ("digits only, no
   dashes").

---

## 7. Artifacts

What each file does. Templates included so you can copy them into a
new project.

### `eval/cases.yaml`

The case database. List of YAML objects:

```yaml
- id: happy_path_basic
  category: happy_path
  description: |
    Verifies the bot completes a standard appointment booking when
    the caller provides name, phone, and preferred time across multiple
    turns. The save_request tool must fire with kind=appointment and
    the actual values the caller spoke.
  user_turns:
    - "I'd like to book a cleaning."
    - "My name is Steve."
    - "201-388-2149."
    - "Tuesday at 2 PM."
    - "No thanks."
  expect:
    tool_called: book_appointment_callback
    tool_args_contain:
      caller_name: "Steve"
      callback_number: "2013882149"
    max_assistant_farewells: 1
```

### `eval/harness.py`

Drives one case through the LLM. Captures tool calls + assistant text.
Mirrors any production-side post-processors so the eval reflects what
the user would actually experience.

Key responsibilities:
- Build the system prompt + tool schema
- Loop: send user turn → read LLM response → execute tool stubs →
  feed results back → loop until tool error or all turns consumed
- Apply post-processors (strippers, dedupers) to assistant text
- Time each LLM call for latency metrics

### `eval/run_eval.py`

CLI runner. Loads cases, calls the harness, evaluates assertions,
writes a markdown report. Supports:
- `--category X` — run only one category
- `--case ID` — run one case
- `--shard N/M` — run a slice (for parallelization)
- `--concurrency N` — fan out across N subprocesses
- `--model NAME` — try a different LLM
- `--json-out FILE` — emit structured output for tooling

### `eval/watch.py`

Regression detector. Runs the full eval, compares to last accepted
baseline (`eval/baseline.json`), emits a focused diff report. Exits
non-zero on regression so cron / CI can flag.

### `eval/trend.py`

Reads `eval/history.jsonl` (appended by `watch.py`) and prints ASCII
sparklines of pass-rate and latency over time. Catches slow drift
across days/weeks that watch.py doesn't see.

### `tests/`

Pytest unit tests for helpers. Run as a pre-commit hook.

### `docs/ITERATION_LOG.md`

Newest-first log of every iteration. Failure mode → hypothesis →
change → result. Critical for not re-deriving failed ideas.

### `docs/QA_WORKFLOW.md`

This file. Project-agnostic. Copy into new projects.

---

## 8. Bootstrapping a new project

Checklist for adding QA to a fresh LLM project.

### Day 1: skeleton
- [ ] Copy `eval/` directory into the new project (rename tools / model in `harness.py`)
- [ ] Copy `docs/QA_WORKFLOW.md` and `docs/ITERATION_LOG.md` (start it empty)
- [ ] Copy `tests/` directory; gut the test contents but keep the structure
- [ ] Wire `make test` and `make eval` (or your equivalent) so commands are one-liner

### Week 1: 30-50 starter cases
- [ ] Identify ~10 categories specific to your project
- [ ] Write 3-5 cases per category covering the happy paths and 1-2 edge cases
- [ ] Get to a baseline pass rate; commit `eval/baseline.json`
- [ ] Run `eval/watch.py` and confirm it correctly flags regressions

### Month 1: full coverage + iteration habit
- [ ] Expand to 100+ cases by writing variants of existing ones (fast)
- [ ] Set up scheduled `eval/watch.py` (cron, GitHub Actions, etc.)
- [ ] Make `ITERATION_LOG.md` updates a habit on every meaningful change
- [ ] Build the audio-level eval (Layer 3) if your app has audio

### Steady state
- [ ] Eval runs nightly; trend.py reviewed weekly
- [ ] Each new feature ships with at least 2 cases (positive + negative)
- [ ] Failures get assigned to a developer; iteration log entry is part of "done"

### Anti-patterns to watch for

- **"100% pass rate" pressure.** No real LLM eval gets to 100%. Aim for
  a high floor (80-90%) with the remaining failures being known and
  documented.
- **Treating eval as a release gate.** It's a regression detector, not
  a release blocker. Use it to inform decisions, not block them.
- **Letting cases rot.** When the product changes, cases become stale.
  Budget time monthly to prune cases for behaviors you no longer want.
- **Optimizing for the eval.** If you keep tweaking the prompt to pass
  the eval, you'll start failing real callers. Add new cases regularly
  to keep the eval honest.

---

## TL;DR

1. Three layers: unit tests, conversation eval, audio eval. All needed.
2. Each case tests ONE failure mode and has a `description` that says
   why it was written.
3. Naming: `<category>_<scenario>_<variant>`. Failures self-explain.
4. Iteration loop: read transcript, classify mode, ONE fix, re-run
   category, log result.
5. Eval is noisy at temp=0. Run candidates 2-3x before declaring a
   real delta.
6. Common failure modes have known fix shapes. Don't reinvent them.
7. Copy this file + `eval/` directory to bootstrap any new LLM project.
