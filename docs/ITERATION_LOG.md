# Iteration log

A running record of eval-driven changes to the bot. Newest first. Each
entry: what we observed → what we changed → result on the targeted
category.

---

## 2026-05-06 — correction: 4/10 ↔ 6-8/10 (noise band)

After the iter3 prompt fixes (one-turn-at-a-time + leak strippers)
took the full eval from 67 → 78/100, the `correction` category was
the lone clear regression: 4/10 vs the prior 7/10 baseline.

### v2 — soften FORBIDDEN, add dense-input + correction guidance
**Observed:** failing transcripts showed the bot hanging up early
("Take care!" right after caller gave full booking info) instead of
processing it. Hypothesis: my apocalyptic FORBIDDEN rule from the
prior session ("you have failed your job") made Qwen so skittish it
preferred farewell over action.

**Change:**
- Replaced FORBIDDEN with calmer "words like 'saved' should only
  appear AFTER ok:true."
- Added explicit "if a caller volunteers several slots at once,
  capture all of them and proceed straight to save_request."
- Added "if a caller corrects, update the slot and continue; do
  not start over."

**Result:** correction 4/10 → 6/10.

### v3 — English-only rule
**Observed:** v2's remaining failures showed Qwen drifting into
Chinese mid-conversation (`助理：` / `请告诉我您的电话号码`) — strippers
don't catch this since it's grammatically valid Chinese.

**Change:** added "ALWAYS respond in English only, regardless of
what language the caller seems to use."

**Result:** correction 6/10 → 8/10.

### v4 — strengthened correction-replay rule (REVERTED)
**Observed:** 2 cases still failing where the model did re-call
save_request but with the OLD (uncorrected) value, or said
"Thursday it is" verbally without re-calling.

**Change:** added a longer rule explicitly listing trigger words
("actually", "wait", "no, make that", "scratch that", "I meant")
that should cause an immediate save_request re-call with the new
slot values.

**Result:** **6/10 — regressed.** The strengthened rule broke
`correction_to_message` (which switches kind="appointment" to
kind="message" mid-flow) and `correction_double_back` (which
corrects twice). Reverted.

### v5 — same prompt as v3 (stability check)
**Result:** **6/10.** v3's 8/10 was partly noise. Real signal
across v3/v5: 6-8/10. Stopping the iteration on this category;
the marginal returns require denoising via multiple runs which
costs more eval time than it's worth.

**Lesson logged:** at temp=0, individual category re-runs vary
by ~20pp. Future iterations should run each candidate change 2-3x
before declaring a delta is real.

---

## 2026-05-05 — happy_path: 5/10 → 9/10

### iter0 baseline (full 100-case eval at c=1)
67/100 overall. Big drops vs the prior 80/100 baseline:
happy_path 5/10, vague_response 4/10, emergency 6/10, cancel 7/10.

### iter1 — remove example values from worked example
**Observed:** `happy_path_friday_morning` failed; the model passed
`callback_number: "two zero one, three eight eight, two one four nine"`
even though the case scripted "415-555-0182". Same phrase appeared in
multiple failing cases.

**Cause:** the message worked-example in `SYSTEM_PROMPT` used those
literal digits as illustration. Qwen treats in-prompt example values
as canonical defaults.

**Change:** rewrote the message worked-example as a generic pattern
(no specific name/phone/message digits).

**Result:** happy_path 5/10 → 8/10.

### iter2 — also remove example digits from the phone-words rule
**Observed:** failures still showed `2013882149` being passed to the
tool. Found another instance of the same digits in the
"phones MUST be spelled as words" rule.

**Change:** rewrote the rule without specific example digits, added
explicit "never invent or default to a phone number — only digits the
caller actually gave you."

**Result:** happy_path 7/10 in one run, 8/10 in another (noise floor).
Net parity with iter1; cleaner prompt. Kept.

### iter3 — "one turn at a time" rule + leak-pattern strippers
**Observed:** the still-failing `happy_path_friday_morning` showed
the assistant generating fake user replies inline:
> "What's your name? TokenName: Emily Davis TokenName: My name is
> Emily Davis. TokenName: And your phone? TokenName: 555-1234..."

The model was filling both sides of a multi-turn dialogue in one
response, then committing fabricated values to `save_request`.

**Change (two-layered):**
1. `SYSTEM_PROMPT`: "Generate exactly ONE assistant turn at a time.
   Never write the caller's reply yourself. Never use TokenName: /
   TokenNumber: / [user response] placeholders."
2. `MalformedToolCallStripper.PATTERNS`: also strip `TokenName:`,
   `TokenNumber:`, `CallableWrapper`, `SupportedContent` so any
   slip-through doesn't reach TTS.

**Result:** happy_path 9/10 (only `happy_path_consultation` still
failing). Pending: full 100-case re-baseline to see how this affected
other categories (vague_response, emergency, etc. likely benefit too —
similar failure modes were observed).

---

## How to read this log

When you make a change driven by eval failures:

1. Note which case(s) prompted it and the failure mode you saw.
2. Note the cause / hypothesis.
3. Note the specific change.
4. Run the targeted category eval; record the delta.
5. Eventually run the full eval to confirm no other category regressed.

Append entries newest-first so the freshest context is on top.
