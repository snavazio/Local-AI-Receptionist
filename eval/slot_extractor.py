"""Deterministic slot extractor for the receptionist FSM.

The whole point: the LLM has been unreliable at "deciding when to commit
to a tool call." So we take that decision away from it. This module
extracts caller name, phone, day+time, message, intent, and yes/no
from each user turn using regex and a small lookup table — pure Python,
no LLM. The state machine in harness_deterministic.py uses the
extractor's output to advance.

The LLM is then only called to *generate the response prose* given the
current state and known slots — a job it does fine.
"""

from __future__ import annotations

import re

from harness import _extract_phone_digits  # already handles digits + word-form

# ──────────────── intent classifier ────────────────────────────────

_EMERGENCY_KEYWORDS = (
    # Trauma
    "knocked out", "knocked-out", "knock out my tooth", "broken tooth",
    "broke a tooth", "broke my tooth", "trauma", "accident",
    # Bleeding
    "bleeding", "blood", "won't stop bleeding", "can't stop the bleeding",
    "can't stop",
    # Severe pain phrasings (real callers use many)
    "severe pain", "extreme pain", "horrible pain", "intense pain",
    "really bad pain", "terrible pain", "throbbing pain", "excruciating",
    "unbearable", "agonizing pain", "horrible throbbing", "can't sleep",
    # Swelling / abscess
    "swelling", "swollen", "facial swelling", "abscess",
    # Direct
    "this is an emergency", "dental emergency", "urgent",
)
_APPOINTMENT_KEYWORDS = (
    "book", "schedule", "make an appointment", "set up an appointment",
    "come in for", "appointment for", "cleaning", "checkup", "exam",
)
_MESSAGE_KEYWORDS = (
    "leave a message", "take a message", "tell the doctor", "tell the dentist",
    "have someone call", "have you call", "have somebody call", "callback",
    "call me back about", "ask the doctor",
)
# Out-of-scope or topical phrasings that should also be routed as "message"
# (the bot has no calendar / no pricing — collect a callback request).
_TOPICAL_MESSAGE_KEYWORDS = (
    "how much", "cost", "price", "pricing", "billing", "insurance",
    "reschedule", "move my appointment", "change my appointment",
    "move it to", "cancel my appointment",   # cancel = also message-like
    "x-ray", "xray", "records", "referral",
    # Human handoff requests should also route to message (we'll take
    # their name + phone and have a human call back)
    "real person", "talk to a human", "speak to a human", "speak to someone",
    "talk to someone", "operator", "live agent",
)


# ──────────────── correction detection ─────────────────────────────

_CORRECTION_MARKERS = re.compile(
    r"\b(actually|wait|no(?:,|\s+sorry|\s+wait)?|sorry,? that's|that's wrong|"
    r"i meant|let me correct|correction|change that|instead|"
    r"not\s+(?:tuesday|wednesday|thursday|friday|saturday|sunday|monday|"
    r"that)\b|"
    r"make that|i actually want|on second thought)\b",
    re.IGNORECASE,
)


def detect_correction(text: str) -> bool:
    """Returns True if the user is correcting / changing something they said."""
    return bool(_CORRECTION_MARKERS.search(text or ""))


def split_at_correction(text: str) -> str:
    """Return the part of the text AFTER the correction marker. The slot
    extractors should run on this — that's what the user actually wants."""
    m = _CORRECTION_MARKERS.search(text or "")
    if not m:
        return text or ""
    return text[m.end():].strip()


def classify_intent(text: str) -> str | None:
    """Return 'appointment' | 'message' | 'emergency' | None.

    Priority: emergency > appointment > message > topical-message.
    Appointment beats message when BOTH appear in the same utterance —
    e.g. "book a cleaning AND leave a message about insurance" — since
    the appointment is the more structured, higher-value action and
    a message can be tacked into the appointment record."""
    t = (text or "").lower()
    if any(kw in t for kw in _EMERGENCY_KEYWORDS):
        return "emergency"
    has_appt = any(kw in t for kw in _APPOINTMENT_KEYWORDS)
    has_msg = any(kw in t for kw in _MESSAGE_KEYWORDS)
    has_topical = any(kw in t for kw in _TOPICAL_MESSAGE_KEYWORDS)
    if has_appt:
        return "appointment"
    if has_msg or has_topical:
        return "message"
    # Implicit appointment signal: caller volunteers a day + time without
    # any other intent words. Common opener: "Wednesday at 3 PM."
    if _find_day(t) and _find_specific_time(t):
        return "appointment"
    return None


# ──────────────── name extractor ───────────────────────────────────

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([a-z][a-z'\.\- ]{1,40})", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am)\s+([a-z][a-z'\.\- ]{1,40})", re.IGNORECASE),
    re.compile(r"\bthis is\s+([a-z][a-z'\.\- ]{1,40})", re.IGNORECASE),
    re.compile(r"\bcall me\s+([a-z][a-z'\.\- ]{1,40})", re.IGNORECASE),
    # "It's X" — case-sensitive on the captured name to avoid matching
    # idioms like "It's been like this" / "It's hurting" / "It's late"
    # where the captured word would be lowercase.
    re.compile(r"\bIt'?s\s+([A-Z][a-z'\.\- ]{1,40})"),
]
# Words that should never be treated as a name (common confusions when
# the caller is short-answering a different prompt).
_NAME_BLOCKLIST = {
    "sometime", "anytime", "tomorrow", "today", "yes", "no", "okay", "sure",
    "fine", "good", "yeah", "nope", "thanks", "please", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night", "noon",
    "a", "an", "the", "and", "or", "but", "with", "without",
    "appointment", "cleaning", "checkup", "message", "emergency",
    "calling", "called", "hello", "hi", "hey",
    # Idioms commonly captured by "call me X" / "I'm X"
    "back", "later", "soon", "right back", "tomorrow morning", "right now",
    "out", "in", "down", "off", "on", "up",
    # Conversational acks that look like 1-word names
    "ok", "alright", "uhm", "umm", "huh", "well",
    # First words that lead to false "I'm X" / "It's X" matches
    "been", "having", "feeling", "experiencing", "trying", "calling",
    "wondering", "thinking", "looking",
    # Pronouns / fillers
    "you", "me", "him", "her", "them", "us", "we", "they",
    # Prepositions
    "from", "to", "for", "about", "regarding",
}


def _clean_name(raw: str) -> str | None:
    """Trim trailing junk + reject blocklisted words."""
    raw = raw.strip().rstrip(".,;:!?")
    # Stop at the first comma or "and" or " for "
    raw = re.split(r"\s+(?:and|for|on|at|to)\s+", raw, maxsplit=1)[0].strip()
    if not raw:
        return None
    parts = [p for p in raw.split() if p]
    if not parts:
        return None
    # Reject if first word is a blocked word (e.g. "Sometime" because the
    # caller answered "Sometime in the afternoon" to a "tell me your name?"
    # question).
    if parts[0].lower() in _NAME_BLOCKLIST:
        return None
    # Take up to 3 words (handles "Anjali Khanna", "Joaquin Wells").
    return " ".join(parts[:3])


def _looks_like_name_chunk(chunk: str) -> str | None:
    """Heuristic: a chunk looks like a name if it's 1-3 alphabetic words
    AND the first word isn't in the blocklist AND it doesn't look like
    a time. Returns the cleaned name or None."""
    chunk = chunk.strip().rstrip(".,;:!?'\"")
    if not chunk:
        return None
    # Reject if the chunk is a time expression like "Three PM", "Ten AM",
    # "Noon", "Midnight", "3:30 pm".
    if _find_specific_time(chunk) is not None:
        return None
    parts = chunk.split()
    if not parts or len(parts) > 3:
        return None
    # All parts must be letters only (allow apostrophe + hyphen)
    for p in parts:
        clean_p = p.strip(".,;:!?'\"")
        if not clean_p or not all(c.isalpha() or c in "'-" for c in clean_p):
            return None
    first = parts[0].strip(".,;:!?'\"").lower()
    if first in _NAME_BLOCKLIST:
        return None
    return " ".join(p.strip(".,;:!?'\"") for p in parts)


def extract_name(text: str, *, in_name_state: bool = False) -> str | None:
    """Try patterns first; fall back to "the whole response is the name"
    only when the state machine is explicitly asking for a name."""
    t = (text or "").strip()
    for pat in _NAME_PATTERNS:
        m = pat.search(t)
        if m:
            cleaned = _clean_name(m.group(1))
            if cleaned:
                return cleaned

    if in_name_state:
        # User is responding to "what's your name?". Try comma-separated
        # chunks (first chunk often is the name even when phone/etc.
        # follow), then the whole utterance.
        for chunk in t.split(","):
            cand = _looks_like_name_chunk(chunk)
            if cand:
                return cand
    return None


# ──────────────── phone extractor ───────────────────────────────────

def extract_phone(text: str) -> str | None:
    """Wrapper around harness._extract_phone_digits — returns digit-only
    string of length ≥7 if a phone number is identifiable, else None."""
    digits = _extract_phone_digits(text or "")
    if digits and len(digits) >= 7:
        return digits
    return None


# ──────────────── day + time extractor ──────────────────────────────

_DAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
)
_DAY_PHRASES = ("today", "tomorrow", "next week", "this week")
_TIME_DIGIT = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", re.IGNORECASE)
_TIME_OCLOCK = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(am|pm|o'clock)\b", re.IGNORECASE)
# Permissive variant: bare word-form number ("eleven", "ten") without AM/PM.
# Only used when the state machine is asking for a window — otherwise too
# many false positives ("eleven cents").
_TIME_BARE_WORD = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.IGNORECASE)
_TIME_NOON = re.compile(r"\b(noon|midnight)\b", re.IGNORECASE)
_VAGUE_TIMES = ("morning", "afternoon", "evening", "night", "anytime", "any time", "whenever")


def _find_day(text: str) -> str | None:
    """Find the FIRST day mentioned in the text. Important for correction
    handling like 'change that to Wednesday, not Tuesday' — we want the
    first day in word-order, not the first day in any fixed list."""
    t = (text or "").lower()
    candidates = []
    for d in _DAYS:
        idx = t.find(d)
        if idx != -1:
            candidates.append((idx, d))
    for p in _DAY_PHRASES:
        idx = t.find(p)
        if idx != -1:
            candidates.append((idx, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1].capitalize()


def _find_specific_time(text: str, *, in_window_state: bool = False) -> str | None:
    """Return a time string only if it's specific (not 'afternoon').

    in_window_state=True relaxes the rule to accept bare word-form
    numbers ("eleven", "ten") without an AM/PM suffix — only safe when
    the bot has explicitly asked for a time."""
    t = (text or "")
    m = _TIME_DIGIT.search(t)
    if m:
        return m.group(0)
    m = _TIME_OCLOCK.search(t)
    if m:
        return m.group(0)
    m = _TIME_NOON.search(t)
    if m:
        return m.group(0)
    if in_window_state:
        m = _TIME_BARE_WORD.search(t)
        if m:
            return m.group(0)
    return None


def is_vague_time(text: str) -> bool:
    t = (text or "").lower()
    return any(v in t for v in _VAGUE_TIMES) and _find_specific_time(text) is None


def extract_window(text: str, *, accumulated_day: str | None = None,
                   accumulated_time: str | None = None) -> tuple[str | None, str | None, bool]:
    """Returns (window_string_or_None, why_not, accept_partial).

    - If we have day AND specific time, return concatenated window.
    - If only day or only time, return None — but the caller may fill
      via accumulated context across turns (we track day/time
      independently in the state machine).
    """
    day = _find_day(text) or accumulated_day
    tm = _find_specific_time(text) or accumulated_time
    if day and tm:
        return f"{day} at {tm}", None, True
    return None, "need both day and specific time", False


# ──────────────── yes / no extractor ────────────────────────────────

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "correct", "right",
              "okay", "ok", "please", "absolutely", "exactly"}
_NO_WORDS = {"no", "nope", "nah", "not really", "incorrect", "wrong"}


def extract_yes_no(text: str) -> str | None:
    t = (text or "").lower().strip().rstrip(".,;:!?")
    if not t:
        return None
    # Match the whole utterance OR the first word — "yeah, sure" → yes
    first = t.split()[0] if t.split() else ""
    if t in _YES_WORDS or first in _YES_WORDS:
        return "yes"
    if t in _NO_WORDS or first in _NO_WORDS or any(p in t for p in _NO_WORDS):
        return "no"
    return None


# ──────────────── message topic extractor ──────────────────────────

def extract_message_topic(text: str, prior_turns: list[str] | None = None) -> str | None:
    """One-line summary of what the caller wants forwarded. For the message
    flow: first try to find an explicit "ask about X" / "questions about X"
    phrasing; otherwise fall back to the gist of the first non-trivial
    earlier turn (which usually contains the topic)."""
    t = (text or "").lower()
    # Common patterns
    for pat, fmt in [
        (re.compile(r"about\s+([a-z\- ]{3,80})"), "About {}"),
        (re.compile(r"call\s+(?:with|about)\s+([a-z\- ]{3,80})"), "About {}"),
        (re.compile(r"reschedul\w+\s+(?:my appointment\s+)?(?:to|from|for)?\s*([a-z\- 0-9]{3,80})"),
         "Reschedule appointment: {}"),
    ]:
        m = pat.search(t)
        if m:
            topic = m.group(1).strip().rstrip(".,;:!?")
            if topic and len(topic) > 2:
                return fmt.format(topic).strip()
    # Fallback: use the first prior turn if it looked substantive
    if prior_turns:
        for prev in prior_turns:
            p = (prev or "").strip()
            if len(p.split()) >= 4 and "?" in p or len(p.split()) >= 5:
                return p[:120]
    # Last resort: use the current turn truncated
    s = (text or "").strip()
    if len(s) > 0:
        return s[:120]
    return None
