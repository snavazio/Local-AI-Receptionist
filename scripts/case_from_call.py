"""Convert a real production call (call_logs/call_*.json) into an eval-case
YAML skeleton.

When a real customer call goes badly, this script reads the saved transcript
and stamps out a starter case you can paste into eval/cases.yaml. The bot
then improves against real failures, not just synthetic ones.

Usage:
  python scripts/case_from_call.py call_logs/call_20260506_124351.json
  python scripts/case_from_call.py call_logs/call_*.json --category cancel

Output is YAML printed to stdout. Pipe to a file or paste into cases.yaml.

Generated case has:
  - id: prod_call_<timestamp> (ready to rename to something descriptive)
  - description: placeholder (you should rewrite to say what behavior to test)
  - category: 'production' by default (override with --category)
  - user_turns: extracted from the conversation transcript
  - expect: empty placeholders for tool_called, must_say, etc.

You'll always edit before committing — this is a starter, not a finished case.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path


def yaml_quote(s: str) -> str:
    """Render a string as a double-quoted YAML scalar with proper escaping."""
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def extract_user_turns(transcript: list[dict]) -> list[str]:
    """Pull ordered user-role messages from the OpenAI-format transcript."""
    turns = []
    for msg in transcript:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                turns.append(content.strip())
    return turns


def extract_tool_calls(transcript: list[dict]) -> list[dict]:
    """Pull tool calls (assistant.tool_calls + tool.content pairs) for context."""
    calls = []
    for msg in transcript:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                calls.append({
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
    return calls


def derive_id(call_path: Path) -> str:
    """Make a sensible default case id from the call's filename timestamp."""
    m = re.search(r"(\d{8}_\d{6})", call_path.name)
    if m:
        return f"prod_call_{m.group(1)}"
    return f"prod_call_{datetime.datetime.now():%Y%m%d_%H%M%S}"


def render_case_yaml(*, case_id: str, category: str, source_file: Path,
                     user_turns: list[str], tool_calls: list[dict]) -> str:
    L: list[str] = []
    L.append(f"- id: {case_id}")
    L.append(f"  category: {category}")
    L.append("  description: |")
    L.append("    GENERATED FROM PRODUCTION CALL — rewrite this description to say")
    L.append("    WHAT BEHAVIOR you want to verify (e.g. 'caller corrects phone")
    L.append("    mid-flow; verify save_request gets the corrected number')")
    L.append("    before committing.")
    L.append(f"    Source: {source_file.name}")
    if tool_calls:
        L.append("    Original call's tool calls (for context):")
        for tc in tool_calls:
            args_short = (tc.get("arguments") or "")[:120]
            L.append(f"      - {tc.get('name')}: {args_short}")
    L.append("  user_turns:")
    if user_turns:
        for t in user_turns:
            L.append(f"    - {yaml_quote(t)}")
    else:
        L.append("    - \"PLACEHOLDER — no user turns found in transcript\"")
    L.append("  expect:")
    L.append("    # FILL THESE IN BEFORE COMMITTING:")
    L.append("    # tool_called: book_appointment_callback")
    L.append("    # tool_args_contain:")
    L.append("    #   caller_name: \"...\"")
    L.append("    #   callback_number: \"...\"")
    L.append("    # tool_must_not_be_called:")
    L.append("    #   - take_message")
    L.append("    # assistant_must_say_any:")
    L.append("    #   - \"...\"")
    L.append("    # assistant_must_not_say:")
    L.append("    #   - \"...\"")
    L.append("    max_assistant_farewells: 1")
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("call_log", help="Path to a call_logs/call_*.json transcript")
    parser.add_argument("--id", help="Case id (default: prod_call_<timestamp>)")
    parser.add_argument("--category", default="production",
                        help="Category to file the case under. Default: production")
    args = parser.parse_args()

    call_path = Path(args.call_log)
    if not call_path.exists():
        print(f"Not found: {call_path}", file=sys.stderr)
        return 2

    try:
        transcript = json.loads(call_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Failed to parse {call_path}: {e}", file=sys.stderr)
        return 2

    user_turns = extract_user_turns(transcript)
    tool_calls = extract_tool_calls(transcript)
    case_id = args.id or derive_id(call_path)

    out = render_case_yaml(
        case_id=case_id,
        category=args.category,
        source_file=call_path,
        user_turns=user_turns,
        tool_calls=tool_calls,
    )
    print(out)
    print(file=sys.stderr)
    print(f"# Extracted {len(user_turns)} user turns and {len(tool_calls)} tool calls.", file=sys.stderr)
    print(f"# Edit description, fill in expect:, then paste into eval/cases.yaml.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
