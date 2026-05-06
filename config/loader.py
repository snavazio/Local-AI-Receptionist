"""Domain config loader.

Reads config/<domain>.yaml and exposes:
  CONFIG          — full dict
  PRACTICE        — flat dict of business facts (back-compat alias)
  GREETING        — first-turn hardcoded greeting
  SYSTEM_PROMPT   — fully interpolated prompt string
  TOOLS           — list of OpenAI-compat tool schemas (for harness)
  TOOLS_SCHEMA    — Pipecat ToolsSchema (for bot.py)

Pick the config via the DOMAIN_CONFIG env var (default: config/dental.yaml,
relative to project root).

This is what makes the project domain-agnostic: same Python, swap the YAML
to point at a different business / persona / tool set.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "dental.yaml"
CONFIG_PATH = Path(os.environ.get("DOMAIN_CONFIG", str(DEFAULT_CONFIG)))


def _interpolate(text: str, *, business: dict, persona: dict, greeting: str) -> str:
    """Replace {business.name}, {persona.role}, {greeting_oneline} placeholders."""
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key == "greeting_oneline":
            return greeting.strip().replace("\n", " ")
        if key.startswith("business."):
            return str(business.get(key.split(".", 1)[1], "{" + key + "}"))
        if key.startswith("persona."):
            return str(persona.get(key.split(".", 1)[1], "{" + key + "}"))
        return "{" + key + "}"  # leave unknown placeholders alone
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}", sub, text)


def load(config_path: Path | None = None) -> dict[str, Any]:
    """Load + interpolate the domain config. Returns a dict with keys:
    business, persona, greeting, prompt (interpolated), tools."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Domain config not found: {path}\n"
            f"Set DOMAIN_CONFIG env var or create {path}."
        )
    raw = yaml.safe_load(path.read_text())
    business = raw.get("business", {}) or {}
    persona = raw.get("persona", {}) or {}
    greeting = (raw.get("greeting") or "").strip()
    prompt = _interpolate(
        raw.get("prompt", ""),
        business=business, persona=persona, greeting=greeting,
    ).strip() + "\n"
    return {
        "business": business,
        "persona": persona,
        "greeting": greeting,
        "prompt": prompt,
        "tools": raw.get("tools", []),
    }


# Convenience: precomputed values matching the names bot.py / harness.py used
# before the config extraction. Importing them works the same as before.
CONFIG = load()
PRACTICE = {
    "name": CONFIG["business"].get("name", ""),
    "doctor": CONFIG["business"].get("doctor", ""),
    "hours": CONFIG["business"].get("hours", ""),
    "address": CONFIG["business"].get("address", ""),
    "emergency_line": CONFIG["business"].get("emergency_line", ""),
}
GREETING = CONFIG["greeting"]
SYSTEM_PROMPT = CONFIG["prompt"]


def _tool_to_openai(tool: dict) -> dict:
    """Convert YAML tool spec to OpenAI-compat function schema (used by harness)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", "").strip(),
            "parameters": {
                "type": "object",
                "properties": tool.get("parameters", {}),
                "required": tool.get("required", []),
            },
        },
    }


TOOLS = [_tool_to_openai(t) for t in CONFIG["tools"]]


def make_pipecat_tools_schema():
    """Return a Pipecat ToolsSchema. Imported lazily to avoid pulling pipecat
    when only the eval harness needs the OpenAI-format TOOLS list."""
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    return ToolsSchema(standard_tools=[
        FunctionSchema(
            name=t["name"],
            description=t.get("description", "").strip(),
            properties=t.get("parameters", {}),
            required=t.get("required", []),
        )
        for t in CONFIG["tools"]
    ])
