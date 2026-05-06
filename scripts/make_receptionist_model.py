"""Generate a 'receptionist-<base>' Ollama model from any base model.

Usage:
  python scripts/make_receptionist_model.py qwen2.5:14b
  python scripts/make_receptionist_model.py llama3.3:8b --temperature 0
  python scripts/make_receptionist_model.py mistral-nemo:12b --create

Behavior:
  1. Loads SYSTEM_PROMPT from bot.py (so it's always in sync with the
     live agent's prompt).
  2. Writes Modelfile.<base>-receptionist with:
       FROM <base_model>
       PARAMETER temperature ...
       PARAMETER num_predict ...
       SYSTEM "<the live system prompt>"
  3. With --create, runs `ollama create receptionist-<base> -f <Modelfile>`
     so the configured model is immediately available via `ollama run`.

Why use this:
  The live bot.py and eval/harness.py inject SYSTEM_PROMPT at call-time, so
  baked-in Modelfiles aren't STRICTLY required. But:
    - Quick interactive testing: `ollama run receptionist-qwen2.5:14b`
      lets you chat with the configured model from a shell.
    - Reproducibility: the Modelfile is a frozen record of the prompt + params
      that produced a given eval result.
    - Easy A/B: swap `model="receptionist-qwen2.5:14b"` in bot.py for a
      different baked configuration.

The script does NOT modify bot.py — only generates Modelfile artifacts.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_system_prompt() -> str:
    """Extract SYSTEM_PROMPT string from bot.py without importing the whole
    file (which would pull in pipecat / loguru / numpy / etc.)."""
    bot_py = (ROOT / "bot.py").read_text()
    # Match: SYSTEM_PROMPT = f"""..."""
    m = re.search(
        r'SYSTEM_PROMPT\s*=\s*f?"""(.*?)"""',
        bot_py, re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not find SYSTEM_PROMPT in bot.py")
    raw = m.group(1)

    # The f-string interpolates PRACTICE['name'] etc. We need to resolve those.
    practice_match = re.search(
        r"PRACTICE\s*=\s*\{(.*?)\}",
        bot_py, re.DOTALL,
    )
    practice: dict[str, str] = {}
    if practice_match:
        for k, v in re.findall(r'"(\w+)"\s*:\s*"([^"]+)"', practice_match.group(1)):
            practice[k] = v

    # Substitute the {PRACTICE['key']} expressions.
    def sub(match: re.Match) -> str:
        key = match.group(1)
        return practice.get(key, "{" + key + "}")
    resolved = re.sub(r"\{PRACTICE\['(\w+)'\]\}", sub, raw)
    return resolved.strip()


def build_modelfile(*, base_model: str, system_prompt: str,
                    temperature: float, num_predict: int) -> str:
    # Escape triple-quotes in system prompt by replacing with single-quote runs
    # (Modelfile uses """ as the delimiter for SYSTEM blocks).
    safe = system_prompt.replace('"""', '\\"\\"\\"')
    return (
        f"FROM {base_model}\n"
        f"\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER num_predict {num_predict}\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER repeat_penalty 1.1\n"
        f"\n"
        f'SYSTEM """{safe}"""\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model",
                        help="Base model name as Ollama knows it (e.g. qwen2.5:14b, llama3.3:8b)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=120)
    parser.add_argument("--out", help="Output Modelfile path. Default: Modelfile.<slug>-receptionist in project root")
    parser.add_argument("--create", action="store_true",
                        help="Also run `ollama create receptionist-<slug> -f <out>`")
    parser.add_argument("--name",
                        help="Override the created model name (default: receptionist-<base_slug>)")
    args = parser.parse_args()

    system_prompt = load_system_prompt()
    print(f"[mkmodel] loaded SYSTEM_PROMPT from bot.py ({len(system_prompt)} chars)", flush=True)

    base_slug = args.base_model.replace(":", "_").replace("/", "_")
    out_path = Path(args.out) if args.out else ROOT / f"Modelfile.{base_slug}-receptionist"
    modelfile = build_modelfile(
        base_model=args.base_model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    out_path.write_text(modelfile)
    print(f"[mkmodel] wrote {out_path}", flush=True)

    if args.create:
        model_name = args.name or f"receptionist-{base_slug}"
        cmd = ["ollama", "create", model_name, "-f", str(out_path)]
        print(f"[mkmodel] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode != 0:
            print(f"[mkmodel] ollama create exited with code {proc.returncode}", flush=True)
            return proc.returncode
        print(f"[mkmodel] success — try: ollama run {model_name}", flush=True)
        print(f"[mkmodel] or: python qa.py --model {model_name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
