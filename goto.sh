#!/usr/bin/env bash
# Quick-jump to the receptionist project + activate its venv.
#
# Usage:
#   source ~/projects/receptionist/pipecat-quickstart-phone-bot/goto.sh
# or, if you've put a symlink/alias in your shell rc:
#   source goto
#
# The `source` is required — `bash goto.sh` runs in a subshell that exits
# immediately, so your terminal won't actually change directories.
#
# After sourcing you'll be in the project root with .venv active. Common next
# commands print at the bottom for reference.

# Detect whether we were sourced or executed. If executed, exit with hint.
(return 0 2>/dev/null) || {
    echo "ERROR: this script must be sourced, not executed."
    echo "Run instead:  source $0"
    exit 1
}

PROJECT_DIR="/home/snavazio/projects/receptionist/pipecat-quickstart-phone-bot"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    return 1
fi

cd "$PROJECT_DIR" || return 1

# Activate the venv if it exists and isn't already active.
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$PROJECT_DIR/.venv" ]; then
        # shellcheck source=/dev/null
        source "$PROJECT_DIR/.venv/bin/activate"
    fi
fi

# Friendly banner.
echo
echo "→ $(pwd)"
[ -n "$VIRTUAL_ENV" ] && echo "→ venv active: $VIRTUAL_ENV"
echo
echo "Common commands:"
echo "  python qa.py --smoke           # ~5 min iteration loop"
echo "  python qa.py                   # full 356-case eval (~30-50 min)"
echo "  python qa.py --model qwen2.5:14b,qwen2.5:7b   # multi-model compare"
echo "  python -m pytest -q            # unit tests (~2 sec)"
echo "  python eval/trend.py           # sparkline trend over runs"
echo "  tail -f qa_runs/qa_*_progress.log    # live watch a running eval"
echo
