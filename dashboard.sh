#!/usr/bin/env bash
# Interactive dashboard for the receptionist QA project.
#
# Wraps the Makefile so you don't have to remember target names. Each option
# shells out to `make <target>` (or a small inline command) and returns to the
# menu when done. Pick `q` (or Ctrl-C) to quit.
#
# Usage:
#   ./dashboard.sh
#
# Tip: pair this with goto.sh —
#   source goto.sh && ./dashboard.sh

set -u

cd "$(dirname "$0")" || exit 1

# ANSI helpers (kept tiny — no tput dep).
B='\033[1m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; N='\033[0m'

banner() {
    clear
    printf "${B}"
    cat <<'EOF'
  ╔════════════════════════════════════════════════════╗
  ║      Receptionist QA — Interactive Dashboard       ║
  ╚════════════════════════════════════════════════════╝
EOF
    printf "${N}\n"
    # Quick state line: git branch + last commit + venv.
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
    last=$(git log -1 --pretty=format:'%h %s' 2>/dev/null || echo "(no git)")
    venv="${VIRTUAL_ENV:-(none)}"
    printf "  ${C}branch${N}  %s     ${C}venv${N}  %s\n" "$branch" "$venv"
    printf "  ${C}last${N}    %s\n\n" "$last"
}

pause() {
    printf "\n${Y}Press Enter to return to dashboard...${N}"
    read -r _
}

run() {
    # $1 = pretty label, rest = command
    local label="$1"; shift
    printf "\n${G}▶ %s${N}\n${C}\$ %s${N}\n\n" "$label" "$*"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        printf "\n${R}✗ exited %d${N}\n" $rc
    else
        printf "\n${G}✓ done${N}\n"
    fi
}

menu() {
    cat <<EOF
  ${B}QA / eval${N}
    1)  Smoke (~5 min, 31 cases)             — make qa-smoke
    2)  Full eval (~30-50 min, 356 cases)    — make qa-full
    3)  Smoke + LLM-as-judge                 — make qa-judge
    4)  Multi-model bake-off                 — make qa-multi
    5)  Watcher (regression gate)            — make watch

  ${B}Tests${N}
    6)  Unit tests (~2 sec)                  — make test
    7)  Unit tests verbose                   — make test-verbose
    8)  Lint (AST + cases.yaml)              — make lint && make lint-yaml

  ${B}Inspect${N}
    9)  Snapshot status                      — make status
   10)  Trend (last 10 runs)                 — make trend
   11)  Last QA report                       — make last-report
   12)  Tail live progress log               — make progress

  ${B}Authoring${N}
   13)  New case from call log               — prompts for CALL=...
   14)  Bake new receptionist Modelfile      — prompts for BASE=...

  ${B}Bot${N}
   15)  Run live LLM bot                     — make bot
   16)  Run FSM-driven bot                   — make bot-flows

  ${B}Housekeeping${N}
   17)  Clean ephemeral reports              — make clean
   18)  Delete qa_runs/ older than 30 days   — make clean-old-runs
   19)  Show full make help                  — make help

    q)  Quit
EOF
}

main_loop() {
    while true; do
        banner
        menu
        printf "\n  ${C}choose>${N} "
        read -r choice
        case "$choice" in
            1)  run "Smoke eval"          make qa-smoke ;;
            2)  run "Full eval"           make qa-full ;;
            3)  run "Smoke + judge"       make qa-judge ;;
            4)  printf "  Models (comma-separated, blank=default): "
                read -r models
                if [ -n "$models" ]; then run "Multi-model" make qa-multi MODELS="$models"
                else run "Multi-model" make qa-multi; fi ;;
            5)  run "Watcher"             make watch ;;
            6)  run "Unit tests"          make test ;;
            7)  run "Unit tests verbose"  make test-verbose ;;
            8)  run "Lint"                bash -c "make lint && make lint-yaml" ;;
            9)  run "Status"              make status ;;
            10) run "Trend"               make trend ;;
            11) run "Last report"         make last-report ;;
            12) run "Tail progress"       make progress ;;
            13) printf "  Path to call log (call_logs/call_*.json): "
                read -r call
                if [ -n "$call" ]; then run "New case" make new-case CALL="$call"
                else printf "${R}cancelled${N}\n"; fi ;;
            14) printf "  Base model (e.g. qwen2.5:14b): "
                read -r base
                printf "  Run 'ollama create' too? [y/N]: "
                read -r doit
                if [ -n "$base" ]; then
                    if [ "$doit" = "y" ] || [ "$doit" = "Y" ]; then
                        run "Bake model" make new-model BASE="$base" CREATE=1
                    else
                        run "Bake Modelfile" make new-model BASE="$base"
                    fi
                else printf "${R}cancelled${N}\n"; fi ;;
            15) run "Live bot"            make bot ;;
            16) run "FSM bot"             make bot-flows ;;
            17) run "Clean"               make clean ;;
            18) run "Clean old runs"      make clean-old-runs ;;
            19) run "Help"                make help ;;
            q|Q|quit|exit)
                printf "\n${G}bye${N}\n"; exit 0 ;;
            "")
                continue ;;
            *)
                printf "${R}unknown choice: %s${N}\n" "$choice" ;;
        esac
        pause
    done
}

main_loop
