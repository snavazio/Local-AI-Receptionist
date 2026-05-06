.DEFAULT_GOAL := help

# Self-documenting Makefile: any target line ending in `## description` shows
# up in `make help`. Keep descriptions short.

PY ?= .venv/bin/python
QA_RUNS_DIR := qa_runs

# ============================================================================
# QA pipeline (qa.py — recommended for most workflows)
# ============================================================================

.PHONY: qa
qa: qa-full   ## Run full QA pipeline (alias for qa-full)

.PHONY: qa-smoke
qa-smoke:   ## Smoke set — ~31 cases, ~5 min. Use for fast iteration.
	$(PY) qa.py --smoke

.PHONY: qa-full
qa-full:   ## Full eval — all 356 cases (~30-50 min at concurrency=1)
	$(PY) qa.py

.PHONY: qa-fast
qa-fast:   ## Full eval at concurrency=2 (faster wall, slightly noisier latency)
	$(PY) qa.py --concurrency 2

.PHONY: qa-judge
qa-judge:   ## Smoke + LLM-as-judge scoring (conversation quality)
	$(PY) qa.py --smoke --judge

.PHONY: qa-multi
qa-multi:   ## Multi-model bake-off. Set MODELS=a,b,c (default qwen 14b+7b)
	$(PY) qa.py --model "$(or $(MODELS),qwen2.5:14b,qwen2.5:7b)"

.PHONY: qa-no-update
qa-no-update:   ## Run eval but DON'T update baseline.json (ad-hoc check)
	$(PY) qa.py --no-update

# ============================================================================
# Tests (deterministic, no LLM)
# ============================================================================

.PHONY: test
test:   ## Run unit tests (~2 sec, no GPU/LLM)
	$(PY) -m pytest -q

.PHONY: test-verbose
test-verbose:   ## Unit tests with verbose per-test output
	$(PY) -m pytest -v

.PHONY: test-watch
test-watch:   ## Re-run tests on file change (fswatch or simple loop)
	@which fswatch >/dev/null 2>&1 && \
	  fswatch -l 1 -o bot.py eval/ tests/ | while read; do clear; $(PY) -m pytest -q; done || \
	  while true; do $(PY) -m pytest -q; sleep 5; done

# ============================================================================
# Lower-level eval commands (for finer-grained control than qa.py)
# ============================================================================

.PHONY: eval
eval:   ## Direct eval at concurrency=2 (no qa.py wrapper, no progress log)
	$(PY) eval/run_eval.py --concurrency 2

.PHONY: eval-smoke
eval-smoke:   ## Smoke set via run_eval.py directly (no qa.py wrapper)
	$(PY) eval/run_eval.py --smoke

.PHONY: eval-fast
eval-fast:   ## Direct eval at concurrency=1 (deterministic latency)
	$(PY) eval/run_eval.py --concurrency 1

.PHONY: watch
watch:   ## Run watcher (eval + diff vs baseline + update if no regression)
	$(PY) eval/watch.py

.PHONY: watch-update
watch-update:   ## Run watcher and force-overwrite baseline.json
	$(PY) eval/watch.py --update-baseline

# ============================================================================
# Inspection / observability
# ============================================================================

.PHONY: trend
trend:   ## ASCII trend over the last 10 runs from history.jsonl
	$(PY) eval/trend.py

.PHONY: last-report
last-report:   ## Print the most recent qa report
	@latest=$$(ls -t $(QA_RUNS_DIR)/qa_*[0-9]*.md 2>/dev/null | grep -v progress | grep -v comparison | head -1); \
	if [ -z "$$latest" ]; then echo "No qa.py reports yet — run 'make qa-smoke'."; exit 1; fi; \
	echo "→ $$latest"; echo; cat "$$latest"

.PHONY: progress
progress:   ## Tail the most recent progress log (live mid-run watch)
	@latest=$$(ls -t $(QA_RUNS_DIR)/qa_*_progress.log 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No progress log yet — start a qa.py run."; exit 1; fi; \
	echo "→ tail -f $$latest"; tail -f "$$latest"

.PHONY: status
status:   ## Snapshot — git, ollama loaded models, GPU, eval cases, baseline
	@echo
	@echo "── Git ─────────────────────────────────────"
	@git log --oneline -3 2>/dev/null || echo "  (not a git repo)"
	@echo
	@echo "── Ollama (loaded) ─────────────────────────"
	@curl -s http://localhost:11434/api/ps 2>/dev/null | $(PY) -c "import sys,json; d=json.load(sys.stdin); models=d.get('models',[]); [print(f'  {m[\"name\"]:<25} {int(m.get(\"size_vram\",0)/1e9)} GB') for m in models]; print('  (none loaded)' if not models else '')" 2>/dev/null || echo "  (Ollama not reachable)"
	@echo
	@echo "── GPU ─────────────────────────────────────"
	@nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi not available)"
	@echo
	@echo "── Eval cases ──────────────────────────────"
	@$(PY) -c "import yaml; cases=yaml.safe_load(open('eval/cases.yaml')); from collections import Counter; c=Counter(x.get('category','?') for x in cases); print(f'  {len(cases)} total cases across {len(c)} categories')"
	@if [ -f eval/baseline.json ]; then \
	  $(PY) -c "import json; b=json.load(open('eval/baseline.json')); print(f'  baseline: {b[\"passed\"]}/{b[\"total\"]} passing ({b.get(\"ts\",\"?\")})')"; \
	else echo "  baseline: not yet established"; fi
	@echo
	@echo "── Running processes ───────────────────────"
	@pgrep -af "qa.py|run_eval" 2>/dev/null | grep -v -E "(grep|pgrep)" | sed 's/^/  /' || echo "  (none)"
	@echo

# ============================================================================
# Authoring helpers
# ============================================================================

.PHONY: new-case
new-case:   ## Generate a case YAML from a call log. Usage: make new-case CALL=...
	@if [ -z "$(CALL)" ]; then \
	  echo "Usage: make new-case CALL=call_logs/call_TIMESTAMP.json"; exit 1; \
	fi
	$(PY) scripts/case_from_call.py "$(CALL)"

.PHONY: new-model
new-model:   ## Bake a receptionist Modelfile. Usage: make new-model BASE=qwen2.5:14b [CREATE=1]
	@if [ -z "$(BASE)" ]; then \
	  echo "Usage: make new-model BASE=qwen2.5:14b [CREATE=1]"; exit 1; \
	fi
	@if [ "$(CREATE)" = "1" ]; then \
	  $(PY) scripts/make_receptionist_model.py "$(BASE)" --create; \
	else \
	  $(PY) scripts/make_receptionist_model.py "$(BASE)"; \
	fi

# ============================================================================
# Run the live bot (audio pipeline)
# ============================================================================

.PHONY: bot
bot:   ## Run the live LLM-driven bot on port 8765
	$(PY) bot.py

.PHONY: bot-flows
bot-flows:   ## Run the FSM-driven bot variant
	$(PY) bot_flows.py

# ============================================================================
# Lint / dev hygiene
# ============================================================================

.PHONY: lint
lint:   ## Python AST syntax check on all .py files
	@for f in bot.py bot_flows.py qa.py eval/*.py tests/*.py scripts/*.py config/*.py; do \
	  [ -f "$$f" ] && ($(PY) -c "import ast; ast.parse(open('$$f').read())" && echo "OK $$f" || echo "FAIL $$f"); \
	done

.PHONY: lint-yaml
lint-yaml:   ## Validate eval/cases.yaml (parse + dup ids + descriptions)
	@$(PY) -c "import yaml; cases=yaml.safe_load(open('eval/cases.yaml')); ids=[c['id'] for c in cases]; dups=[x for x in set(ids) if ids.count(x)>1]; assert not dups, f'dups: {dups}'; assert all(c.get('description','').strip() for c in cases); print(f'OK — {len(cases)} cases, no dups, all have descriptions')"

.PHONY: precommit-install
precommit-install:   ## One-time: install pre-commit hooks
	$(PY) -m pip install pre-commit && pre-commit install

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:   ## Remove ephemeral eval reports (keeps baseline.json + history.jsonl)
	rm -f eval/report*.md eval/run*.log eval/regression_report.md
	@echo "(eval/baseline.json and eval/history.jsonl preserved)"

.PHONY: clean-old-runs
clean-old-runs:   ## Delete qa_runs/ files older than 30 days
	@find $(QA_RUNS_DIR) -type f -mtime +30 -name "qa_*" -print -delete 2>/dev/null || true
	@echo "Done."

# ============================================================================
# Help — auto-generated from `## description` comments
# ============================================================================

.PHONY: help
help:   ## Show this help (default target)
	@echo
	@echo "  ╔══════════════════════════════════════════╗"
	@echo "  ║   Receptionist QA — common commands      ║"
	@echo "  ╚══════════════════════════════════════════╝"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Quick start:"
	@echo "    make qa-smoke              # 5-min iteration loop"
	@echo "    make status                # snapshot of running state"
	@echo "    ./dashboard.sh             # interactive menu"
	@echo
