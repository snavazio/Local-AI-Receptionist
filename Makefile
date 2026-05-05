.PHONY: help test test-watch eval eval-fast eval-baseline watch trend bot bot-flows lint clean

PY ?= .venv/bin/python

help:
	@echo "Targets:"
	@echo "  test           - run pytest unit tests (~2s, no Ollama)"
	@echo "  eval           - run full 100-case eval at concurrency 2 (~10-15 min)"
	@echo "  eval-fast      - run full 100-case eval at concurrency 1 (~25 min, deterministic)"
	@echo "  eval-baseline  - run eval and force-overwrite baseline.json"
	@echo "  watch          - run watcher (eval + diff against baseline + update if no regression)"
	@echo "  trend          - show ASCII trend of pass rate / latency from history.jsonl"
	@echo "  bot            - run the live LLM-driven bot (bot.py) on port 8765"
	@echo "  bot-flows      - run the FSM-driven bot (bot_flows.py) on port 8765"
	@echo "  lint           - python ast syntax check on all .py files"
	@echo "  clean          - remove generated eval reports (not state files)"

test:
	$(PY) -m pytest tests/ -v

test-watch:
	@which fswatch >/dev/null 2>&1 || (echo "fswatch not installed; using simple loop" && \
		while true; do $(PY) -m pytest tests/ -q; sleep 5; done)
	@fswatch -l 1 -o bot.py eval/ tests/ 2>/dev/null | while read; do clear; $(PY) -m pytest tests/ -q; done

eval:
	$(PY) eval/run_eval.py --concurrency 2

eval-fast:
	$(PY) eval/run_eval.py --concurrency 1

eval-baseline:
	$(PY) eval/watch.py --update-baseline

watch:
	$(PY) eval/watch.py

trend:
	$(PY) eval/trend.py

bot:
	$(PY) bot.py

bot-flows:
	$(PY) bot_flows.py

lint:
	@for f in bot.py bot_flows.py eval/*.py tests/*.py; do \
		$(PY) -c "import ast; ast.parse(open('$$f').read())" && echo "OK $$f" || echo "FAIL $$f"; \
	done

clean:
	rm -f eval/report*.md eval/run*.log eval/regression_report.md
	@echo "(eval/baseline.json and eval/history.jsonl preserved)"
