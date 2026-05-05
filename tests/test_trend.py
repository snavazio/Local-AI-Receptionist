"""Tests for eval/trend.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.trend import sparkline, load_history  # noqa: E402


class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_constant(self):
        # All values equal -> midline character
        out = sparkline([5, 5, 5, 5])
        assert len(out) == 4
        assert len(set(out)) == 1

    def test_monotonic_increasing(self):
        out = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
        # First char should be lowest, last char should be highest
        assert out[0] != out[-1]
        # Non-decreasing across consecutive
        chars = " ▁▂▃▄▅▆▇█"
        idx = [chars.index(c) for c in out]
        for i in range(1, len(idx)):
            assert idx[i] >= idx[i - 1]

    def test_length_matches(self):
        assert len(sparkline([1, 2, 3])) == 3
        assert len(sparkline([1] * 20)) == 20


class TestLoadHistory:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_history(tmp_path / "nope.jsonl") == []

    def test_valid_jsonl(self, tmp_path):
        path = tmp_path / "h.jsonl"
        path.write_text(
            json.dumps({"ts": "2026-01-01", "passed": 75, "total": 100}) + "\n" +
            json.dumps({"ts": "2026-01-02", "passed": 78, "total": 100}) + "\n"
        )
        rows = load_history(path)
        assert len(rows) == 2
        assert rows[0]["passed"] == 75
        assert rows[1]["passed"] == 78

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "h.jsonl"
        path.write_text(
            json.dumps({"a": 1}) + "\n\n" +
            json.dumps({"a": 2}) + "\n"
        )
        rows = load_history(path)
        assert len(rows) == 2

    def test_skips_malformed(self, tmp_path):
        path = tmp_path / "h.jsonl"
        path.write_text(
            json.dumps({"a": 1}) + "\n" +
            "not json at all\n" +
            json.dumps({"a": 3}) + "\n"
        )
        rows = load_history(path)
        assert len(rows) == 2
        assert rows[0]["a"] == 1
        assert rows[1]["a"] == 3
