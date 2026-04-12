"""Tests for the rule-based baseline agent."""

from __future__ import annotations

from baseline.run_baseline import run_all_tasks


class TestBaseline:
    def test_baseline_runs_without_error(self):
        results = run_all_tasks(seed=42)
        assert set(results.keys()) == {"task1", "task2", "task3"}
        for v in results.values():
            assert 0.0 <= v["score"] <= 1.0
