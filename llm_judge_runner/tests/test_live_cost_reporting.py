from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.cost_estimation import (
    calculate_live_cost,
    cost_row_from_result,
    summarize_live_costs,
    write_live_cost_reports,
)
from src.runner import _prepare_run_dir


class OutputRoutingTest(unittest.TestCase):
    def test_dry_and_live_runs_use_separate_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(results_root=Path(temp_dir))
            _, dry_dir = _prepare_run_dir(settings, True, "same_name")
            _, live_dir = _prepare_run_dir(settings, False, "same_name")

            self.assertEqual(dry_dir, Path(temp_dir) / "dryruns" / "same_name")
            self.assertEqual(live_dir, Path(temp_dir) / "live_runs" / "same_name")
            self.assertNotEqual(dry_dir, live_dir)


class LiveCostTest(unittest.TestCase):
    def test_qwen_implicit_cached_usage_is_costed_separately(self) -> None:
        cost = calculate_live_cost(
            adapter="openai_compatible_chat",
            usage={
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens": 200,
                "total_tokens": 1200,
            },
            pricing={
                "input_per_million": 2.0,
                "cached_input_per_million": 0.25,
                "cache_write_per_million": None,
                "output_per_million": 6.0,
            },
            api_attempts=1,
            provider_response_received=True,
        )

        usage = cost["normalized_usage"]
        self.assertEqual(usage["uncached_input_tokens"], 200)
        self.assertEqual(usage["cached_input_tokens"], 800)
        self.assertEqual(usage["output_tokens"], 200)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.0018)
        self.assertTrue(cost["cost_complete"])

    def test_deepseek_cache_hit_and_miss_usage_is_costed_separately(self) -> None:
        cost = calculate_live_cost(
            adapter="openai_compatible_chat",
            usage={
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 750,
                "prompt_cache_miss_tokens": 250,
                "completion_tokens": 200,
                "completion_tokens_details": {"reasoning_tokens": 120},
                "total_tokens": 1200,
            },
            pricing={
                "input_per_million": 1.32,
                "cached_input_per_million": 0.044,
                "cache_write_per_million": None,
                "output_per_million": 3.96,
            },
            api_attempts=1,
            provider_response_received=True,
        )

        usage = cost["normalized_usage"]
        self.assertEqual(usage["uncached_input_tokens"], 250)
        self.assertEqual(usage["cached_input_tokens"], 750)
        self.assertEqual(usage["output_tokens"], 200)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.001155)
        self.assertTrue(cost["cost_complete"])

    def test_openai_cached_usage_is_not_double_counted(self) -> None:
        cost = calculate_live_cost(
            adapter="openai_responses",
            usage={
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 600},
                "output_tokens": 200,
            },
            pricing={
                "input_per_million": 1.0,
                "cached_input_per_million": 0.1,
                "cache_write_per_million": None,
                "output_per_million": 2.0,
            },
            api_attempts=1,
            provider_response_received=True,
        )

        self.assertEqual(cost["normalized_usage"]["uncached_input_tokens"], 400)
        self.assertEqual(cost["normalized_usage"]["cached_input_tokens"], 600)
        self.assertAlmostEqual(cost["total_cost_usd"], 0.00086)
        self.assertTrue(cost["cost_complete"])

    def test_anthropic_cache_categories_are_separate(self) -> None:
        cost = calculate_live_cost(
            adapter="anthropic_messages",
            usage={
                "input_tokens": 100,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 300,
                "output_tokens": 50,
            },
            pricing={
                "input_per_million": 1.0,
                "cached_input_per_million": 0.1,
                "cache_write_per_million": 1.25,
                "output_per_million": 2.0,
            },
            api_attempts=1,
            provider_response_received=True,
        )

        usage = cost["normalized_usage"]
        self.assertEqual(usage["uncached_input_tokens"], 100)
        self.assertEqual(usage["cached_input_tokens"], 200)
        self.assertEqual(usage["cache_write_input_tokens"], 300)
        self.assertTrue(cost["cost_complete"])

    def test_missing_failed_attempt_usage_marks_summary_incomplete(self) -> None:
        cost = calculate_live_cost(
            adapter="openai_responses",
            usage={},
            pricing={
                "input_per_million": 1.0,
                "cached_input_per_million": 0.1,
                "cache_write_per_million": None,
                "output_per_million": 2.0,
            },
            api_attempts=2,
            provider_response_received=False,
        )
        record = {
            "execution_key": "key",
            "run_id": "run",
            "collected_at_utc": "now",
            "case": {"case_id": "case"},
            "experiment": {"test_id": "T0"},
            "judge": {"model_key": "model"},
            "response": {"collection_attempt": 1, "status": "api_error", "api_attempts": 2},
            "cost": cost,
        }
        summary = summarize_live_costs(
            [cost_row_from_result(record)],
            {"currency": "USD", "prices_as_of": "2026-09-02"},
            "hash",
        )

        self.assertFalse(summary["summary"]["cost_complete"])
        self.assertIsNone(summary["summary"]["total_cost_usd"])
        self.assertEqual(summary["summary"]["incomplete_cost_record_count"], 1)

    def test_report_files_are_written_from_result_records(self) -> None:
        cost = calculate_live_cost(
            adapter="openai_responses",
            usage={"input_tokens": 100, "output_tokens": 20},
            pricing={
                "input_per_million": 1.0,
                "cached_input_per_million": 0.1,
                "cache_write_per_million": None,
                "output_per_million": 2.0,
            },
            api_attempts=1,
            provider_response_received=True,
        )
        record = {
            "execution_key": "key",
            "run_id": "run",
            "collected_at_utc": "now",
            "case": {"case_id": "case"},
            "experiment": {"test_id": "T0"},
            "judge": {"model_key": "model"},
            "response": {"collection_attempt": 1, "status": "ok", "api_attempts": 1},
            "cost": cost,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            write_live_cost_reports(
                run_dir,
                [record],
                {"currency": "USD", "prices_as_of": "2026-09-02"},
                "hash",
            )

            self.assertTrue((run_dir / "costs.jsonl").exists())
            self.assertTrue((run_dir / "cost_summary.json").exists())
            self.assertTrue((run_dir / "cost_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
