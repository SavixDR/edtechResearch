from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_experiments import (
    collect_interactive_arguments,
    collect_interactive_settings,
    is_interactive_invocation,
    settings_argument_was_supplied,
)


def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        settings="settings.json",
        models=None,
        tests=None,
        case_ids=None,
        repeats=None,
        dry_run=False,
        run_id=None,
        resume=False,
        retry_api_errors=False,
    )


class InteractiveDetectionTest(unittest.TestCase):
    def test_bare_or_settings_only_invocation_is_interactive(self) -> None:
        self.assertTrue(is_interactive_invocation([]))
        self.assertTrue(is_interactive_invocation(["--settings", "custom.json"]))
        self.assertTrue(is_interactive_invocation(["--settings=custom.json"]))

    def test_execution_argument_keeps_cli_non_interactive(self) -> None:
        self.assertFalse(is_interactive_invocation(["--models", "model_a"]))
        self.assertFalse(is_interactive_invocation(["--dry-run"]))

    def test_explicit_settings_argument_is_detected(self) -> None:
        self.assertTrue(settings_argument_was_supplied(["--settings", "custom.json"]))
        self.assertTrue(settings_argument_was_supplied(["--settings=custom.json"]))
        self.assertFalse(settings_argument_was_supplied([]))


class InteractiveSettingsTest(unittest.TestCase):
    @staticmethod
    def _settings_payload() -> dict[str, object]:
        return {
            "experiment_name": "test",
            "suite_version": "1",
            "prompt_version": "V3",
            "cases_file": "cases.jsonl",
            "models_file": "models.json",
            "prompt_variants_file": "prompts.json",
            "assignments_file": "assignments.json",
            "pricing_file": "pricing.json",
            "results_root": "outputs",
        }

    def test_number_selects_discovered_settings_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_dir = project_root / "config" / "settings"
            settings_dir.mkdir(parents=True)
            selected_path = settings_dir / "settings.json"
            selected_path.write_text(json.dumps(self._settings_payload()), encoding="utf-8")

            path, settings = collect_interactive_settings(
                project_root,
                "config/settings/settings.json",
                input_fn=lambda _question: "1",
                output_fn=lambda _message: None,
            )

            self.assertEqual(path, selected_path)
            self.assertEqual(settings.experiment_name, "test")

    def test_missing_path_is_rejected_and_asked_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            valid_path = project_root / "valid.json"
            valid_path.write_text(json.dumps(self._settings_payload()), encoding="utf-8")
            responses = iter(["missing.json", "valid.json"])
            messages: list[str] = []

            path, _settings = collect_interactive_settings(
                project_root,
                "default.json",
                input_fn=lambda _question: next(responses),
                output_fn=messages.append,
            )

            self.assertEqual(path, valid_path)
            self.assertTrue(any("not found" in message for message in messages))


class InteractiveArgumentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.models = {
            "model_a": SimpleNamespace(display_name="Model A", enabled=True),
            "model_b": SimpleNamespace(display_name="Model B", enabled=False),
        }
        self.cases = [
            {"case_id": "case_1", "test_id": "T0"},
            {"case_id": "case_2", "test_id": "T1"},
        ]

    def _run(self, responses: list[str]) -> argparse.Namespace:
        iterator = iter(responses)
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(default_repeats=1, results_root=Path(temp_dir))
            return collect_interactive_arguments(
                base_args(),
                settings,
                self.models,
                self.cases,
                input_fn=lambda _question: next(iterator),
                output_fn=lambda _message: None,
            )

    def test_blank_values_create_safe_dry_run(self) -> None:
        args = self._run(["", "", "", "", "", "dry_test"])
        self.assertTrue(args.dry_run)
        self.assertFalse(args.resume)
        self.assertIsNone(args.models)
        self.assertEqual(args.repeats, 1)
        self.assertEqual(args.run_id, "dry_test")

    def test_live_mode_requires_exact_confirmation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Live run cancelled"):
            self._run(["live", "model_a", "T0", "case_1", "2", "live_test", "yes"])

    def test_live_mode_accepts_explicit_confirmation(self) -> None:
        args = self._run(["live", "model_a", "T0", "case_1", "2", "live_test", "LIVE"])
        self.assertFalse(args.dry_run)
        self.assertEqual(args.models, ["model_a"])
        self.assertEqual(args.tests, ["T0"])
        self.assertEqual(args.case_ids, ["case_1"])
        self.assertEqual(args.repeats, 2)

    def test_resume_requires_id_and_can_retry_api_errors(self) -> None:
        args = self._run([
            "resume", "live", "model_a", "T0", "case_1", "1", "", "existing_run", "y", "LIVE"
        ])
        self.assertTrue(args.resume)
        self.assertFalse(args.dry_run)
        self.assertTrue(args.retry_api_errors)
        self.assertEqual(args.run_id, "existing_run")


if __name__ == "__main__":
    unittest.main()
