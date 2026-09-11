from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.cases import load_cases
from src.config import ModelConfig, Settings, load_models, load_settings
from src.runner import run_collection


EXECUTION_ARGUMENTS = {
    "--models",
    "--tests",
    "--case-ids",
    "--repeats",
    "--dry-run",
    "--run-id",
    "--resume",
    "--retry-api-errors",
}


def csv_list(value: str | list[str] | None) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return value
    return [x.strip() for x in value.split(",") if x.strip()]


def is_interactive_invocation(argv: list[str]) -> bool:
    """Use the questionnaire when no execution-selection/control argument was supplied."""
    return not any(token.split("=", 1)[0] in EXECUTION_ARGUMENTS for token in argv)


def settings_argument_was_supplied(argv: list[str]) -> bool:
    return any(token.split("=", 1)[0] == "--settings" for token in argv)


def _resolve_settings_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def collect_interactive_settings(
    project_root: Path,
    default_value: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> tuple[Path, Settings]:
    """Ask for a settings file and return it only after its JSON structure loads."""
    settings_dir = project_root / "config" / "settings"
    discovered = sorted(settings_dir.glob("*.json")) if settings_dir.exists() else []
    if discovered:
        output_fn("Available settings files:")
        for index, path in enumerate(discovered, start=1):
            output_fn(f"  {index}. {path.relative_to(project_root)}")

    while True:
        raw = input_fn(f"Settings file [{default_value}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(discovered):
            settings_path = discovered[int(raw) - 1]
        else:
            settings_path = _resolve_settings_path(project_root, raw or default_value)

        if not settings_path.is_file():
            output_fn(f"Settings file not found: {settings_path}")
            continue
        try:
            settings = load_settings(project_root, settings_path)
        except Exception as exc:
            output_fn(f"Invalid settings file: {type(exc).__name__}: {exc}")
            continue
        output_fn(f"Using settings: {settings_path}")
        return settings_path, settings


def _ask_choice(
    question: str,
    choices: set[str],
    default: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> str:
    while True:
        value = input_fn(question).strip().lower() or default
        if value in choices:
            return value
        output_fn(f"Enter one of: {', '.join(sorted(choices))}.")


def _ask_csv_selection(
    question: str,
    available: set[str],
    default_values: list[str] | None,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> list[str] | None:
    while True:
        raw = input_fn(question).strip()
        if not raw or raw.lower() == "all":
            return default_values
        values = csv_list(raw) or []
        unknown = sorted(set(values) - available)
        if not unknown:
            return values
        output_fn(f"Unknown value(s): {', '.join(unknown)}")


def _ask_repeats(
    default: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> int:
    while True:
        raw = input_fn(f"Repeats [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return value
        output_fn("Repeats must be a positive integer.")


def collect_interactive_arguments(
    args: argparse.Namespace,
    settings: Settings,
    models: dict[str, ModelConfig],
    cases: list[dict[str, Any]],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> argparse.Namespace:
    """Collect and validate execution options for a bare command invocation."""
    output_fn("Interactive experiment setup (dry run is the safe default).")
    mode = _ask_choice(
        "Mode [dry/live/resume] (dry): ",
        {"dry", "live", "resume"},
        "dry",
        input_fn,
        output_fn,
    )
    if mode == "resume":
        resume_kind = _ask_choice(
            "Resume which kind [live/dry] (live): ",
            {"live", "dry"},
            "live",
            input_fn,
            output_fn,
        )
        args.resume = True
        args.dry_run = resume_kind == "dry"
    else:
        args.resume = False
        args.dry_run = mode == "dry"

    enabled_keys = [key for key, model in models.items() if model.enabled]
    output_fn("Configured models:")
    for key, model in models.items():
        state = "enabled" if model.enabled else "disabled"
        output_fn(f"  {key}: {model.display_name} ({state})")
    args.models = _ask_csv_selection(
        "Model key(s), comma-separated [all enabled]: ",
        set(models),
        None,
        input_fn,
        output_fn,
    )
    chosen_model_keys = args.models or enabled_keys
    if not chosen_model_keys:
        raise ValueError("No models are enabled; select a configured model explicitly.")

    available_tests = sorted({str(case["test_id"]) for case in cases})
    output_fn(f"Available test IDs: {', '.join(available_tests)}")
    args.tests = _ask_csv_selection(
        "Test IDs, comma-separated [all]: ",
        set(available_tests),
        None,
        input_fn,
        output_fn,
    )
    args.case_ids = _ask_csv_selection(
        "Exact case IDs, comma-separated [all]: ",
        {str(case["case_id"]) for case in cases},
        None,
        input_fn,
        output_fn,
    )
    args.repeats = _ask_repeats(settings.default_repeats, input_fn, output_fn)

    while True:
        requested_run_id = input_fn(
            "Existing run ID to resume: " if args.resume else "Run ID [automatic timestamp]: "
        ).strip()
        if requested_run_id or not args.resume:
            break
        output_fn("A run ID is required when resuming.")
    args.run_id = requested_run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    args.retry_api_errors = False
    if args.resume:
        retry = _ask_choice(
            "Retry recorded API errors [y/N]: ",
            {"y", "n"},
            "n",
            input_fn,
            output_fn,
        )
        args.retry_api_errors = retry == "y"

    selected_cases = cases
    if args.tests:
        selected_tests = set(args.tests)
        selected_cases = [case for case in selected_cases if case["test_id"] in selected_tests]
    if args.case_ids:
        selected_ids = set(args.case_ids)
        selected_cases = [case for case in selected_cases if case["case_id"] in selected_ids]
    if not selected_cases:
        raise ValueError("The selected filters contain no cases.")

    run_kind = "dryruns" if args.dry_run else "live_runs"
    run_dir = settings.results_root / run_kind / args.run_id
    planned_requests = len(selected_cases) * len(chosen_model_keys) * args.repeats
    output_fn("")
    output_fn("Run summary:")
    output_fn(f"  mode: {'dry run' if args.dry_run else 'LIVE API run'}")
    output_fn(f"  models: {', '.join(chosen_model_keys)}")
    output_fn(f"  cases: {len(selected_cases)}")
    output_fn(f"  repeats: {args.repeats}")
    output_fn(f"  planned requests: {planned_requests}")
    output_fn(f"  output: {run_dir}")

    if not args.dry_run:
        confirmation = input_fn("Type LIVE to allow model API calls: ").strip()
        if confirmation != "LIVE":
            raise RuntimeError("Live run cancelled; confirmation was not entered.")
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect LLM-as-a-Judge outputs for prepared experiment cases.")
    parser.add_argument("--settings", default="config/settings/settings.json")
    parser.add_argument("--models", help="Comma-separated model keys. Omit to use all enabled models.")
    parser.add_argument("--tests", help="Comma-separated test IDs, e.g. T1,T3,T7")
    parser.add_argument("--case-ids", help="Comma-separated exact case IDs")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and write manifest without API calls.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run directory name.")
    parser.add_argument("--resume", action="store_true", help="Skip execution keys already present in the chosen run directory.")
    parser.add_argument(
        "--retry-api-errors",
        action="store_true",
        help="With --resume, retry conditions whose only recorded result is api_error.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    interactive = is_interactive_invocation(raw_argv)
    if interactive:
        if not sys.stdin.isatty():
            parser.error(
                "interactive setup requires a terminal; supply execution arguments for non-interactive use"
            )
        try:
            if settings_argument_was_supplied(raw_argv):
                settings_path = _resolve_settings_path(project_root, args.settings)
                settings = load_settings(project_root, settings_path)
                print(f"Using settings: {settings_path}")
            else:
                settings_path, settings = collect_interactive_settings(
                    project_root,
                    args.settings,
                )
                args.settings = str(settings_path)
            args = collect_interactive_arguments(
                args,
                settings,
                load_models(settings.models_file),
                load_cases(settings.cases_file),
            )
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive setup cancelled.")
            raise SystemExit(130)
        except (RuntimeError, ValueError) as exc:
            print(f"Interactive setup stopped: {exc}")
            raise SystemExit(1)
    else:
        settings_path = _resolve_settings_path(project_root, args.settings)
        settings = load_settings(project_root, settings_path)

    run_dir = run_collection(
        project_root=project_root,
        settings=settings,
        selected_model_keys=csv_list(args.models),
        selected_tests=csv_list(args.tests),
        selected_case_ids=csv_list(args.case_ids),
        repeats=args.repeats,
        dry_run=args.dry_run,
        run_id=args.run_id,
        resume=args.resume,
        retry_api_errors=args.retry_api_errors,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
