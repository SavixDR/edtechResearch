from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings
from src.runner import run_collection


def csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LLM-as-a-Judge outputs for prepared experiment cases.")
    parser.add_argument("--settings", default="config/settings.json")
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        settings_path = project_root / settings_path
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
