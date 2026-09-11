from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.cases import load_cases, render_conversation
from src.codebook import load_codebook
from src.config import load_models, load_settings
from src.cost_estimation import summarize_costs
from src.prompts import load_prompt_variants, render_system_prompt, render_user_prompt
from src.token_counting import offline_request_tokens, openai_request_tokens
from src.utils import load_json, project_path, sha256_json, sha256_text, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count T0 prompt tokens and estimate USD cost without calling any generation endpoint."
    )
    parser.add_argument("--settings", default="config/settings/settings_pilot_t0.json")
    parser.add_argument("--model", required=True, help="One model key from config/models.json")
    parser.add_argument("--mode", choices=("provider-count", "offline-estimate", "both"))
    parser.add_argument("--case-ids", help="Optional comma-separated case IDs")
    parser.add_argument("--run-id", help="Optional output directory name")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_assignment(project_root: Path, assignments: dict[str, Any], assignment_id: str) -> dict[str, Any]:
    if assignment_id not in assignments:
        raise KeyError(f"No assignment configuration for {assignment_id}")
    config = assignments[assignment_id]
    codebook_path = project_path(project_root, config["codebook_file"])
    if codebook_path is None or not codebook_path.exists():
        raise FileNotFoundError(f"Codebook not found for {assignment_id}: {codebook_path}")
    result = load_codebook(codebook_path, assignment_id)
    for key in ("assignment_reference", "human_validated_few_shot_examples"):
        inline_value = config.get(key)
        file_value = config.get(f"{key}_file")
        if inline_value is not None and file_value is not None:
            raise ValueError(f"Configure only one of {key} or {key}_file for {assignment_id}")
        if file_value is not None:
            optional_path = project_path(project_root, file_value)
            if optional_path is None:
                raise ValueError(f"Invalid {key}_file for {assignment_id}")
            result[key] = optional_path.read_text(encoding="utf-8")
        else:
            result[key] = str(inline_value or "")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    settings_path = project_path(project_root, args.settings)
    if settings_path is None:
        raise ValueError("A settings file is required")
    raw_settings = load_json(settings_path)
    pilot = raw_settings.get("pilot", {})
    if pilot.get("generation_enabled") is not False:
        raise RuntimeError('Pilot safety check failed: "pilot.generation_enabled" must be false')

    settings = load_settings(project_root, settings_path)
    mode = args.mode or pilot.get("default_mode", "both")
    if mode not in {"provider-count", "offline-estimate", "both"}:
        raise ValueError(f"Unsupported pilot mode: {mode}")
    load_dotenv(project_root / ".env")

    models = load_models(settings.models_file)
    if args.model not in models:
        raise KeyError(f"Unknown model key: {args.model}")
    model = models[args.model]
    pricing_path = project_path(project_root, pilot.get("pricing_file", "config/pricing.json"))
    if pricing_path is None:
        raise ValueError("pilot.pricing_file is required")
    pricing_config = load_json(pricing_path)
    if args.model not in pricing_config.get("models", {}):
        raise KeyError(f"No pricing entry for model: {args.model}")
    model_pricing = pricing_config["models"][args.model]
    counter_config = model_pricing.get("token_counter", {})
    multiplier = float(counter_config.get("conservative_multiplier", 1.0))
    if multiplier < 1:
        raise ValueError("conservative_multiplier must be at least 1.0")

    cases = load_cases(settings.cases_file)
    if args.case_ids:
        selected = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        cases = [case for case in cases if case["case_id"] in selected]
    required_test_id = pilot.get("required_test_id")
    if required_test_id:
        wrong = [case["case_id"] for case in cases if case["test_id"] != required_test_id]
        if wrong:
            raise ValueError(f"T0 pilot received non-{required_test_id} cases: {wrong[:5]}")
    if not cases:
        raise RuntimeError("No pilot cases selected")

    prompt_variants = load_prompt_variants(settings.prompt_variants_file, project_root)
    assignments = load_json(settings.assignments_file)
    schema = load_json(settings.output_schema_file) if settings.output_schema_file else {}
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    codebooks: dict[str, dict[str, Any]] = {}
    systems: dict[tuple[str, str], dict[str, Any]] = {}
    rendered: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for case in cases:
        assignment_id = case["assignment_id"]
        variant_id = case["prompt_variant_id"]
        if variant_id not in prompt_variants:
            raise KeyError(f"Unknown prompt_variant_id={variant_id}")
        if assignment_id not in codebooks:
            codebooks[assignment_id] = resolve_assignment(project_root, assignments, assignment_id)
        cache_key = (assignment_id, variant_id)
        if cache_key not in systems:
            cb = codebooks[assignment_id]
            systems[cache_key] = render_system_prompt(
                prompt_variants[variant_id],
                assignment_id,
                cb["frozen_tasks_text"],
                cb["task_kc_mapping_text"],
                cb["frozen_codebook_text"],
                cb["assignment_reference"],
                cb["human_validated_few_shot_examples"],
            )
        user = render_user_prompt(prompt_variants[variant_id], case, render_conversation(case), schema_text)
        rendered.append((case, systems[cache_key], user))

    run_id = args.run_id or datetime.now().strftime("pilot_%Y%m%d_%H%M%S")
    run_dir = settings.results_root / run_id
    if run_dir.exists():
        raise RuntimeError(f"Pilot output already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    chars_per_token = float(pilot.get("offline_chars_per_token", 4.0))
    use_offline = mode in {"offline-estimate", "both"}
    use_provider = mode in {"provider-count", "both"}
    reference_model = str(counter_config.get("reference_model", "gpt-5.6-sol"))
    api_key_env = str(pilot.get("openai_counter_api_key_env", "OPENAI_API_KEY"))
    stable_provider_counts: dict[str, int] = {}
    stable_offline_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    provider_errors: list[str] = []

    snapshots = run_dir / "prompt_snapshots"
    snapshots.mkdir()
    for case, system, user in rendered:
        cache_group = sha256_text(system["text"] + "\n" + user["stable_prefix_text"])
        offline_full = offline_stable = None
        if use_offline:
            offline_full = offline_request_tokens(system["text"], user["text"], chars_per_token)
            if cache_group not in stable_offline_counts:
                stable_offline_counts[cache_group] = offline_request_tokens(
                    system["text"], user["stable_prefix_text"], chars_per_token
                )
            offline_stable = stable_offline_counts[cache_group]

        provider_full = provider_stable = None
        provider_error = None
        if use_provider:
            try:
                if cache_group not in stable_provider_counts:
                    stable_provider_counts[cache_group] = openai_request_tokens(
                        reference_model=reference_model,
                        system_prompt=system["text"],
                        user_prompt=user["stable_prefix_text"],
                        api_key_env=api_key_env,
                    )
                provider_stable = stable_provider_counts[cache_group]
                provider_full = openai_request_tokens(
                    reference_model=reference_model,
                    system_prompt=system["text"],
                    user_prompt=user["text"],
                    api_key_env=api_key_env,
                )
            except Exception as exc:
                provider_error = f"{type(exc).__name__}: {exc}"
                provider_errors.append(f"{case['case_id']}: {provider_error}")

        chosen_source = "openai_provider_count" if provider_full is not None else "offline_estimate"
        chosen_full = provider_full if provider_full is not None else offline_full
        chosen_stable = provider_stable if provider_stable is not None else offline_stable
        if chosen_full is None or chosen_stable is None:
            raise RuntimeError(
                f"No token count available for {case['case_id']}. Provider error: {provider_error or 'none'}"
            )
        adjusted_full = math.ceil(chosen_full * multiplier)
        adjusted_stable = min(adjusted_full, math.ceil(chosen_stable * multiplier))
        row = {
            "case_id": case["case_id"],
            "base_conversation_id": case["base_conversation_id"],
            "conversation_id": case["conversation_id"],
            "chunk_id": case["chunk_id"],
            "chunk_index": case.get("chunk_index"),
            "chunk_count": case.get("chunk_count"),
            "assignment_id": case["assignment_id"],
            "test_id": case["test_id"],
            "model_key": model.key,
            "target_model_id": model.model_id,
            "count_source": chosen_source,
            "reference_tokenizer_model": reference_model if provider_full is not None else None,
            "conservative_multiplier": multiplier,
            "provider_full_input_tokens": provider_full,
            "provider_stable_input_tokens": provider_stable,
            "offline_full_input_tokens": offline_full,
            "offline_stable_input_tokens": offline_stable,
            "adjusted_full_input_tokens": adjusted_full,
            "adjusted_stable_input_tokens": adjusted_stable,
            "adjusted_dynamic_input_tokens": adjusted_full - adjusted_stable,
            "cache_group": cache_group,
            "provider_count_error": provider_error,
        }
        rows.append(row)
        safe_case = "".join(char if char.isalnum() or char in "-_." else "_" for char in case["case_id"])
        (snapshots / f"{safe_case}__system.txt").write_text(system["text"], encoding="utf-8")
        (snapshots / f"{safe_case}__user.txt").write_text(user["text"], encoding="utf-8")

    cache_config = pricing_config.get("cache_projection", {})
    summary = summarize_costs(
        rows,
        model_pricing,
        max_output_tokens=settings.max_output_tokens,
        repeats=settings.default_repeats,
        hit_rates=[float(value) for value in cache_config.get("assumed_hit_rates", [0, 0.5, 0.9, 1])],
        first_request_populates_cache=bool(cache_config.get("first_request_populates_cache", True)),
    )
    missing_prices = [
        key
        for key in ("input_per_million", "cached_input_per_million", "output_per_million")
        if model_pricing.get(key) is None
    ]
    report = {
        "pilot_version": "1.0",
        "created_at_utc": utc_now(),
        "generation_api_calls": 0,
        "token_count_api_calls_may_have_occurred": use_provider,
        "mode": mode,
        "model": {"key": model.key, "provider": model.provider, "model_id": model.model_id},
        "pricing_currency": pricing_config.get("currency", "USD"),
        "prices_as_of": pricing_config.get("prices_as_of"),
        "pricing": model_pricing,
        "output_schema_hash": sha256_json(schema),
        "tokenizer_warning": (
            "OpenAI provider counts are exact only for the configured OpenAI reference model. "
            "Counts for other providers are proxy estimates after the configured multiplier."
        ),
        "cache_warning": (
            "Token counting cannot observe real cache hits. Cache scenarios are projections; actual eligibility, "
            "retention, writes, and discounts are provider-specific. Cache storage charges are not included."
        ),
        "missing_price_fields": missing_prices,
        "provider_count_errors": provider_errors,
        "summary": summary,
    }
    write_jsonl(run_dir / "token_counts.jsonl", rows)
    write_csv(run_dir / "token_counts.csv", rows)
    (run_dir / "pilot_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(run_dir / "cache_scenarios.csv", summary["cache_scenarios"])
    write_jsonl(run_dir / "cases_snapshot.jsonl", cases)
    (run_dir / "pilot_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "settings_file": str(settings_path),
                "settings_hash": sha256_json(raw_settings),
                "pricing_file": str(pricing_path),
                "pricing_hash": sha256_json(pricing_config),
                "generation_enabled": False,
                "max_output_tokens": settings.max_output_tokens,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Pilot report written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
