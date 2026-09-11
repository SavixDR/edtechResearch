from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .cases import load_cases, render_conversation
from .codebook import load_codebook
from .config import ModelConfig, Settings, load_models
from .cost_estimation import calculate_live_cost, write_live_cost_reports
from .prompts import load_prompt_variants, render_system_prompt, render_user_prompt
from .providers.registry import get_provider
from .result_writer import append_result, rebuild_flat_csv
from .utils import load_json, read_jsonl, sha256_json, sha256_text, write_jsonl
from .validation import parse_output, validate_context, validate_schema


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_key(model_key: str, case_id: str, repeat_index: int) -> str:
    return sha256_text(f"{model_key}::{case_id}::{repeat_index}")[:24]


def safe_filename(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", value)


def format_completion_outcome(
    *,
    status: str,
    api_error: str | None,
    parse_error: str | None,
    validation_errors: list[str],
    finish_reason: str | None,
    usage: dict[str, Any],
    max_output_tokens: int,
) -> str:
    """Return a concise terminal explanation for a completed request."""
    if api_error:
        reason = f"API failure: {api_error}"
    elif parse_error:
        reason = f"output parsing failed: {parse_error}"
    elif validation_errors:
        reason = f"validation failed ({len(validation_errors)} error(s)): {validation_errors[0]}"
    elif finish_reason == "length":
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        reason = f"completed with output limit/truncation (finish_reason=length, output_tokens={output_tokens or 'unknown'}, max_output_tokens={max_output_tokens})"
    elif finish_reason:
        reason = f"completed (finish_reason={finish_reason})"
    else:
        reason = "completed"
    return f"{status}: {reason}"


def _resolve_codebook(project_root: Path, assignments: dict[str, Any], assignment_id: str) -> dict[str, Any]:
    if assignment_id not in assignments:
        raise KeyError(f"No assignment configuration for {assignment_id}")
    assignment_config = assignments[assignment_id]
    p = Path(assignment_config["codebook_file"])
    if not p.is_absolute():
        p = project_root / p
    if not p.exists():
        raise FileNotFoundError(f"Codebook not found for {assignment_id}: {p}")
    codebook = load_codebook(p, assignment_id)
    for key in ("assignment_reference", "human_validated_few_shot_examples"):
        inline_value = assignment_config.get(key)
        file_value = assignment_config.get(f"{key}_file")
        if inline_value is not None and file_value is not None:
            raise ValueError(f"Configure only one of {key} or {key}_file for {assignment_id}")
        if file_value is not None:
            optional_path = Path(file_value)
            if not optional_path.is_absolute():
                optional_path = project_root / optional_path
            codebook[key] = optional_path.read_text(encoding="utf-8")
            codebook[f"{key}_source"] = str(optional_path)
        else:
            codebook[key] = str(inline_value or "")
            codebook[f"{key}_source"] = None
    return codebook


def _prepare_run_dir(
    settings: Settings, dry_run: bool, requested_run_id: str | None = None
) -> tuple[str, Path]:
    run_id = requested_run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    category = "dryruns" if dry_run else "live_runs"
    run_dir = settings.results_root / category / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _existing_execution_keys(results_path: Path, retry_api_errors: bool = False) -> set[str]:
    if not results_path.exists():
        return set()
    statuses_by_key: dict[str, list[str]] = {}
    for row in read_jsonl(results_path):
        key = row.get("execution_key")
        if key:
            statuses_by_key.setdefault(str(key), []).append(str(row.get("response", {}).get("status", "")))
    return {
        key for key, statuses in statuses_by_key.items()
        if not retry_api_errors or any(status != "api_error" for status in statuses)
    }


def _execution_attempt_counts(results_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not results_path.exists():
        return counts
    for row in read_jsonl(results_path):
        key = row.get("execution_key")
        if key:
            counts[str(key)] = counts.get(str(key), 0) + 1
    return counts


def run_collection(
    project_root: Path,
    settings: Settings,
    selected_model_keys: list[str] | None = None,
    selected_tests: list[str] | None = None,
    selected_case_ids: list[str] | None = None,
    repeats: int | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    resume: bool = False,
    retry_api_errors: bool = False,
) -> Path:
    if retry_api_errors and not resume:
        raise ValueError("--retry-api-errors requires --resume")
    load_dotenv(project_root / ".env")
    models = load_models(settings.models_file)
    prompt_variants = load_prompt_variants(settings.prompt_variants_file, project_root)
    assignments = load_json(settings.assignments_file)
    cases = load_cases(settings.cases_file)
    output_schema = load_json(settings.output_schema_file) if settings.output_schema_file else None
    output_schema_text = json.dumps(output_schema, indent=2, ensure_ascii=False) if output_schema else None
    pricing_config = load_json(settings.pricing_file)
    pricing_hash = sha256_json(pricing_config)

    if selected_model_keys:
        missing = sorted(set(selected_model_keys) - set(models))
        if missing:
            raise KeyError(f"Unknown model keys: {missing}")
        chosen_models = [models[k] for k in selected_model_keys]
    else:
        chosen_models = [m for m in models.values() if m.enabled]

    if selected_tests:
        allowed = set(selected_tests)
        cases = [c for c in cases if c["test_id"] in allowed]
    if selected_case_ids:
        allowed_ids = set(selected_case_ids)
        cases = [c for c in cases if c["case_id"] in allowed_ids]
    if not cases:
        raise RuntimeError("No cases selected")
    if not chosen_models:
        raise RuntimeError("No models selected")
    missing_pricing = sorted(
        model.key for model in chosen_models if model.key not in pricing_config.get("models", {})
    )
    if missing_pricing:
        raise KeyError(f"No pricing entries for selected models: {missing_pricing}")

    repeats = repeats if repeats is not None else settings.default_repeats
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    run_id, run_dir = _prepare_run_dir(settings, dry_run, run_id)
    results_jsonl = run_dir / "results.jsonl"
    results_csv = run_dir / "results_flat.csv"
    errors_jsonl = run_dir / "errors.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if resume and not manifest_path.exists():
        raise RuntimeError(f"Cannot resume: run manifest does not exist at {manifest_path}")
    if not resume and manifest_path.exists():
        raise RuntimeError(f"Run {run_id} already exists. Use --resume or choose a new --run-id.")

    # Resolve all referenced prompts/codebooks before any API call.
    prompt_cache: dict[tuple[str, str], dict[str, Any]] = {}
    user_prompt_cache: dict[str, dict[str, Any]] = {}
    codebook_cache: dict[str, dict[str, Any]] = {}
    for case in cases:
        assignment_id = case["assignment_id"]
        if assignment_id not in codebook_cache:
            codebook_cache[assignment_id] = _resolve_codebook(project_root, assignments, assignment_id)
        variant_id = case["prompt_variant_id"]
        if variant_id not in prompt_variants:
            raise KeyError(f"Case {case['case_id']} references unknown prompt_variant_id={variant_id}")
        cache_key = (assignment_id, variant_id)
        if cache_key not in prompt_cache:
            prompt_cache[cache_key] = render_system_prompt(
                prompt_variants[variant_id],
                assignment_id,
                codebook_cache[assignment_id]["frozen_tasks_text"],
                codebook_cache[assignment_id]["task_kc_mapping_text"],
                codebook_cache[assignment_id]["frozen_codebook_text"],
                codebook_cache[assignment_id]["assignment_reference"],
                codebook_cache[assignment_id]["human_validated_few_shot_examples"],
            )
        conversation_text = render_conversation(case)
        user_prompt_cache[str(case["case_id"])] = render_user_prompt(
            prompt_variants[variant_id], case, conversation_text, output_schema_text or "{}"
        )

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "experiment_name": settings.experiment_name,
        "suite_version": settings.suite_version,
        "prompt_version": settings.prompt_version,
        "parse_mode": settings.parse_mode,
        "output_schema_file": str(settings.output_schema_file) if settings.output_schema_file else None,
        "output_schema_hash": sha256_json(output_schema) if output_schema else None,
        "pricing": {
            "source_file": str(settings.pricing_file),
            "hash": pricing_hash,
            "currency": pricing_config.get("currency", "USD"),
            "prices_as_of": pricing_config.get("prices_as_of"),
            "selected_model_rates": {
                model.key: pricing_config["models"][model.key] for model in chosen_models
            },
        },
        "models": [m.__dict__ for m in chosen_models],
        "selected_tests": sorted({c["test_id"] for c in cases}),
        "selected_case_count": len(cases),
        "repeats": repeats,
        "planned_requests": len(cases) * len(chosen_models) * repeats,
        "dry_run": dry_run,
        "python": sys.version,
        "platform": platform.platform(),
        "prompt_snapshots": {
            f"{a}::{p}": {
                "source_file": obj["source_file"],
                "template_hash": obj["template_hash"],
                "rendered_hash": obj["rendered_hash"],
                "user_source_file": str(prompt_variants[p]["resolved_user_file"]),
                "user_template_hash": sha256_text(
                    prompt_variants[p]["resolved_user_file"].read_text(encoding="utf-8")
                ),
            }
            for (a, p), obj in prompt_cache.items()
        },
        "rendered_user_prompts": {
            case_id: {
                "source_file": obj["source_file"],
                "template_hash": obj["template_hash"],
                "rendered_hash": obj["rendered_hash"],
            }
            for case_id, obj in user_prompt_cache.items()
        },
        "codebooks": {
            a: {
                "source_path": cb["source_path"],
                "raw_hash": cb["raw_hash"],
                "compact_hash": cb["compact_hash"],
                "task_count": len(cb["valid_task_ids"]),
                "kc_count": len(cb["valid_kc_ids"]),
            }
            for a, cb in codebook_cache.items()
        },
    }
    resume_basis = {
        key: manifest[key] for key in (
            "experiment_name", "suite_version", "prompt_version", "parse_mode", "output_schema_hash",
            "models", "selected_tests", "selected_case_count", "repeats", "planned_requests", "dry_run",
            "prompt_snapshots", "rendered_user_prompts", "codebooks", "pricing",
        )
    }
    resume_basis["cases"] = [
        {"case_id": case["case_id"], "case_hash": case["case_hash"]} for case in cases
    ]
    resume_basis["max_output_tokens"] = settings.max_output_tokens
    manifest["resume_signature"] = sha256_json(resume_basis)

    if resume:
        existing_manifest = load_json(manifest_path)
        existing_signature = existing_manifest.get("resume_signature")
        if existing_signature is None:
            raise RuntimeError(
                "This run predates resume-signature validation and cannot be resumed safely. Start a new run ID."
            )
        if existing_signature != manifest["resume_signature"]:
            raise RuntimeError(
                "Resume configuration does not match the original run. Use the original models, cases, repeats, "
                "prompts, schema, and codebooks, or start a new run ID."
            )
        manifest = existing_manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if settings.save_case_snapshot:
        write_jsonl(run_dir / "cases_snapshot.jsonl", cases)

    if settings.save_prompt_snapshots:
        snap_dir = run_dir / "prompt_snapshots"
        snap_dir.mkdir(exist_ok=True)
        for (assignment_id, variant_id), prompt in prompt_cache.items():
            safe = safe_filename(f"{assignment_id}__{variant_id}__system.txt")
            (snap_dir / safe).write_text(prompt["text"], encoding="utf-8")
            user_template = prompt_variants[variant_id]["resolved_user_file"]
            user_safe = safe_filename(f"{assignment_id}__{variant_id}__user_template.txt")
            (snap_dir / user_safe).write_text(user_template.read_text(encoding="utf-8"), encoding="utf-8")
        for case in cases:
            rendered_user = user_prompt_cache[str(case["case_id"])]
            user_safe = safe_filename(f"{case['case_id']}__user.txt")
            (snap_dir / user_safe).write_text(rendered_user["text"], encoding="utf-8")

    if dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return run_dir

    if resume:
        rebuild_flat_csv(results_jsonl, results_csv)
        existing_records = read_jsonl(results_jsonl) if results_jsonl.exists() else []
    else:
        existing_records = []
    write_live_cost_reports(run_dir, existing_records, pricing_config, pricing_hash)
    done = _existing_execution_keys(results_jsonl, retry_api_errors=retry_api_errors) if resume else set()
    collection_attempt_counts = _execution_attempt_counts(results_jsonl) if resume else {}
    total = len(cases) * len(chosen_models) * repeats
    current = 0

    for model in chosen_models:
        provider = None
        provider_init_error = None
        try:
            provider = get_provider(model.adapter)
        except Exception as exc:
            provider_init_error = f"{type(exc).__name__}: {exc}"
            print(f"MODEL INIT ERROR {model.key} | {provider_init_error}")
        for case in cases:
            conversation_text = render_conversation(case)
            conversation_hash = sha256_text(conversation_text)
            prompt = prompt_cache[(case["assignment_id"], case["prompt_variant_id"])]
            user_prompt = user_prompt_cache[str(case["case_id"])]
            user_prompt_hash = user_prompt["rendered_hash"]

            for repeat_index in range(1, repeats + 1):
                current += 1
                ek = execution_key(model.key, case["case_id"], repeat_index)
                if ek in done:
                    print(f"[{current}/{total}] SKIP {model.key} | {case['case_id']} | repeat={repeat_index}")
                    continue
                print(f"[{current}/{total}] RUN  {model.key} | {case['case_id']} | repeat={repeat_index}")

                started = time.perf_counter()
                attempts = 0
                provider_result = None
                api_error = provider_init_error
                while provider is not None and attempts < settings.max_api_attempts:
                    attempts += 1
                    try:
                        provider_result = provider.generate(
                            model=model,
                            system_prompt=prompt["text"],
                            user_prompt=user_prompt["text"],
                            max_output_tokens=settings.max_output_tokens,
                            timeout_seconds=settings.request_timeout_seconds,
                        )
                        api_error = None
                        break
                    except Exception as exc:
                        api_error = f"{type(exc).__name__}: {exc}"
                        if attempts < settings.max_api_attempts:
                            time.sleep(settings.retry_backoff_seconds * attempts)

                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                parsed = None
                parse_method = None
                parse_error = None
                schema_errors: list[str] = []

                if provider_result is not None:
                    parsed, parse_method, parse_error = parse_output(provider_result.raw_text, settings.parse_mode)
                    schema_errors = validate_schema(parsed, output_schema)
                    schema_errors.extend(validate_context(
                        parsed,
                        case,
                        codebook_cache[case["assignment_id"]]["valid_task_ids"],
                        codebook_cache[case["assignment_id"]]["valid_kc_ids"],
                    ))

                if api_error is not None:
                    status = "api_error"
                elif parse_error is not None:
                    status = "parse_error"
                elif schema_errors:
                    status = "schema_error"
                else:
                    status = "ok"

                record = {
                    "record_version": "1.0",
                    "execution_key": ek,
                    "run_id": run_id,
                    "collected_at_utc": utc_now(),
                    "experiment": {
                        "suite_version": settings.suite_version,
                        "test_id": case["test_id"],
                        "variant_id": case["variant_id"],
                        "variant_type": case["variant_type"],
                        "is_baseline": bool(case["is_baseline"]),
                        "prompt_variant_id": case["prompt_variant_id"],
                        "transformation_metadata": case.get("transformation_metadata", {}),
                    },
                    "case": {
                        "case_id": case["case_id"],
                        "base_conversation_id": case["base_conversation_id"],
                        "conversation_id": case["conversation_id"],
                        "chunk_id": case["chunk_id"],
                        "chunk_index": case.get("chunk_index"),
                        "chunk_count": case.get("chunk_count"),
                        "assignment_id": case["assignment_id"],
                        "student_id": case.get("student_id"),
                        "case_hash": case["case_hash"],
                    },
                    "judge": {
                        "model_key": model.key,
                        "display_name": model.display_name,
                        "provider": model.provider,
                        "country": model.country,
                        "adapter": model.adapter,
                        "model_id": model.model_id,
                        "requested_version_label": model.requested_version_label,
                        "reasoning_effort": model.reasoning_effort,
                        "temperature": model.temperature,
                        "repeat_index": repeat_index,
                    },
                    "input_provenance": {
                        "prompt_version": settings.prompt_version,
                        "prompt_source_file": prompt["source_file"],
                        "prompt_template_hash": prompt["template_hash"],
                        "rendered_system_prompt_hash": prompt["rendered_hash"],
                        "user_prompt_hash": user_prompt_hash,
                        "user_prompt_source_file": user_prompt["source_file"],
                        "user_prompt_template_hash": user_prompt["template_hash"],
                        "conversation_hash": conversation_hash,
                        "codebook_source": codebook_cache[case["assignment_id"]]["source_path"],
                        "codebook_hash": codebook_cache[case["assignment_id"]]["compact_hash"],
                        "output_schema_hash": manifest["output_schema_hash"],
                    },
                    "response": {
                        "status": status,
                        "raw_text": provider_result.raw_text if provider_result else "",
                        "parsed_json": parsed,
                        "parse_method": parse_method,
                        "parse_error": parse_error,
                        "validation_errors": schema_errors,
                        "provider_response_id": provider_result.provider_response_id if provider_result else None,
                        "provider_model": provider_result.provider_model if provider_result else None,
                        "finish_reason": provider_result.finish_reason if provider_result else None,
                        "usage": provider_result.usage if provider_result else {},
                        "provider_metadata": provider_result.provider_metadata if provider_result else {},
                        "latency_ms": latency_ms,
                        "collection_attempt": collection_attempt_counts.get(ek, 0) + 1,
                        "api_attempts": attempts,
                        "api_error": api_error,
                    },
                }
                record["cost"] = calculate_live_cost(
                    adapter=model.adapter,
                    usage=provider_result.usage if provider_result else {},
                    pricing=pricing_config["models"][model.key],
                    api_attempts=attempts,
                    provider_response_received=provider_result is not None,
                )
                append_result(results_jsonl, results_csv, record)
                existing_records.append(record)
                write_live_cost_reports(run_dir, existing_records, pricing_config, pricing_hash)
                collection_attempt_counts[ek] = collection_attempt_counts.get(ek, 0) + 1
                if status != "ok":
                    from .utils import append_jsonl
                    append_jsonl(errors_jsonl, record)
                print(
                    f"[{current}/{total}] DONE {model.key} | {case['case_id']} | repeat={repeat_index} | "
                    f"{format_completion_outcome(
                        status=status,
                        api_error=api_error,
                        parse_error=parse_error,
                        validation_errors=schema_errors,
                        finish_reason=provider_result.finish_reason if provider_result else None,
                        usage=provider_result.usage if provider_result else {},
                        max_output_tokens=settings.max_output_tokens,
                    )} | latency_ms={latency_ms}"
                )

    return run_dir
