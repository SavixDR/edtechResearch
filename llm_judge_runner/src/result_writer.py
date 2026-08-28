from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .utils import append_jsonl, read_jsonl

FLAT_FIELDS = [
    "execution_key", "run_id", "case_id", "base_conversation_id", "conversation_id", "chunk_id",
    "assignment_id", "student_id", "test_id", "variant_id", "variant_type", "is_baseline",
    "prompt_variant_id", "model_key", "model_display_name", "provider", "model_id",
    "requested_version_label", "repeat_index", "status", "parse_method", "schema_error_count",
    "latency_ms", "collection_attempt", "api_attempts", "provider_response_id", "finish_reason", "prompt_hash",
    "conversation_hash", "case_hash", "judge_output_json", "raw_text", "validation_errors_json",
    "usage_json", "transformation_metadata_json", "api_error"
]


def append_result(
    results_jsonl: Path,
    flat_csv: Path,
    record: dict[str, Any],
    *,
    append_json_record: bool = True,
) -> None:
    if append_json_record:
        append_jsonl(results_jsonl, record)
    flat = {
        "execution_key": record["execution_key"],
        "run_id": record["run_id"],
        "case_id": record["case"]["case_id"],
        "base_conversation_id": record["case"]["base_conversation_id"],
        "conversation_id": record["case"]["conversation_id"],
        "chunk_id": record["case"]["chunk_id"],
        "assignment_id": record["case"]["assignment_id"],
        "student_id": record["case"].get("student_id"),
        "test_id": record["experiment"]["test_id"],
        "variant_id": record["experiment"]["variant_id"],
        "variant_type": record["experiment"]["variant_type"],
        "is_baseline": record["experiment"]["is_baseline"],
        "prompt_variant_id": record["experiment"]["prompt_variant_id"],
        "model_key": record["judge"]["model_key"],
        "model_display_name": record["judge"]["display_name"],
        "provider": record["judge"]["provider"],
        "model_id": record["judge"]["model_id"],
        "requested_version_label": record["judge"].get("requested_version_label"),
        "repeat_index": record["judge"]["repeat_index"],
        "status": record["response"]["status"],
        "parse_method": record["response"].get("parse_method"),
        "schema_error_count": len(record["response"].get("validation_errors", [])),
        "latency_ms": record["response"].get("latency_ms"),
        "collection_attempt": record["response"].get("collection_attempt", 1),
        "api_attempts": record["response"].get("api_attempts"),
        "provider_response_id": record["response"].get("provider_response_id"),
        "finish_reason": record["response"].get("finish_reason"),
        "prompt_hash": record["input_provenance"]["rendered_system_prompt_hash"],
        "conversation_hash": record["input_provenance"]["conversation_hash"],
        "case_hash": record["case"]["case_hash"],
        "judge_output_json": json.dumps(record["response"].get("parsed_json"), ensure_ascii=False),
        "raw_text": record["response"].get("raw_text", ""),
        "validation_errors_json": json.dumps(record["response"].get("validation_errors", []), ensure_ascii=False),
        "usage_json": json.dumps(record["response"].get("usage", {}), ensure_ascii=False),
        "transformation_metadata_json": json.dumps(record["experiment"].get("transformation_metadata", {}), ensure_ascii=False),
        "api_error": record["response"].get("api_error"),
    }
    new_file = not flat_csv.exists()
    with flat_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(flat)


def rebuild_flat_csv(results_jsonl: Path, flat_csv: Path) -> None:
    """Rebuild the derived CSV from the authoritative append-only JSONL log."""
    if flat_csv.exists():
        flat_csv.unlink()
    if not results_jsonl.exists():
        return
    for record in read_jsonl(results_jsonl):
        append_result(results_jsonl, flat_csv, record, append_json_record=False)
