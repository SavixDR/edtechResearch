from __future__ import annotations

import json
import re
from typing import Any


def parse_output(text: str, mode: str = "strict") -> tuple[dict[str, Any] | None, str | None, str | None]:
    stripped = (text or "").strip()
    if not stripped:
        return None, None, "empty_output"

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, "direct_json", None
        return None, "direct_json", "top_level_json_is_not_object"
    except json.JSONDecodeError as exc:
        if mode == "strict":
            return None, None, f"json_parse_error: {exc}"

    # Optional tolerant mode. Parse method is recorded so later analysis can distinguish repaired outputs.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.I | re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj, "json_code_fence", None
        except json.JSONDecodeError:
            pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        try:
            obj = json.loads(stripped[first:last + 1])
            if isinstance(obj, dict):
                return obj, "extracted_json_object", None
        except json.JSONDecodeError as exc:
            return None, None, f"json_parse_error_after_tolerant_extraction: {exc}"

    return None, None, "no_parseable_json_object"


def validate_schema(parsed: dict[str, Any] | None, schema: dict[str, Any] | None) -> list[str]:
    if parsed is None or schema is None:
        return []
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema is required when output_schema_file is configured") from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def validate_context(
    parsed: dict[str, Any] | None,
    case: dict[str, Any],
    valid_task_ids: list[str],
    valid_kc_ids: list[str],
) -> list[str]:
    """Validate case-specific values that a static JSON Schema cannot enumerate."""
    if parsed is None:
        return []

    errors: list[str] = []
    for field in ("conversation_id", "chunk_id", "assignment_id"):
        if parsed.get(field) != case.get(field):
            errors.append(f"{field}: expected {case.get(field)!r}, got {parsed.get(field)!r}")

    allowed_tasks = set(valid_task_ids)
    allowed_kcs = set(valid_kc_ids)
    returned_task_ids: list[str] = []
    for index, task in enumerate(parsed.get("tasks", [])):
        if isinstance(task, dict):
            task_id = task.get("task_id")
            if isinstance(task_id, str):
                returned_task_ids.append(task_id)
            if task_id not in allowed_tasks:
                errors.append(f"tasks/{index}/task_id: unknown frozen task ID {task_id!r}")

    if len(returned_task_ids) != len(set(returned_task_ids)):
        errors.append("tasks: duplicate task_id values are not allowed")
    if returned_task_ids != sorted(returned_task_ids):
        errors.append("tasks: entries must be sorted by task_id")

    returned_kc_ids: list[str] = []
    for index, label in enumerate(parsed.get("labels", [])):
        if not isinstance(label, dict):
            continue
        kc_id = label.get("kc_id")
        if isinstance(kc_id, str):
            returned_kc_ids.append(kc_id)
        if kc_id not in allowed_kcs:
            errors.append(f"labels/{index}/kc_id: unknown frozen KC ID {kc_id!r}")
        for linked_index, task_id in enumerate(label.get("linked_task_ids", [])):
            if task_id not in allowed_tasks:
                errors.append(
                    f"labels/{index}/linked_task_ids/{linked_index}: unknown frozen task ID {task_id!r}"
                )
    if len(returned_kc_ids) != len(set(returned_kc_ids)):
        errors.append("labels: duplicate kc_id values are not allowed")
    if returned_kc_ids != sorted(returned_kc_ids):
        errors.append("labels: entries must be sorted by kc_id")
    return errors
