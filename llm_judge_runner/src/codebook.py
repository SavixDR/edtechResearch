from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import load_json, sha256_json, sha256_text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _compact_list(value: Any, max_items: int = 10) -> str:
    if not value:
        return "-"
    if isinstance(value, str):
        return _text(value)
    if not isinstance(value, list):
        return _text(value)
    vals = [_text(v) for v in value if _text(v)]
    if len(vals) > max_items:
        vals = vals[:max_items] + [f"...({len(value) - max_items} more)"]
    return "; ".join(vals) if vals else "-"


def _compact_difficulty_signals(value: Any, max_items: int = 5) -> str:
    if not value:
        return "-"
    if not isinstance(value, list):
        return _text(value)
    rows: list[str] = []
    for sig in value[:max_items]:
        if isinstance(sig, dict):
            pieces = [
                _text(sig.get("signal_type")),
                _text(sig.get("description")),
                _text(sig.get("interpretation")),
            ]
            row = " | ".join(p for p in pieces if p)
            if "sufficient_alone" in sig:
                row += f" | sufficient_alone={sig.get('sufficient_alone')}"
            rows.append(row)
        else:
            rows.append(_text(sig))
    if len(value) > max_items:
        rows.append(f"...({len(value) - max_items} more)")
    return "; ".join(rows)


def build_prompt_sections(kc_model: dict[str, Any], configured_assignment_id: str | None = None) -> dict[str, Any]:
    assignment = kc_model.get("assignment", {})
    tasks = assignment.get("tasks", [])
    kcs = kc_model.get("knowledge_components", [])
    mapping = kc_model.get("task_kc_mapping", [])
    assignment_id = _text(assignment.get("assignment_id")) or _text(configured_assignment_id)
    prompt_tasks = [
        {key: task.get(key) for key in (
            "task_id", "description", "required_output", "assessed_behaviour"
        ) if key in task}
        for task in tasks
    ]
    kc_prompt_fields = (
        "kc_id", "canonical_name", "definition", "kc_type", "granularity", "requirement_level",
        "aliases", "student_must_know_or_do", "coding_use", "successful_performance_indicators",
        "possible_difficulty_signals", "parent_kc_id", "prerequisite_kc_ids",
        "commonly_confused_with_kc_ids", "alternative_kc_ids", "related_task_ids", "is_choice_based",
    )
    prompt_kcs = [
        {key: kc.get(key) for key in kc_prompt_fields if key in kc}
        for kc in kcs
    ]
    frozen_tasks = {
        "assignment_id": assignment_id,
        "assignment_title": assignment.get("assignment_title"),
        "objective": assignment.get("objective"),
        "tasks": prompt_tasks,
    }
    frozen_codebook = {
        "knowledge_components": prompt_kcs,
    }
    return {
        "frozen_tasks_text": json.dumps(frozen_tasks, indent=2, ensure_ascii=False),
        "task_kc_mapping_text": json.dumps(mapping, indent=2, ensure_ascii=False),
        "frozen_codebook_text": json.dumps(frozen_codebook, indent=2, ensure_ascii=False),
        "valid_kc_ids": [_text(kc.get("kc_id")) for kc in kcs if _text(kc.get("kc_id"))],
        "valid_task_ids": [_text(task.get("task_id")) for task in tasks if _text(task.get("task_id"))],
    }


def load_codebook(path: Path, configured_assignment_id: str | None = None) -> dict[str, Any]:
    raw = load_json(path)
    sections = build_prompt_sections(raw, configured_assignment_id)
    combined = "\n\n".join([
        sections["frozen_tasks_text"],
        sections["task_kc_mapping_text"],
        sections["frozen_codebook_text"],
    ])
    return {
        "raw": raw,
        "raw_hash": sha256_json(raw),
        "compact_text": combined,
        "compact_hash": sha256_text(combined),
        **sections,
        "source_path": str(path),
    }
