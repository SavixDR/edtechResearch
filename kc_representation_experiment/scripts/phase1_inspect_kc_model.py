r"""
Phase 1: inspect the Assignment KC JSON model and create a frozen KC codebook.

This script is intentionally schema-checking first. It reads the actual uploaded
JSON, summarizes its structure, identifies task/KC/relationship candidates, and
writes a representation-neutral KC codebook for later JSON-vs-KG experiments.

Colab example:
    !python kc_representation_experiment/scripts/phase1_inspect_kc_model.py \
      --input "/content/f24_a2_kc_model.json" \
      --output-dir "/content/kc_experiment/phase1"

Local example:
    python kc_representation_experiment/scripts/phase1_inspect_kc_model.py `
      --input "E:\PostGraduate\Research\edtech\KC Generation\f24_a2\f24_a2_kc_model.json" `
      --output-dir "kc_representation_experiment/outputs/phase1"
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "assignment",
    "knowledge_components",
    "kc_relationships",
    "task_kc_mapping",
}

MINIMUM_CODEBOOK_FIELDS = [
    "kc_id",
    "canonical_name",
    "definition",
    "aliases",
    "parent_kc_id",
]

CODEBOOK_EXCLUDED_FIELDS = [
    "related_task_ids",
    "prerequisite_kc_ids",
    "source_evidence",
    "student_must_know_or_do",
    "successful_performance_indicators",
    "possible_difficulty_signals",
    "commonly_confused_with_kc_ids",
    "requirement_level",
    "confidence",
]


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object at top level, got {type(data).__name__}")
    return data


def type_name(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "list[]"
        inner = sorted({type_name(item) for item in value[:25]})
        return "list[" + "|".join(inner) + "]"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def field_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    keys = sorted(set().union(*(row.keys() for row in rows)))
    profile = []
    for key in keys:
        values = [row.get(key) for row in rows]
        profile.append(
            {
                "field": key,
                "types": sorted({type_name(value) for value in values}),
                "missing_or_blank": sum(
                    1 for row in rows if key not in row or row.get(key) in (None, "")
                ),
                "total_rows": len(rows),
            }
        )
    return profile


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_diagnostic_signal(kc: dict[str, Any]) -> dict[str, Any] | None:
    signals = kc.get("possible_difficulty_signals")
    if not isinstance(signals, list) or not signals:
        return None
    first = signals[0]
    if not isinstance(first, dict):
        return None
    return {
        "signal_type": first.get("signal_type"),
        "description": first.get("description"),
    }


def make_codebook(
    knowledge_components: list[dict[str, Any]],
    include_diagnostic_example: bool,
) -> list[dict[str, Any]]:
    codebook = []
    for kc in knowledge_components:
        row = {
            "kc_id": kc.get("kc_id"),
            "canonical_name": kc.get("canonical_name"),
            "definition": kc.get("definition"),
            "aliases": listify(kc.get("aliases")),
            "parent_kc_id": kc.get("parent_kc_id"),
        }
        if include_diagnostic_example:
            row["diagnostic_example"] = first_diagnostic_signal(kc)
        codebook.append(row)
    return codebook


def validate_codebook(codebook: list[dict[str, Any]]) -> list[str]:
    errors = []
    seen_ids = set()
    valid_ids = {row.get("kc_id") for row in codebook}

    for index, row in enumerate(codebook):
        for field in MINIMUM_CODEBOOK_FIELDS:
            if field not in row:
                errors.append(f"row {index}: missing field {field}")

        kc_id = row.get("kc_id")
        if not kc_id:
            errors.append(f"row {index}: blank kc_id")
        elif kc_id in seen_ids:
            errors.append(f"row {index}: duplicate kc_id {kc_id}")
        seen_ids.add(kc_id)

        if not row.get("canonical_name"):
            errors.append(f"{kc_id}: blank canonical_name")
        if not row.get("definition"):
            errors.append(f"{kc_id}: blank definition")

        aliases = row.get("aliases")
        if not isinstance(aliases, list):
            errors.append(f"{kc_id}: aliases must be a list")

        parent = row.get("parent_kc_id")
        if parent not in (None, "") and parent not in valid_ids:
            errors.append(f"{kc_id}: parent_kc_id {parent} not found in codebook")

        for excluded in CODEBOOK_EXCLUDED_FIELDS:
            if excluded in row:
                errors.append(f"{kc_id}: non-neutral field present in codebook: {excluded}")

    return errors


def assignment_tasks(data: dict[str, Any]) -> list[dict[str, Any]]:
    assignment = data.get("assignment", {})
    if not isinstance(assignment, dict):
        return []
    tasks = assignment.get("tasks", [])
    return [task for task in tasks if isinstance(task, dict)]


def collect_referenced_ids(data: dict[str, Any]) -> dict[str, Any]:
    kcs = data.get("knowledge_components", [])
    relationships = data.get("kc_relationships", [])
    mappings = data.get("task_kc_mapping", [])
    tasks = assignment_tasks(data)

    kc_ids = {kc.get("kc_id") for kc in kcs if isinstance(kc, dict)}
    task_ids = {task.get("task_id") for task in tasks if isinstance(task, dict)}

    kc_refs = Counter()
    task_refs = Counter()

    for kc in kcs:
        if not isinstance(kc, dict):
            continue
        for field in ("parent_kc_id",):
            value = kc.get(field)
            if value:
                kc_refs[value] += 1
        for field in ("prerequisite_kc_ids", "commonly_confused_with_kc_ids"):
            for value in listify(kc.get(field)):
                if value:
                    kc_refs[value] += 1
        for value in listify(kc.get("related_task_ids")):
            if value:
                task_refs[value] += 1

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        for field in ("source_kc_id", "target_kc_id"):
            value = rel.get(field)
            if value:
                kc_refs[value] += 1

    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        if mapping.get("task_id"):
            task_refs[mapping["task_id"]] += 1
        for field in (
            "core_kc_ids",
            "supporting_kc_ids",
            "prerequisite_kc_ids",
            "optional_kc_ids",
            "assessed_kc_ids",
        ):
            for value in listify(mapping.get(field)):
                if value:
                    kc_refs[value] += 1

    return {
        "undefined_kc_references": sorted(kc for kc in kc_refs if kc not in kc_ids),
        "undefined_task_references": sorted(task for task in task_refs if task not in task_ids),
        "unreferenced_kc_ids": sorted(kc for kc in kc_ids if kc not in kc_refs),
        "unreferenced_task_ids": sorted(task for task in task_ids if task not in task_refs),
    }


def summarize(data: dict[str, Any], input_path: Path) -> dict[str, Any]:
    kcs = data.get("knowledge_components", [])
    relationships = data.get("kc_relationships", [])
    mappings = data.get("task_kc_mapping", [])
    tasks = assignment_tasks(data)

    if not isinstance(kcs, list):
        raise TypeError("knowledge_components must be a list")
    if not isinstance(relationships, list):
        raise TypeError("kc_relationships must be a list")
    if not isinstance(mappings, list):
        raise TypeError("task_kc_mapping must be a list")

    kc_rows = [row for row in kcs if isinstance(row, dict)]
    relationship_rows = [row for row in relationships if isinstance(row, dict)]
    mapping_rows = [row for row in mappings if isinstance(row, dict)]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "top_level_keys": list(data.keys()),
        "missing_expected_top_level_keys": sorted(REQUIRED_TOP_LEVEL_KEYS - set(data.keys())),
        "assignment_id": data.get("assignment", {}).get("assignment_id")
        if isinstance(data.get("assignment"), dict)
        else None,
        "assignment_title": data.get("assignment", {}).get("assignment_title")
        if isinstance(data.get("assignment"), dict)
        else None,
        "counts": {
            "tasks": len(tasks),
            "knowledge_components": len(kc_rows),
            "kc_relationships": len(relationship_rows),
            "task_kc_mapping_rows": len(mapping_rows),
        },
        "candidate_kc_nodes": {
            "source_section": "knowledge_components",
            "id_field": "kc_id",
            "name_field": "canonical_name",
            "count": len(kc_rows),
            "kc_type_counts": dict(Counter(row.get("kc_type") for row in kc_rows)),
            "granularity_counts": dict(Counter(row.get("granularity") for row in kc_rows)),
            "requirement_level_counts": dict(
                Counter(row.get("requirement_level") for row in kc_rows)
            ),
        },
        "candidate_task_nodes": {
            "source_section": "assignment.tasks",
            "id_field": "task_id",
            "count": len(tasks),
        },
        "possible_relationship_types": {
            "kg_edge_sources_assumed_by_user": [
                "task_kc_mapping",
                "kc_relationships",
            ],
            "kc_relationship_type_counts": dict(
                Counter(row.get("relationship") for row in relationship_rows)
            ),
            "task_kc_mapping_edge_fields": [
                "core_kc_ids",
                "supporting_kc_ids",
                "prerequisite_kc_ids",
                "optional_kc_ids",
                "assessed_kc_ids",
            ],
        },
        "schema_profiles": {
            "knowledge_components": field_profile(kc_rows),
            "kc_relationships": field_profile(relationship_rows),
            "task_kc_mapping": field_profile(mapping_rows),
            "assignment_tasks": field_profile(tasks),
        },
        "reference_integrity": collect_referenced_ids(data),
        "frozen_codebook_policy": {
            "included_fields": MINIMUM_CODEBOOK_FIELDS,
            "excluded_fields_to_keep_representation_neutral": CODEBOOK_EXCLUDED_FIELDS,
            "reason": (
                "The codebook is the fixed label vocabulary. Retrieval-only content such as "
                "task mappings, prerequisites, source evidence, performance indicators, "
                "confusion links, and difficulty signals should not be included by default "
                "because those fields are part of the KB content used by JSON/KG retrievers."
            ),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_codebook_csv(path: Path, codebook: list[dict[str, Any]]) -> None:
    fieldnames = list(codebook[0].keys()) if codebook else MINIMUM_CODEBOOK_FIELDS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in codebook:
            csv_row = dict(row)
            csv_row["aliases"] = json.dumps(row.get("aliases", []), ensure_ascii=False)
            if "diagnostic_example" in csv_row:
                csv_row["diagnostic_example"] = json.dumps(
                    csv_row["diagnostic_example"], ensure_ascii=False
                )
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to f24_a2_kc_model.json")
    parser.add_argument("--output-dir", required=True, help="Directory for Phase 1 outputs")
    parser.add_argument(
        "--include-diagnostic-example",
        action="store_true",
        help=(
            "Include one concise difficulty signal per KC. Leave off for the strictest "
            "representation-neutral codebook."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)
    summary = summarize(data, input_path)
    codebook = make_codebook(
        data["knowledge_components"],
        include_diagnostic_example=args.include_diagnostic_example,
    )
    validation_errors = validate_codebook(codebook)

    write_json(output_dir / "phase1_schema_report.json", summary)
    write_json(output_dir / "frozen_kc_codebook.json", codebook)
    write_codebook_csv(output_dir / "frozen_kc_codebook.csv", codebook)
    write_json(
        output_dir / "frozen_kc_codebook_validation.json",
        {
            "valid": not validation_errors,
            "error_count": len(validation_errors),
            "errors": validation_errors,
        },
    )

    print("Phase 1 inspection complete.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"KCs: {summary['counts']['knowledge_components']}")
    print(f"Tasks: {summary['counts']['tasks']}")
    print(f"KC relationships: {summary['counts']['kc_relationships']}")
    print(f"Task-KC mapping rows: {summary['counts']['task_kc_mapping_rows']}")
    print(f"Codebook valid: {not validation_errors}")
    if validation_errors:
        print("First validation errors:")
        for error in validation_errors[:10]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
