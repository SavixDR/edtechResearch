"""
Validate a frozen KC codebook before using it in labelling, retrieval, or training.

Colab example:
    !python kc_representation_experiment/scripts/phase1_validate_frozen_codebook.py \
      --codebook "/content/kc_experiment/phase1/frozen_kc_codebook.json"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "kc_id",
    "canonical_name",
    "definition",
    "aliases",
    "parent_kc_id",
}

NON_NEUTRAL_FIELDS = {
    "related_task_ids",
    "prerequisite_kc_ids",
    "source_evidence",
    "student_must_know_or_do",
    "successful_performance_indicators",
    "possible_difficulty_signals",
    "commonly_confused_with_kc_ids",
    "requirement_level",
    "confidence",
}


def load_codebook(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise TypeError(f"Expected codebook JSON list, got {type(data).__name__}")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Codebook row {index} is {type(item).__name__}, not object")
        rows.append(item)
    return rows


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    kc_ids = [row.get("kc_id") for row in rows]
    id_counts = Counter(kc_ids)
    valid_ids = set(kc_ids)

    for index, row in enumerate(rows):
        label = row.get("kc_id") or f"row_{index}"
        missing = sorted(REQUIRED_FIELDS - set(row.keys()))
        for field in missing:
            errors.append(f"{label}: missing required field {field}")

        if not row.get("kc_id"):
            errors.append(f"row {index}: blank kc_id")
        elif id_counts[row["kc_id"]] > 1:
            errors.append(f"{label}: duplicate kc_id")

        if not row.get("canonical_name"):
            errors.append(f"{label}: blank canonical_name")

        if not row.get("definition"):
            errors.append(f"{label}: blank definition")

        if "aliases" in row and not isinstance(row["aliases"], list):
            errors.append(f"{label}: aliases must be a list")

        parent = row.get("parent_kc_id")
        if parent not in (None, "") and parent not in valid_ids:
            errors.append(f"{label}: parent_kc_id {parent} not found")

        for field in sorted(NON_NEUTRAL_FIELDS & set(row.keys())):
            errors.append(f"{label}: non-neutral field present: {field}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", required=True, help="Path to frozen_kc_codebook.json")
    args = parser.parse_args()

    codebook_path = Path(args.codebook)
    rows = load_codebook(codebook_path)
    errors = validate(rows)

    print(f"Codebook: {codebook_path}")
    print(f"Rows: {len(rows)}")
    print(f"Valid: {not errors}")
    print(f"Error count: {len(errors)}")

    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
