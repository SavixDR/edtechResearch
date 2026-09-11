"""List available case IDs and calculate the IDs needed for a run.

Examples:
    python list_case_ids.py \
        --file ../data/balanced_conversations_24.jsonl \
        --exclude "case-a,case-b"

    python list_case_ids.py \
        --file ../data/balanced_conversations_24.jsonl \
        --exclude-file unnecessary_case_ids.txt \
        --output necessary

The input case file may be JSONL, a JSON array of objects, or a JSON object
containing a ``cases`` array. An exclusion file may contain comma-separated or
newline-separated IDs, JSON objects with ``case_id``, or JSONL case objects.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def _case_records(value: Any) -> Iterable[dict[str, Any]]:
    """Yield case-like dictionaries from common JSON/JSONL structures."""
    if isinstance(value, dict):
        if "case_id" in value:
            yield value
            return
        cases = value.get("cases")
        if isinstance(cases, list):
            for item in cases:
                yield from _case_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _case_records(item)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _ids_from_json_text(text: str) -> list[str]:
    """Extract case IDs from JSON text, JSON arrays, or JSONL text."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        values: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                values.extend(
                    str(record["case_id"])
                    for record in _case_records(json.loads(line))
                    if str(record.get("case_id", "")).strip()
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        return _unique(values)

    return _unique(
        str(record["case_id"])
        for record in _case_records(parsed)
        if str(record.get("case_id", "")).strip()
    )


def load_case_ids(path: Path) -> list[str]:
    """Load case IDs from a JSON, JSONL, or case-ID text file."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        ids = _ids_from_json_text(text)
    except ValueError:
        # A plain exclusion list is also accepted: one ID per line or comma
        # separated. Case files still fail clearly if they contain malformed
        # JSON rather than silently producing an empty selection.
        ids = _unique(re.split(r"[,\r\n]+", text))
    if ids:
        return ids
    raise ValueError(f"No case_id values found in {path}")


def parse_id_arguments(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return _unique(
        part
        for value in values
        for part in re.split(r"[,\r\n]+", value)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print available, excluded, or necessary case IDs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        type=Path,
        help="JSONL/JSON case file containing case_id values.",
    )
    source.add_argument(
        "--available-ids",
        nargs="+",
        help="Available IDs supplied directly, separated by commas or spaces.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="IDs to exclude; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        help="File containing IDs to exclude (text, JSON, or JSONL).",
    )
    parser.add_argument(
        "--output",
        choices=("all", "available", "excluded", "necessary"),
        default="all",
        help="Output one labelled result or only one comma-separated list (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        available = (
            load_case_ids(args.file)
            if args.file is not None
            else parse_id_arguments(args.available_ids)
        )
        if not available:
            raise ValueError("No available case IDs were supplied")

        excluded = parse_id_arguments(args.exclude)
        if args.exclude_file is not None:
            excluded = _unique([*excluded, *load_case_ids(args.exclude_file)])

        available_set = set(available)
        excluded_in_file = [case_id for case_id in excluded if case_id in available_set]
        not_found = [case_id for case_id in excluded if case_id not in available_set]
        necessary = [case_id for case_id in available if case_id not in set(excluded)]
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    values = {
        "available": ",".join(available),
        "excluded": ",".join(excluded_in_file),
        "necessary": ",".join(necessary),
    }
    if args.output == "all":
        print(f"available_case_ids={values['available']}")
        print(f"excluded_case_ids={values['excluded']}")
        print(f"necessary_case_ids={values['necessary']}")
    else:
        print(values[args.output])

    if not_found:
        print(
            "Warning: excluded IDs not found in the available case file: "
            + ",".join(not_found),
            file=sys.stderr,
        )
    print(
        f"Counts: available={len(available)}, excluded={len(excluded_in_file)}, "
        f"necessary={len(necessary)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
