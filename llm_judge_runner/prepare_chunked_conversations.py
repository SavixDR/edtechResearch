from __future__ import annotations

"""Split prepared JSONL cases at complete student-assistant exchange boundaries."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.cases import load_cases, validate_case
from src.utils import read_jsonl


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "balanced_conversations_24.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "balanced_conversations_24_chunked.jsonl"
DEFAULT_MAX_CHARACTERS = 12_000


def flatten(units: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [turn for unit in units for turn in unit]


def conversation_characters(turns: list[dict[str, Any]]) -> int:
    return sum(len(str(turn.get("text", ""))) for turn in turns)


def make_exchange_units(case: dict[str, Any]) -> list[list[dict[str, Any]]]:
    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"Case {case.get('case_id', '<unknown>')} has no turns")
    if len(turns) % 2:
        raise ValueError(
            f"Case {case['case_id']} has an unmatched final turn; "
            "the chunker will not split incomplete exchanges"
        )

    units: list[list[dict[str, Any]]] = []
    for position in range(0, len(turns), 2):
        student = turns[position]
        assistant = turns[position + 1]
        if student.get("speaker") != "student" or assistant.get("speaker") != "assistant":
            raise ValueError(
                f"Case {case['case_id']} turns {position + 1}-{position + 2} are not a "
                "student-assistant exchange"
            )
        units.append([student, assistant])
    return units


def chunk_exchange_units(
    case: dict[str, Any], max_characters: int
) -> list[list[dict[str, Any]]]:
    units = make_exchange_units(case)
    chunks: list[list[dict[str, Any]]] = []
    current: list[list[dict[str, Any]]] = []
    current_characters = 0

    for exchange_index, unit in enumerate(units, start=1):
        unit_characters = conversation_characters(unit)
        if unit_characters > max_characters:
            raise ValueError(
                f"Case {case['case_id']} exchange {exchange_index} contains "
                f"{unit_characters:,} characters, exceeding the {max_characters:,}-character "
                "limit; the chunker will not split a student-assistant exchange"
            )
        if current and current_characters + unit_characters > max_characters:
            chunks.append(flatten(current))
            current = [unit]
            current_characters = unit_characters
        else:
            current.append(unit)
            current_characters += unit_characters

    if current:
        chunks.append(flatten(current))
    return chunks


def _with_chunk_position(
    case: dict[str, Any], chunk_index: int, chunk_count: int, turns: list[dict[str, Any]], max_characters: int
) -> dict[str, Any]:
    if chunk_count == 1:
        case_id = case["case_id"]
        chunk_id = case["chunk_id"]
    else:
        suffix = f"chunk_{chunk_index:03d}"
        chunk_id = f"{case['conversation_id']}::{suffix}"
        case_id = (
            f"{case['assignment_id']}::{case['conversation_id']}::{suffix}::"
            f"{case['test_id']}::{case['variant_id']}"
        )

    existing_metadata = case.get("transformation_metadata", {})
    if not isinstance(existing_metadata, dict):
        raise ValueError(f"Case {case['case_id']} transformation_metadata must be an object")
    if "chunking" in existing_metadata:
        raise ValueError(f"Case {case['case_id']} already has chunking transformation metadata")
    metadata = {
        **existing_metadata,
        "chunking": {
            "source_case_id": case["case_id"],
            "strategy": "complete_student_assistant_exchanges",
            "max_conversation_characters": max_characters,
            "overlap_exchange_count": 0,
        },
    }

    output: dict[str, Any] = {}
    for key, value in case.items():
        if key in {"chunk_index", "chunk_count"}:
            continue
        if key == "case_id":
            output[key] = case_id
        elif key == "chunk_id":
            output[key] = chunk_id
            output["chunk_index"] = chunk_index
            output["chunk_count"] = chunk_count
        elif key == "transformation_metadata":
            output[key] = metadata
        elif key == "turns":
            output[key] = turns
        else:
            output[key] = value
    return output


def build_chunked_cases(
    source_cases: list[dict[str, Any]], max_characters: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_characters < 1:
        raise ValueError("max_characters must be a positive integer")
    source_ids: set[str] = set()
    output_cases: list[dict[str, Any]] = []
    split_conversations = 0

    for case in source_cases:
        validate_case(case)
        source_case_id = str(case["case_id"])
        if source_case_id in source_ids:
            raise ValueError(f"Duplicate source case_id: {source_case_id}")
        source_ids.add(source_case_id)
        if case.get("chunk_index") is not None or case.get("chunk_count") is not None:
            raise ValueError(f"Source case {source_case_id} already contains chunk position fields")

        chunks = chunk_exchange_units(case, max_characters)
        if len(chunks) > 1:
            split_conversations += 1
        for chunk_index, turns in enumerate(chunks, start=1):
            output_cases.append(
                _with_chunk_position(case, chunk_index, len(chunks), turns, max_characters)
            )

    output_ids = [case["case_id"] for case in output_cases]
    chunk_ids = [case["chunk_id"] for case in output_cases]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("Generated duplicate case_id values")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Generated duplicate chunk_id values")
    return output_cases, {
        "source_conversations": len(source_cases),
        "split_conversations": split_conversations,
        "singleton_conversations": len(source_cases) - split_conversations,
        "output_chunks": len(output_cases),
    }


def validate_reconstruction(
    source_cases: list[dict[str, Any]], output_cases: list[dict[str, Any]], max_characters: int
) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in output_cases:
        source_id = chunk["transformation_metadata"]["chunking"]["source_case_id"]
        by_source[str(source_id)].append(chunk)
        if conversation_characters(chunk["turns"]) > max_characters:
            raise ValueError(f"Generated chunk {chunk['case_id']} exceeds the character limit")

    for source in source_cases:
        chunks = sorted(by_source.get(source["case_id"], []), key=lambda item: item["chunk_index"])
        if not chunks:
            raise ValueError(f"Source case {source['case_id']} has no generated chunks")
        expected_count = len(chunks)
        if [chunk["chunk_index"] for chunk in chunks] != list(range(1, expected_count + 1)):
            raise ValueError(f"Source case {source['case_id']} has non-contiguous chunk indexes")
        if any(chunk["chunk_count"] != expected_count for chunk in chunks):
            raise ValueError(f"Source case {source['case_id']} has inconsistent chunk_count values")
        reconstructed = [turn for chunk in chunks for turn in chunk["turns"]]
        if reconstructed != source["turns"]:
            raise ValueError(f"Turn reconstruction failed for source case {source['case_id']}")

    if set(by_source) != {str(case["case_id"]) for case in source_cases}:
        raise ValueError("Generated chunks contain an unknown source_case_id")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk prepared conversation JSONL without splitting student-assistant exchanges."
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-characters",
        type=int,
        default=DEFAULT_MAX_CHARACTERS,
        help="Maximum sum of turn-text characters per chunk (default: 12000).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")

    source_cases = read_jsonl(input_path)
    output_cases, summary = build_chunked_cases(source_cases, args.max_characters)
    validate_reconstruction(source_cases, output_cases, args.max_characters)
    write_jsonl_atomic(output_path, output_cases, args.overwrite)
    loaded = load_cases(output_path)
    if len(loaded) != len(output_cases):
        raise RuntimeError("Existing case loader returned an unexpected number of chunks")

    print(f"Wrote {summary['output_chunks']:,} chunks to {output_path}")
    print(f"Source conversations: {summary['source_conversations']:,}")
    print(f"Conversations split: {summary['split_conversations']:,}")
    print(f"Singleton conversations: {summary['singleton_conversations']:,}")
    print(f"Maximum conversation characters per chunk: {args.max_characters:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
