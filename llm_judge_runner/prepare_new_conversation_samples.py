from __future__ import annotations

"""Build baseline T0 JSONL cases from new_conversation_samples.csv.

The CSV is used only as the selection/metadata table. Conversation text is
loaded from the canonical per-chat JSON files.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


# The selection CSV still carries a large, legacy conversation_rows column.
# Windows builds may expose a 32-bit C long even when sys.maxsize is 64-bit.
_csv_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_limit)
        break
    except OverflowError:
        _csv_limit //= 10


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "new_conversation_samples.csv"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "data"
    / "student_chat_conversations_preprocessing"
    / "file_per_chat"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "new_conversation_samples_t0.jsonl"


@dataclass(frozen=True)
class TokenCounter:
    count: Callable[[str], int]
    description: str


def make_token_counter(mode: str, model_or_encoding: str, chars_per_token: float) -> TokenCounter:
    """Build an offline counter, optionally using a locally available tiktoken table."""
    def heuristic() -> TokenCounter:
        if chars_per_token <= 0:
            raise ValueError("--chars-per-token must be greater than zero")
        return TokenCounter(
            count=lambda text: math.ceil(len(text.encode("utf-8")) / chars_per_token),
            description=f"UTF-8-byte heuristic ({chars_per_token:g} bytes/token)",
        )

    if mode == "heuristic":
        return heuristic()

    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            encoding = tiktoken.encoding_for_model(model_or_encoding)
        except KeyError:
            encoding = tiktoken.get_encoding(model_or_encoding)
        return TokenCounter(
            count=lambda text: len(encoding.encode(text)),
            description=f"tiktoken:{encoding.name}",
        )
    # A first tiktoken use may try to download its encoding table. In auto
    # mode, treat a missing/offline cache like an unavailable dependency.
    except Exception as exc:
        if mode == "tiktoken":
            raise RuntimeError(
                "Could not initialize tiktoken. Install/cache the requested encoding, "
                "or use --tokenizer heuristic."
            ) from exc
        return heuristic()


def required_text(row: dict[str, str], field: str, row_number: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"CSV row {row_number} has no {field!r}")
    return value


def build_source_index(source_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in source_root.rglob("*.json"):
        previous = index.setdefault(path.stem, path)
        if previous != path:
            duplicates.setdefault(path.stem, [previous]).append(path)
    if duplicates:
        details = "; ".join(
            f"{chat_id}: {', '.join(map(str, paths))}"
            for chat_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate per-chat JSON filenames found: {details}")
    return index


def locate_source(
    source_root: Path,
    source_index: dict[str, Path],
    semester: str,
    topic: str,
    chat_id: str,
) -> Path:
    expected = source_root / "categorized_student_chats" / semester / topic / f"{chat_id}.json"
    if expected.is_file():
        return expected
    try:
        return source_index[chat_id]
    except KeyError as exc:
        raise FileNotFoundError(f"No per-chat JSON found for chatId={chat_id}") from exc


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def make_exchange_units(payload: dict[str, Any], source_path: Path) -> list[list[dict[str, str]]]:
    raw_rows = payload.get("conversation_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"No non-empty conversation_rows list in {source_path}")

    units: list[list[dict[str, str]]] = []
    for position, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"conversation_rows[{position}] is not an object in {source_path}")
        original_index = raw.get("interactionCount", position)
        prompt = str(raw.get("prompt") or "").strip()
        response = str(raw.get("response") or "").strip()
        unit: list[dict[str, str]] = []
        if prompt:
            unit.append(
                {
                    "turn_index": f"{original_index}.student",
                    "speaker": "student",
                    "text": prompt,
                }
            )
        if response:
            unit.append(
                {
                    "turn_index": f"{original_index}.assistant",
                    "speaker": "assistant",
                    "text": response,
                }
            )
        if unit:
            units.append(unit)

    if not units:
        raise ValueError(f"Conversation has no non-empty prompt/response text in {source_path}")
    return units


def flatten(units: Iterable[list[dict[str, str]]]) -> list[dict[str, str]]:
    return [turn for unit in units for turn in unit]


def turn_payload_tokens(turns: list[dict[str, str]], counter: TokenCounter) -> int:
    # Compact JSON closely approximates the dynamic payload while retaining all
    # field names. A separate fixed allowance covers rendered headers and IDs.
    text = json.dumps({"turns": turns}, ensure_ascii=False, separators=(",", ":"))
    return counter.count(text)


def chunk_exchange_units(
    units: list[list[dict[str, str]]],
    max_payload_tokens: int,
    per_chunk_overhead_tokens: int,
    counter: TokenCounter,
    conversation_id: str,
) -> list[list[dict[str, str]]]:
    """Greedily chunk at exchange boundaries without separating a Q/A pair."""
    if max_payload_tokens <= per_chunk_overhead_tokens:
        raise ValueError("The chunk token budget must exceed --per-chunk-overhead-tokens")

    chunks: list[list[dict[str, str]]] = []
    current_units: list[list[dict[str, str]]] = []

    for unit in units:
        unit_tokens = turn_payload_tokens(unit, counter) + per_chunk_overhead_tokens
        if unit_tokens > max_payload_tokens:
            index = unit[0]["turn_index"].split(".", 1)[0]
            raise ValueError(
                f"Conversation {conversation_id}, interaction {index} needs about "
                f"{unit_tokens:,} tokens, exceeding the {max_payload_tokens:,}-token chunk budget. "
                "Increase the context/budget values; the script will not silently split a "
                "student-assistant exchange."
            )

        candidate = flatten([*current_units, unit])
        candidate_tokens = turn_payload_tokens(candidate, counter) + per_chunk_overhead_tokens
        if current_units and candidate_tokens > max_payload_tokens:
            chunks.append(flatten(current_units))
            current_units = [unit]
        else:
            current_units.append(unit)

    if current_units:
        chunks.append(flatten(current_units))
    return chunks


def build_cases(
    csv_path: Path,
    source_root: Path,
    test_id: str,
    variant_id: str,
    prompt_variant_id: str,
    max_payload_tokens: int,
    per_chunk_overhead_tokens: int,
    counter: TokenCounter,
    include_singleton_chunk_position: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_index = build_source_index(source_root)
    cases: list[dict[str, Any]] = []
    seen_conversations: set[str] = set()
    selected_count = 0
    chunked_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for row_number, row in enumerate(reader, start=2):
            selected_count += 1
            chat_id = required_text(row, "chatId", row_number)
            student_id = required_text(row, "userId", row_number)
            topic = required_text(row, "topic", row_number)
            semester = required_text(row, "semester", row_number)
            assignment_id = f"{semester}_{topic}"

            if chat_id in seen_conversations:
                raise ValueError(f"Duplicate chatId={chat_id} in selection CSV")
            seen_conversations.add(chat_id)

            source_path = locate_source(source_root, source_index, semester, topic, chat_id)
            payload = load_json_object(source_path)
            source_chat_id = str(payload.get("chatId") or "").strip()
            source_student_id = str(payload.get("userId") or "").strip()
            if source_chat_id and source_chat_id != chat_id:
                raise ValueError(f"chatId mismatch between CSV and {source_path}")
            if source_student_id and source_student_id != student_id:
                raise ValueError(f"userId mismatch between CSV and {source_path}")

            units = make_exchange_units(payload, source_path)
            chunks = chunk_exchange_units(
                units,
                max_payload_tokens,
                per_chunk_overhead_tokens,
                counter,
                chat_id,
            )
            chunk_count = len(chunks)
            if chunk_count > 1:
                chunked_count += 1

            for chunk_index, turns in enumerate(chunks, start=1):
                if chunk_count == 1:
                    chunk_id = chat_id
                    case_id = f"{assignment_id}::{chat_id}::{test_id}::{variant_id}"
                else:
                    chunk_suffix = f"chunk_{chunk_index:03d}"
                    chunk_id = f"{chat_id}::{chunk_suffix}"
                    case_id = (
                        f"{assignment_id}::{chat_id}::{chunk_suffix}::"
                        f"{test_id}::{variant_id}"
                    )

                case: dict[str, Any] = {
                    "case_id": case_id,
                    "base_conversation_id": chat_id,
                    "conversation_id": chat_id,
                    "chunk_id": chunk_id,
                    "assignment_id": assignment_id,
                    "student_id": student_id,
                    "test_id": test_id,
                    "variant_id": variant_id,
                    "variant_type": "baseline",
                    "is_baseline": True,
                    "prompt_variant_id": prompt_variant_id,
                    "transformation_metadata": {},
                    "turns": turns,
                }
                if chunk_count > 1 or include_singleton_chunk_position:
                    # Insert the pair before assignment_id in the serialized order.
                    case = {
                        **dict(list(case.items())[:4]),
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        **dict(list(case.items())[4:]),
                    }
                cases.append(case)

    if selected_count == 0:
        raise ValueError(f"Selection CSV contains no data rows: {csv_path}")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Generated duplicate case_id values")
    return cases, {
        "selected_conversations": selected_count,
        "output_cases": len(cases),
        "chunked_conversations": chunked_count,
    }


def write_jsonl_atomic(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected per-chat conversations to baseline T0 JSONL cases."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-id", default="T0")
    parser.add_argument("--variant-id", default="base")
    parser.add_argument("--prompt-variant-id", default="canonical")
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=128_000,
        help="Use the smallest context window among the target models (default: 128000).",
    )
    parser.add_argument(
        "--reserved-prompt-tokens",
        type=int,
        default=32_000,
        help="Reserve for system/user templates, codebook, schema, and other input (default: 32000).",
    )
    parser.add_argument(
        "--reserved-output-tokens",
        type=int,
        default=8_000,
        help="Reserve for model output (default: 8000).",
    )
    parser.add_argument(
        "--safety-margin-tokens",
        type=int,
        default=8_000,
        help="Additional context-window safety margin (default: 8000).",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        help="Explicit payload budget; overrides context minus all reserves.",
    )
    parser.add_argument(
        "--per-chunk-overhead-tokens",
        type=int,
        default=512,
        help="Allowance for rendered turn headers and case metadata (default: 512).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("heuristic", "auto", "tiktoken"),
        default="heuristic",
        help="Token counting method (default: conservative offline heuristic).",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o",
        help="tiktoken model or encoding name when tiktoken is installed (default: gpt-4o).",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.0,
        help="Conservative UTF-8-byte fallback ratio if tiktoken is unavailable (default: 3.0).",
    )
    parser.add_argument(
        "--include-singleton-chunk-position",
        action="store_true",
        help="Also write chunk_index=1 and chunk_count=1 for unchunked conversations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in (
        "context_window_tokens",
        "reserved_prompt_tokens",
        "reserved_output_tokens",
        "safety_margin_tokens",
        "per_chunk_overhead_tokens",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")

    max_payload_tokens = args.max_chunk_tokens
    if max_payload_tokens is None:
        max_payload_tokens = (
            args.context_window_tokens
            - args.reserved_prompt_tokens
            - args.reserved_output_tokens
            - args.safety_margin_tokens
        )
    if max_payload_tokens <= 0:
        raise ValueError("Token reserves leave no room for conversation chunks")

    counter = make_token_counter(args.tokenizer, args.tokenizer_model, args.chars_per_token)
    cases, summary = build_cases(
        csv_path=args.input_csv.resolve(),
        source_root=args.source_root.resolve(),
        test_id=args.test_id,
        variant_id=args.variant_id,
        prompt_variant_id=args.prompt_variant_id,
        max_payload_tokens=max_payload_tokens,
        per_chunk_overhead_tokens=args.per_chunk_overhead_tokens,
        counter=counter,
        include_singleton_chunk_position=args.include_singleton_chunk_position,
    )
    output_path = args.output.resolve()
    write_jsonl_atomic(output_path, cases)

    print(f"Wrote {summary['output_cases']:,} cases to {output_path}")
    print(f"Selected conversations: {summary['selected_conversations']:,}")
    print(f"Conversations split into multiple chunks: {summary['chunked_conversations']:,}")
    print(f"Conversation payload budget: {max_payload_tokens:,} tokens")
    print(f"Token counter: {counter.description}")
    if counter.description.startswith("UTF-8-byte heuristic"):
        print(
            "Warning: using a conservative cross-provider token estimate.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
