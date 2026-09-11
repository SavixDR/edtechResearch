from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.result_writer import rebuild_flat_csv
from src.utils import load_json, read_jsonl, write_jsonl
from src.validation import parse_output, validate_schema


DEFAULT_INPUTS = [
    "outputs/pilots/live_runs/run_20260905_221424_glm/results.jsonl",
    "outputs/pilots/live_runs/run_20260906_004117_glm/results.jsonl",
    "outputs/pilots/live_runs/run_20260906_021613_glm/results.jsonl",
    "outputs/pilots/live_runs/run_20260906_053853_glm/results.jsonl",
]


def merge_results(project_root: Path, input_paths: list[Path], output_dir: Path) -> Path:
    schema = load_json(project_root / "assets/judge_v3_output_schema.json")
    by_execution_key: dict[str, list[dict[str, Any]]] = {}
    for input_path in input_paths:
        for record in read_jsonl(input_path):
            by_execution_key.setdefault(str(record["execution_key"]), []).append(record)

    merged: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for execution_key, candidates in by_execution_key.items():
        selected = next(
            (record for record in reversed(candidates) if record["response"]["status"] == "ok"),
            None,
        )
        recovered = False
        if selected is None:
            selected = next((record for record in reversed(candidates) if record["response"]["raw_text"]), None)
            if selected is None:
                unresolved.append({
                    "execution_key": execution_key,
                    "case_id": candidates[-1]["case"]["case_id"],
                    "statuses": [record["response"]["status"] for record in candidates],
                })
                continue
            response = selected["response"]
            parsed, parse_method, parse_error = parse_output(response["raw_text"], "tolerant")
            schema_errors = validate_schema(parsed, schema)
            if parsed is None or parse_error is not None or schema_errors:
                unresolved.append({
                    "execution_key": execution_key,
                    "case_id": selected["case"]["case_id"],
                    "statuses": [record["response"]["status"] for record in candidates],
                    "parse_error": parse_error,
                    "validation_errors": schema_errors,
                })
                continue
            selected = json.loads(json.dumps(selected))
            response = selected["response"]
            response["status"] = "ok"
            response["parsed_json"] = parsed
            response["parse_method"] = parse_method
            response["parse_error"] = None
            response["validation_errors"] = schema_errors
            recovered = True
        else:
            selected = json.loads(json.dumps(selected))

        selected["merge_provenance"] = {
            "source_run_id": selected["run_id"],
            "source_status": candidates[-1]["response"]["status"],
            "source_execution_attempts": len(candidates),
            "recovered_from_raw_text": recovered,
        }
        merged.append(selected)

    merged.sort(key=lambda record: str(record["case"]["case_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    write_jsonl(results_path, merged)
    rebuild_flat_csv(results_path, output_dir / "results_flat.csv")
    manifest = {
        "run_id": output_dir.name,
        "source_files": [str(path) for path in input_paths],
        "record_count": len(merged),
        "unique_input_case_count": len(by_execution_key),
        "unresolved_count": len(unresolved),
        "unresolved_cases": unresolved,
        "recovered_record_count": sum(
            1 for record in merged if record["merge_provenance"]["recovered_from_raw_text"]
        ),
        "selection_rule": "latest successful record per execution_key; otherwise recover latest raw response in tolerant mode",
        "output_schema_file": str(project_root / "assets/judge_v3_output_schema.json"),
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge complementary model T0 result files.")
    parser.add_argument("--output-dir", default="outputs/pilots/live_runs/merged_glm_t0")
    parser.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    result = merge_results(project_root, [project_root / path for path in args.inputs], project_root / args.output_dir)
    print(result)


if __name__ == "__main__":
    main()