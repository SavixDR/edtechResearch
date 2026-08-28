from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_json, project_path


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    provider: str
    adapter: str
    model_id: str
    country: str
    enabled: bool
    api_key_env: str
    base_url_env: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    requested_version_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    experiment_name: str
    suite_version: str
    prompt_version: str
    cases_file: Path
    models_file: Path
    prompt_variants_file: Path
    assignments_file: Path
    results_root: Path
    max_output_tokens: int
    default_repeats: int
    parse_mode: str
    output_schema_file: Path | None
    request_timeout_seconds: int
    max_api_attempts: int
    retry_backoff_seconds: float
    save_case_snapshot: bool
    save_prompt_snapshots: bool


def load_settings(project_root: Path, settings_path: Path) -> Settings:
    raw = load_json(settings_path)
    parse_mode = raw.get("parse_mode", "strict")
    if parse_mode not in {"strict", "tolerant"}:
        raise ValueError("parse_mode must be 'strict' or 'tolerant'")
    return Settings(
        experiment_name=raw["experiment_name"],
        suite_version=str(raw["suite_version"]),
        prompt_version=str(raw["prompt_version"]),
        cases_file=project_path(project_root, raw["cases_file"]),
        models_file=project_path(project_root, raw["models_file"]),
        prompt_variants_file=project_path(project_root, raw["prompt_variants_file"]),
        assignments_file=project_path(project_root, raw["assignments_file"]),
        results_root=project_path(project_root, raw["results_root"]),
        max_output_tokens=int(raw.get("max_output_tokens", 8000)),
        default_repeats=int(raw.get("default_repeats", 1)),
        parse_mode=parse_mode,
        output_schema_file=project_path(project_root, raw.get("output_schema_file")),
        request_timeout_seconds=int(raw.get("request_timeout_seconds", 300)),
        max_api_attempts=max(1, int(raw.get("max_api_attempts", 2))),
        retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 3)),
        save_case_snapshot=bool(raw.get("save_case_snapshot", True)),
        save_prompt_snapshots=bool(raw.get("save_prompt_snapshots", True)),
    )


def load_models(path: Path) -> dict[str, ModelConfig]:
    raw = load_json(path)
    out: dict[str, ModelConfig] = {}
    for key, item in raw.items():
        out[key] = ModelConfig(
            key=key,
            display_name=item.get("display_name", key),
            provider=item["provider"],
            adapter=item["adapter"],
            model_id=item["model_id"],
            country=item.get("country", ""),
            enabled=bool(item.get("enabled", True)),
            api_key_env=item["api_key_env"],
            base_url_env=item.get("base_url_env"),
            reasoning_effort=item.get("reasoning_effort"),
            temperature=item.get("temperature"),
            requested_version_label=item.get("requested_version_label"),
            extra=item.get("extra", {}),
        )
    return out
