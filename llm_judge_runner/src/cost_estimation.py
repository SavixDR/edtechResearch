from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from .utils import write_jsonl


def token_cost(tokens: float, price_per_million: float | None) -> float | None:
    if tokens == 0:
        return 0.0
    if price_per_million is None:
        return None
    return tokens * float(price_per_million) / 1_000_000


def _sum_cost(parts: list[float | None]) -> float | None:
    return None if any(value is None for value in parts) else sum(value for value in parts if value is not None)


def summarize_costs(
    rows: list[dict[str, Any]],
    pricing: dict[str, Any],
    *,
    max_output_tokens: int,
    repeats: int,
    hit_rates: list[float],
    first_request_populates_cache: bool,
) -> dict[str, Any]:
    request_count = len(rows) * repeats
    input_tokens = sum(row["adjusted_full_input_tokens"] for row in rows) * repeats
    output_upper_tokens = request_count * max_output_tokens
    regular_rate = pricing.get("input_per_million")
    cached_rate = pricing.get("cached_input_per_million")
    write_rate = pricing.get("cache_write_per_million")
    output_rate = pricing.get("output_per_million")
    no_cache_input_cost = token_cost(input_tokens, regular_rate)
    output_upper_cost = token_cost(output_upper_tokens, output_rate)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cache_group"]].append(row)

    scenarios = []
    for hit_rate in hit_rates:
        dynamic_tokens = sum(row["adjusted_dynamic_input_tokens"] for row in rows) * repeats
        first_stable_tokens = 0
        remaining_stable_tokens = 0
        for group_rows in grouped.values():
            stable = max(row["adjusted_stable_input_tokens"] for row in group_rows)
            group_requests = len(group_rows) * repeats
            if first_request_populates_cache and group_requests:
                first_stable_tokens += stable
                remaining_stable_tokens += stable * (group_requests - 1)
            else:
                remaining_stable_tokens += stable * group_requests
        cached_tokens = remaining_stable_tokens * hit_rate
        uncached_stable_tokens = remaining_stable_tokens - cached_tokens
        first_rate = write_rate if write_rate is not None else regular_rate
        input_cost = _sum_cost(
            [
                token_cost(dynamic_tokens + uncached_stable_tokens, regular_rate),
                token_cost(first_stable_tokens, first_rate),
                token_cost(cached_tokens, cached_rate),
            ]
        )
        scenarios.append(
            {
                "assumed_cache_hit_rate": hit_rate,
                "dynamic_uncached_tokens": dynamic_tokens,
                "first_or_cache_write_tokens": first_stable_tokens,
                "stable_uncached_tokens": uncached_stable_tokens,
                "stable_cached_tokens": cached_tokens,
                "projected_input_cost_usd": input_cost,
                "projected_total_upper_cost_usd": _sum_cost([input_cost, output_upper_cost]),
            }
        )

    return {
        "case_count": len(rows),
        "repeats": repeats,
        "planned_request_count": request_count,
        "max_output_tokens_per_request": max_output_tokens,
        "no_cache_input_tokens": input_tokens,
        "output_upper_bound_tokens": output_upper_tokens,
        "no_cache_input_cost_usd": no_cache_input_cost,
        "output_upper_bound_cost_usd": output_upper_cost,
        "no_cache_total_upper_cost_usd": _sum_cost([no_cache_input_cost, output_upper_cost]),
        "cache_scenarios": scenarios,
    }


def _number(value: Any) -> int | None:
    """Return a non-negative integer token count, or None when unavailable."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, number)


def _nested_number(mapping: dict[str, Any], *path: str) -> int | None:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _number(current)


def _first_number(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def normalize_live_usage(adapter: str, usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider usage into mutually exclusive billable token categories."""
    warnings: list[str] = []
    if not isinstance(usage, dict) or not usage:
        return {
            "uncached_input_tokens": None,
            "cached_input_tokens": None,
            "cache_write_input_tokens": None,
            "output_tokens": None,
            "provider_total_tokens": None,
            "usage_available": False,
            "warnings": ["provider_usage_missing"],
        }

    if adapter == "anthropic_messages":
        uncached = _number(usage.get("input_tokens"))
        cached = _number(usage.get("cache_read_input_tokens")) or 0
        cache_write = _number(usage.get("cache_creation_input_tokens")) or 0
        output = _number(usage.get("output_tokens"))
        provider_total = None
    elif adapter == "gemini_generate_content":
        prompt_total = _number(usage.get("prompt_token_count"))
        cached = _number(usage.get("cached_content_token_count")) or 0
        uncached = None if prompt_total is None else max(0, prompt_total - cached)
        cache_write = 0
        candidates = _number(usage.get("candidates_token_count"))
        thoughts = _number(usage.get("thoughts_token_count")) or 0
        output = None if candidates is None else candidates + thoughts
        provider_total = _number(usage.get("total_token_count"))
    else:
        prompt_total = _first_number(
            _number(usage.get("input_tokens")),
            _number(usage.get("prompt_tokens")),
        )
        cached = _first_number(
            _nested_number(usage, "input_tokens_details", "cached_tokens"),
            _nested_number(usage, "prompt_tokens_details", "cached_tokens"),
            _number(usage.get("prompt_cache_hit_tokens")),
        ) or 0
        cache_miss = _number(usage.get("prompt_cache_miss_tokens"))
        uncached = cache_miss if cache_miss is not None else (
            None if prompt_total is None else max(0, prompt_total - cached)
        )
        cache_write = 0
        output = _first_number(
            _number(usage.get("output_tokens")),
            _number(usage.get("completion_tokens")),
        )
        provider_total = _number(usage.get("total_tokens"))

    if uncached is None:
        warnings.append("input_token_count_missing")
    if output is None:
        warnings.append("output_token_count_missing")
    return {
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "output_tokens": output,
        "provider_total_tokens": provider_total,
        "usage_available": True,
        "warnings": warnings,
    }


def calculate_live_cost(
    *,
    adapter: str,
    usage: dict[str, Any],
    pricing: dict[str, Any],
    api_attempts: int,
    provider_response_received: bool,
) -> dict[str, Any]:
    """Calculate USD cost from provider-reported usage and configured rates."""
    normalized = normalize_live_usage(adapter, usage)
    warnings = list(normalized["warnings"])

    # A provider initialization failure made no request and therefore incurred no API token cost.
    if not provider_response_received and api_attempts == 0:
        normalized = {
            **normalized,
            "uncached_input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
        }
        warnings = [warning for warning in warnings if warning != "provider_usage_missing"]
        warnings.append("no_api_request_was_made")
    elif not provider_response_received:
        warnings.append("failed_api_attempt_usage_unavailable")

    if api_attempts > 1:
        warnings.append("earlier_retry_attempt_usage_unavailable")

    categories = {
        "uncached_input": (normalized["uncached_input_tokens"], pricing.get("input_per_million")),
        "cached_input": (normalized["cached_input_tokens"], pricing.get("cached_input_per_million")),
        "cache_write_input": (normalized["cache_write_input_tokens"], pricing.get("cache_write_per_million")),
        "output": (normalized["output_tokens"], pricing.get("output_per_million")),
    }
    component_costs: dict[str, float | None] = {}
    for name, (tokens, rate) in categories.items():
        component_costs[f"{name}_cost_usd"] = None if tokens is None else token_cost(tokens, rate)
        if tokens not in (None, 0) and rate is None:
            warnings.append(f"{name}_price_missing")

    complete = not warnings or warnings == ["no_api_request_was_made"]
    total = _sum_cost(list(component_costs.values())) if complete else None
    known_cost = sum(value for value in component_costs.values() if value is not None)
    return {
        "normalized_usage": normalized,
        "rates_per_million_usd": {
            "uncached_input": pricing.get("input_per_million"),
            "cached_input": pricing.get("cached_input_per_million"),
            "cache_write_input": pricing.get("cache_write_per_million"),
            "output": pricing.get("output_per_million"),
        },
        **component_costs,
        "known_cost_usd": known_cost,
        "total_cost_usd": total,
        "cost_complete": complete,
        "warnings": sorted(set(warnings)),
    }


def cost_row_from_result(record: dict[str, Any]) -> dict[str, Any]:
    cost = record.get("cost", {})
    normalized = cost.get("normalized_usage", {})
    response = record.get("response", {})
    return {
        "execution_key": record.get("execution_key"),
        "run_id": record.get("run_id"),
        "collected_at_utc": record.get("collected_at_utc"),
        "case_id": record.get("case", {}).get("case_id"),
        "test_id": record.get("experiment", {}).get("test_id"),
        "model_key": record.get("judge", {}).get("model_key"),
        "collection_attempt": response.get("collection_attempt", 1),
        "status": response.get("status"),
        "api_attempts": response.get("api_attempts", 0),
        "uncached_input_tokens": normalized.get("uncached_input_tokens"),
        "cached_input_tokens": normalized.get("cached_input_tokens"),
        "cache_write_input_tokens": normalized.get("cache_write_input_tokens"),
        "output_tokens": normalized.get("output_tokens"),
        "uncached_input_cost_usd": cost.get("uncached_input_cost_usd"),
        "cached_input_cost_usd": cost.get("cached_input_cost_usd"),
        "cache_write_input_cost_usd": cost.get("cache_write_input_cost_usd"),
        "output_cost_usd": cost.get("output_cost_usd"),
        "known_cost_usd": cost.get("known_cost_usd"),
        "total_cost_usd": cost.get("total_cost_usd"),
        "cost_complete": bool(cost.get("cost_complete", False)),
        "warnings": cost.get("warnings", ["cost_data_missing"]),
    }


def _aggregate_live_cost_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_fields = (
        "uncached_input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"
    )
    cost_fields = (
        "uncached_input_cost_usd", "cached_input_cost_usd", "cache_write_input_cost_usd", "output_cost_usd"
    )
    complete = all(bool(row.get("cost_complete")) for row in rows)
    totals: dict[str, Any] = {
        "request_record_count": len(rows),
        "successful_record_count": sum(row.get("status") == "ok" for row in rows),
        "error_record_count": sum(row.get("status") != "ok" for row in rows),
        "complete_cost_record_count": sum(bool(row.get("cost_complete")) for row in rows),
        "incomplete_cost_record_count": sum(not bool(row.get("cost_complete")) for row in rows),
    }
    for field in token_fields:
        totals[field] = sum(int(row.get(field) or 0) for row in rows)
    for field in cost_fields:
        totals[field] = sum(float(row.get(field) or 0.0) for row in rows)
    totals["known_cost_usd"] = sum(float(row.get("known_cost_usd") or 0.0) for row in rows)
    totals["total_cost_usd"] = totals["known_cost_usd"] if complete else None
    totals["cost_complete"] = complete
    return totals


def summarize_live_costs(
    rows: list[dict[str, Any]], pricing_config: dict[str, Any], pricing_hash: str
) -> dict[str, Any]:
    def grouped(field: str) -> dict[str, dict[str, Any]]:
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row.get(field) or "unknown")].append(row)
        return {key: _aggregate_live_cost_rows(group_rows) for key, group_rows in sorted(values.items())}

    return {
        "report_version": "1.0",
        "currency": pricing_config.get("currency", "USD"),
        "prices_as_of": pricing_config.get("prices_as_of"),
        "pricing_hash": pricing_hash,
        "calculation_basis": "Provider-reported token usage multiplied by configured per-million-token rates.",
        "summary": _aggregate_live_cost_rows(rows),
        "by_model": grouped("model_key"),
        "by_test": grouped("test_id"),
        "by_status": grouped("status"),
    }


def write_live_cost_reports(
    run_dir: Path,
    result_records: list[dict[str, Any]],
    pricing_config: dict[str, Any],
    pricing_hash: str,
) -> None:
    rows = [cost_row_from_result(record) for record in result_records]
    costs_path = run_dir / "costs.jsonl"
    costs_temp = run_dir / "costs.jsonl.tmp"
    write_jsonl(costs_temp, rows)
    costs_temp.replace(costs_path)
    summary = summarize_live_costs(rows, pricing_config, pricing_hash)
    summary["run_id"] = run_dir.name
    summary_path = run_dir / "cost_summary.json"
    summary_temp = run_dir / "cost_summary.json.tmp"
    summary_temp.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_temp.replace(summary_path)

    csv_rows: list[dict[str, Any]] = [{"scope": "run", "key": "all", **summary["summary"]}]
    csv_rows.extend(
        {"scope": "model", "key": key, **value} for key, value in summary["by_model"].items()
    )
    fields: list[str] = []
    for row in csv_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    csv_path = run_dir / "cost_summary.csv"
    csv_temp = run_dir / "cost_summary.csv.tmp"
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    csv_temp.replace(csv_path)
