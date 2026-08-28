from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from ..config import ModelConfig
from .base import ProviderResult


def _dump_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {"repr": str(usage)}


class OpenAIResponsesProvider:
    def generate(self, model: ModelConfig, system_prompt: str, user_prompt: str,
                 max_output_tokens: int, timeout_seconds: int) -> ProviderResult:
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {model.api_key_env}")
        client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_output_tokens,
            "store": bool(model.extra.get("store", False)),
        }
        if model.reasoning_effort:
            kwargs["reasoning"] = {"effort": model.reasoning_effort}
        if model.temperature is not None:
            kwargs["temperature"] = model.temperature
        response = client.responses.create(**kwargs)
        return ProviderResult(
            raw_text=response.output_text or "",
            provider_response_id=getattr(response, "id", None),
            provider_model=getattr(response, "model", None),
            usage=_dump_usage(getattr(response, "usage", None)),
            finish_reason=getattr(response, "status", None),
            provider_metadata={"api_family": "responses"},
        )
