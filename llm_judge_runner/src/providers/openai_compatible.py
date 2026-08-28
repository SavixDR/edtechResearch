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


class OpenAICompatibleChatProvider:
    def generate(self, model: ModelConfig, system_prompt: str, user_prompt: str,
                 max_output_tokens: int, timeout_seconds: int) -> ProviderResult:
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {model.api_key_env}")
        if not model.base_url_env:
            raise RuntimeError(f"Model {model.key} has no base_url_env configured")
        base_url = os.getenv(model.base_url_env)
        if not base_url:
            raise RuntimeError(f"Missing environment variable {model.base_url_env}")

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if model.temperature is not None:
            kwargs["temperature"] = model.temperature
        if model.reasoning_effort:
            kwargs["reasoning_effort"] = model.reasoning_effort
        if model.extra:
            kwargs["extra_body"] = model.extra

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        return ProviderResult(
            raw_text=message.content or "",
            provider_response_id=getattr(response, "id", None),
            provider_model=getattr(response, "model", None),
            usage=_dump_usage(getattr(response, "usage", None)),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            provider_metadata={"api_family": "chat_completions"},
        )
