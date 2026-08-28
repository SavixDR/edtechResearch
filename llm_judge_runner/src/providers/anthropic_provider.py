from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

from ..config import ModelConfig
from .base import ProviderResult


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {k: getattr(usage, k) for k in dir(usage) if k.endswith("tokens") and not k.startswith("_")}


class AnthropicMessagesProvider:
    def generate(self, model: ModelConfig, system_prompt: str, user_prompt: str,
                 max_output_tokens: int, timeout_seconds: int) -> ProviderResult:
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {model.api_key_env}")
        client = Anthropic(api_key=api_key, timeout=timeout_seconds)
        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
        }
        if model.temperature is not None:
            kwargs["temperature"] = model.temperature
        response = client.messages.create(**kwargs)
        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return ProviderResult(
            raw_text="\n".join(text_parts).strip(),
            provider_response_id=getattr(response, "id", None),
            provider_model=getattr(response, "model", None),
            usage=_usage_dict(getattr(response, "usage", None)),
            finish_reason=getattr(response, "stop_reason", None),
            provider_metadata={"api_family": "messages"},
        )
