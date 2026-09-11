from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from ..config import ModelConfig
from .base import ProviderResult


def _usage_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump()
    keys = [
        "prompt_token_count", "candidates_token_count", "total_token_count",
        "cached_content_token_count", "thoughts_token_count"
    ]
    return {k: getattr(meta, k, None) for k in keys if getattr(meta, k, None) is not None}


class GeminiGenerateContentProvider:
    def generate(self, model: ModelConfig, system_prompt: str, user_prompt: str,
                 max_output_tokens: int, timeout_seconds: int) -> ProviderResult:
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing environment variable {model.api_key_env}")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                    timeout=timeout_seconds * 1000),
                )
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_output_tokens,
        }
        
        if model.reasoning_effort:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=model.reasoning_effort
            )
            
        if model.temperature is not None:
            config_kwargs["temperature"] = model.temperature
        response = client.models.generate_content(
            model=model.model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        finish_reason = None
        try:
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass
        return ProviderResult(
            raw_text=response.text or "",
            provider_response_id=getattr(response, "response_id", None),
            provider_model=model.model_id,
            usage=_usage_dict(getattr(response, "usage_metadata", None)),
            finish_reason=finish_reason,
            provider_metadata={"api_family": "generate_content"},
        )
