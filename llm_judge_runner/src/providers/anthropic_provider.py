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
        # Keep retries under the runner's control. The Anthropic SDK otherwise
        # retries connection failures twice internally, which can turn one
        # recorded runner attempt into three billable provider requests.
        client = Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        cache_control = (model.extra or {}).get("cache_control")
        if cache_control is not None:
            if not isinstance(cache_control, dict):
                raise ValueError(
                    f"Model {model.key} extra.cache_control must be an object"
                )
            if cache_control.get("type") != "ephemeral":
                raise ValueError(
                    f"Model {model.key} extra.cache_control.type must be 'ephemeral'"
                )
            if cache_control.get("ttl") not in {None, "5m", "1h"}:
                raise ValueError(
                    f"Model {model.key} extra.cache_control.ttl must be '5m' or '1h'"
                )
            system: str | list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": dict(cache_control),
                }
            ]
        else:
            system = system_prompt

        kwargs: dict[str, Any] = {
            "model": model.model_id,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
        }
        if model.reasoning_effort:
            kwargs["output_config"] = {
                    "effort": model.reasoning_effort
                }

        if model.temperature is not None:
            kwargs["temperature"] = model.temperature
        # Stream long generations so intermediaries see regular network
        # activity instead of one idle non-streaming connection. The SDK
        # accumulates the events into the same final Message shape used below.
        with client.messages.stream(**kwargs) as stream:
            for _event in stream:
                pass
            response = stream.get_final_message()
        text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return ProviderResult(
            raw_text="\n".join(text_parts).strip(),
            provider_response_id=getattr(response, "id", None),
            provider_model=getattr(response, "model", None),
            usage=_usage_dict(getattr(response, "usage", None)),
            finish_reason=getattr(response, "stop_reason", None),
            provider_metadata={
                "api_family": "messages",
                "streaming": True,
                "sdk_max_retries": 0,
            },
        )
