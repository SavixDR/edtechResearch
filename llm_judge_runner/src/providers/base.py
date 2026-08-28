from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import ModelConfig


@dataclass
class ProviderResult:
    raw_text: str
    provider_response_id: str | None = None
    provider_model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class JudgeProvider(Protocol):
    def generate(
        self,
        model: ModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        timeout_seconds: int,
    ) -> ProviderResult:
        ...
