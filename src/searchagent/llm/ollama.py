"""Ollama OpenAI-compatible client adapter."""

from __future__ import annotations

from typing import Any

from searchagent.llm.base import ClientConfig, OllamaConfig
from searchagent.llm.openai import OpenAIClient


class OllamaClient(OpenAIClient):
    def __init__(self, *, config: ClientConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            provider_config = config.ollama or OllamaConfig()
            openai_config = ClientConfig(
                type="openai",
                model=config.model,
                retry_policy=config.retry_policy,
                default_kwargs=config.default_kwargs,
                openai=provider_config,
            )
            super().__init__(config=openai_config)
            return
        super().__init__(**kwargs)
