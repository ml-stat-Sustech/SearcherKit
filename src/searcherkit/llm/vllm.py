"""vLLM OpenAI-compatible server adapter."""

from __future__ import annotations

from typing import Any

from searcherkit.llm.base import ClientConfig
from searcherkit.llm.openai_client import OpenAIClient


class VllmClient(OpenAIClient):
    def __init__(self, *, config: ClientConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            openai_config = ClientConfig(
                type="openai",
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url or "http://127.0.0.1:8000/v1",
                concurrency_limit=config.concurrency_limit,
                extra_client_kwargs=config.extra_client_kwargs,
                retry_policy=config.retry_policy,
                default_kwargs=config.default_kwargs,
            )
            super().__init__(config=openai_config)
            return
        super().__init__(**kwargs)
