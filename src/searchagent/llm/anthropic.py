"""Anthropic client adapter placeholder."""

from __future__ import annotations

from typing import Any, Iterable

from searchagent.llm.base import Client, ClientConfig


class AnthropicClient(Client):
    def __init__(self, *, config: ClientConfig | None = None, **kwargs: Any) -> None:
        raise NotImplementedError(
            "AnthropicClient is not implemented yet. Use an OpenAI-compatible gateway "
            "or add the anthropic SDK adapter here."
        )

    async def complete(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any | None]:
        raise NotImplementedError
