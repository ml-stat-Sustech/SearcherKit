from __future__ import annotations

from collections.abc import Awaitable, Iterable
from typing import Any, Callable

from openai.types.completion_usage import CompletionUsage

from areal.experimental.openai import ArealOpenAI

from searcherkit.llm.base import Client
from searcherkit.common.log import get_logger

logger = get_logger(__name__)


class ARealClient(Client):
    """Wrapper around the AReaL OpenAI-compatible client."""

    def __init__(
        self,
        client: ArealOpenAI,
        *,
        default_kwargs: dict[str, object] | None = None,
        **extra_client_kwargs: Any,
    ) -> None:
        self.client = client
        self.default_kwargs = default_kwargs or {}
        self._create_completion: Callable[
            [list[dict[str, Any]], dict[str, Any]],
            Awaitable[Any],
        ] = self._create_completion_no_retry

    async def complete(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return (await self.complete_with_usage(messages, **kwargs))[0]

    async def _create_completion_no_retry(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> Any:
        logger.debug(
            "Submitting LLM completion model=%s messages=%s",
            "AReaL Loaded Model",
            len(messages),
        )
        return await self.client.chat.completions.create(
            model="default",
            messages=messages,
            **payload,
        )

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], CompletionUsage | None]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        resp = await self._create_completion(message_list, payload)
        logger.debug(
            "LLM completion finished model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            "AReaL Loaded Model",
            getattr(resp.usage, "prompt_tokens", None),
            getattr(resp.usage, "completion_tokens", None),
            getattr(resp.usage, "total_tokens", None),
        )
        return resp.choices[0].message.model_dump(), resp.usage
