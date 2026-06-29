"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import asyncio
import random
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, overload

from openai import AsyncOpenAI

from searchagent.common.retry import RetryPolicy, wrap_async
from searchagent.llm.base import Client, ClientConfig
from searchagent.common.log import get_logger

if TYPE_CHECKING:
    from openai.types.completion_usage import CompletionUsage

logger = get_logger(__name__)


class OpenAIClient(Client):
    """Wrapper around the OpenAI-compatible chat completions API."""

    @overload
    def __init__(self, *, config: ClientConfig) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        concurrency_limit: int | None = None,
        default_kwargs: dict[str, Any] | None = None,
        **extra_client_kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        config: ClientConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | list[str] | None = None,
        retry_policy: RetryPolicy | None = None,
        concurrency_limit: int | None = None,
        default_kwargs: dict[str, Any] | None = None,
        **extra_client_kwargs: Any,
    ) -> None:
        if config is not None:
            if config.openai is None:
                raise ValueError("Please specify openai config when using openai client")
            self.__init__(
                model=config.model,
                api_key=config.openai.api_key,
                base_url=config.openai.base_url,
                retry_policy=config.retry_policy,
                concurrency_limit=config.openai.concurrency_limit,
                default_kwargs=config.default_kwargs,
                **(config.openai.extra_client_kwargs or {}),
            )
            return

        if isinstance(base_url, str) or base_url is None:
            base_urls = [base_url]
        else:
            base_urls = base_url

        if not model:
            raise ValueError("model must be provided")

        self.clients = [
            AsyncOpenAI(
                base_url=url,
                api_key=api_key,
                **extra_client_kwargs,
            )
            for url in base_urls
        ]
        self.model = model
        self.default_kwargs = default_kwargs or {}
        self.llm_concurrency_lock = asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()
        self._create_completion: Callable[
            [list[dict[str, Any]], dict[str, Any], int | None],
            Awaitable[Any],
        ] = self._create_completion_no_retry
        if retry_policy is not None:
            self._create_completion = wrap_async(
                self._create_completion_no_retry,
                policy=retry_policy,
                op_name="openai.chat.completions.create",
            )

    async def complete(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return (await self.complete_with_usage(messages, session_id, **kwargs))[0]

    async def _create_completion_no_retry(
        self,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
        session_id: int | None = None,
    ) -> Any:
        logger.debug(
            "Submitting LLM completion provider=openai model=%s messages=%s",
            self.model,
            len(messages),
        )
        if session_id is None:
            session_id = random.randint(0, len(self.clients) - 1)
        async with self.llm_concurrency_lock:
            return await self.clients[session_id % len(self.clients)].chat.completions.create(
                model=self.model,
                messages=messages,
                **payload,
            )

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], "CompletionUsage | None"]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        resp = await self._create_completion(message_list, payload, session_id)
        logger.debug(
            "LLM completion finished provider=openai model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            getattr(resp.usage, "prompt_tokens", None),
            getattr(resp.usage, "completion_tokens", None),
            getattr(resp.usage, "total_tokens", None),
        )
        return resp.choices[0].message.model_dump(), resp.usage
