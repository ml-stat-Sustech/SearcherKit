"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterable, overload

from openai import AsyncOpenAI

from searchagent.common.log import get_logger
from searchagent.common.retry import RetryPolicy, wrap_async
from searchagent.llm.base import Client, ClientConfig, LLMStreamChunk

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
            self.__init__(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                retry_policy=config.retry_policy,
                concurrency_limit=config.concurrency_limit,
                default_kwargs=config.default_kwargs,
                **(config.extra_client_kwargs or {}),
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

    async def stream_complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        payload.pop("stream", None)
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, Mapping):
            stream_options = dict(stream_options)
        else:
            stream_options = {}
        stream_options.setdefault("include_usage", True)
        payload["stream_options"] = stream_options
        logger.debug(
            "Submitting streaming LLM completion provider=openai model=%s messages=%s",
            self.model,
            len(message_list),
        )
        stream = await self._create_completion(message_list, {**payload, "stream": True}, session_id)
        role = "assistant"
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        usage: Any | None = None

        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            delta_payload = delta.model_dump(exclude_none=True) if hasattr(delta, "model_dump") else {}
            role = str(delta_payload.get("role") or role)
            content_delta = _stream_text_delta(delta_payload, "content")
            thinking_delta = _stream_text_delta(delta_payload, "reasoning_content", "reasoning")
            if content_delta:
                content_parts.append(content_delta)
            if thinking_delta:
                thinking_parts.append(thinking_delta)
            _merge_tool_call_deltas(tool_call_parts, delta_payload.get("tool_calls") or [])
            if content_delta or thinking_delta or delta_payload.get("tool_calls"):
                yield LLMStreamChunk(
                    content_delta=content_delta,
                    thinking_delta=thinking_delta,
                    raw_delta=delta_payload,
                )

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts) or None,
        }
        thinking = "".join(thinking_parts)
        if thinking:
            message["reasoning"] = thinking
            message["reasoning_content"] = thinking
        tool_calls = _finalize_tool_calls(tool_call_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        logger.debug(
            "Streaming LLM completion finished provider=openai model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )
        yield LLMStreamChunk(message=message, usage=usage, done=True)


def _stream_text_delta(delta: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = delta.get(key)
        if isinstance(value, str):
            return value
    return ""


def _merge_tool_call_deltas(
    tool_call_parts: dict[int, dict[str, Any]],
    deltas: list[Any],
) -> None:
    """Accumulate tool-call fragments from one streaming chunk.

    A single chunk may contain deltas for multiple tool calls, and each tool
    call is split across many chunks. We group fragments by their ``index``
    and append partial ``id``, ``type``, ``name`` and ``arguments`` values.
    """
    for fallback_index, delta in enumerate(deltas):
        # Skip malformed deltas; the rest of the stream may still be usable.
        if not isinstance(delta, dict):
            continue

        # The index tells us which tool call this fragment belongs to.
        # Fall back to the position in the chunk list if it is missing or invalid.
        raw_index = delta.get("index", fallback_index)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index

        # Create a skeleton for this tool call the first time we see its index.
        target = tool_call_parts.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )

        tool_call_id = delta.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            target["id"] = tool_call_id
        tool_call_type = delta.get("type")
        if isinstance(tool_call_type, str) and tool_call_type:
            target["type"] = tool_call_type

        # Tool name and arguments are nested under the ``function`` field.
        function_delta = delta.get("function")
        if not isinstance(function_delta, dict):
            continue
        function = target.setdefault("function", {"name": "", "arguments": ""})

        # Append streamed fragments for the tool name and its JSON arguments.
        name_delta = function_delta.get("name")
        if isinstance(name_delta, str):
            function["name"] = str(function.get("name") or "") + name_delta
        arguments_delta = function_delta.get("arguments")
        if isinstance(arguments_delta, str):
            function["arguments"] = str(function.get("arguments") or "") + arguments_delta


def _finalize_tool_calls(tool_call_parts: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert accumulated tool-call parts into a clean, ordered list.

    By the time this is called the stream has finished, so each entry in
    ``tool_call_parts`` is fully assembled. We only need to sort by index,
    convert the dict to a list, and provide safe defaults for missing fields.
    """
    tool_calls: list[dict[str, Any]] = []
    for index in sorted(tool_call_parts):
        item = tool_call_parts[index]
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        tool_calls.append(
            {
                # Generate a deterministic id if the provider never sent one.
                "id": str(item.get("id") or f"call_{index}"),
                "type": str(item.get("type") or "function"),
                "function": {
                    "name": str(function.get("name") or ""),
                    # Empty arguments would break downstream JSON parsing, so default to {}.
                    "arguments": str(function.get("arguments") or "{}"),
                },
            }
        )
    return tool_calls
