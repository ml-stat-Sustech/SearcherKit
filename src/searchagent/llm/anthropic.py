"""Anthropic Messages API client adapter."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from typing import Any, Awaitable, Callable, Iterable, overload

from searchagent.common.retry import RetryPolicy, wrap_async
from searchagent.llm.base import Client, ClientConfig
from searchagent.log import get_logger

logger = get_logger(__name__)

_ANTHROPIC_INSTALL_MSG = (
    "The 'anthropic' package is required to use AnthropicClient. "
    "Install it with: pip install searchagent[anthropic]"
)


def _extract_system(messages: Iterable[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                parts.append(content)
    return "\n\n".join(parts) if parts else None


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function", tool)
        converted.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get(
                    "parameters",
                    func.get("input_schema", {"type": "object", "properties": {}}),
                ),
            }
        )
    return converted


def _convert_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages to Anthropic content-block format."""
    msg_list = list(messages)
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(msg_list):
        msg = msg_list[i]
        role = msg.get("role", "")

        if role == "system":
            i += 1
            continue

        if role == "user":
            content = [{"type": "text", "text": msg.get("content") or ""}]
            result.append({"role": "user", "content": content})
            i += 1

        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                else:
                    arguments = raw_args
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": func.get("name", ""),
                        "input": arguments,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            result.append({"role": "assistant", "content": blocks})
            i += 1

        elif role == "tool":
            tool_result_blocks: list[dict[str, Any]] = []
            while i < len(msg_list) and msg_list[i].get("role") == "tool":
                tool_msg = msg_list[i]
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_msg.get("tool_call_id", ""),
                        "content": tool_msg.get("content", ""),
                    }
                )
                i += 1
            if tool_result_blocks:
                result.append({"role": "user", "content": tool_result_blocks})

        else:
            i += 1

    # Enforce alternation: merge consecutive same-role messages
    merged: list[dict[str, Any]] = []
    for msg in result:
        if merged and msg["role"] == merged[-1]["role"] == "user":
            merged[-1]["content"].extend(msg["content"])
        elif merged and msg["role"] == merged[-1]["role"] == "assistant":
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append(msg)

    # Anthropic requires the first message to be user
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"type": "text", "text": ""}]})

    return merged


def _convert_response(response: Any) -> dict[str, Any]:
    """Convert Anthropic response to OpenAI-format message dict."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in response.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    result: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(content_parts) if content_parts else "",
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    if hasattr(response, "stop_reason") and response.stop_reason is not None:
        result["stop_reason"] = response.stop_reason

    return result


def _convert_usage(usage: Any) -> dict[str, Any]:
    """Convert Anthropic Usage to OpenAI-compatible usage dict."""
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class AnthropicClient(Client):
    """Anthropic Messages API client adapter."""

    @overload
    def __init__(self, *, config: ClientConfig) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
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
        base_url: str | None = None,
        retry_policy: RetryPolicy | None = None,
        concurrency_limit: int | None = None,
        default_kwargs: dict[str, Any] | None = None,
        **extra_client_kwargs: Any,
    ) -> None:
        if config is not None:
            if config.anthropic is None:
                raise ValueError(
                    "Please specify anthropic config when using anthropic client"
                )
            self.__init__(
                model=config.model,
                api_key=config.anthropic.api_key,
                base_url=config.anthropic.base_url,
                retry_policy=config.retry_policy,
                concurrency_limit=config.anthropic.concurrency_limit,
                default_kwargs=config.default_kwargs,
                **(config.anthropic.extra_client_kwargs or {}),
            )
            return

        if not model:
            raise ValueError("model must be provided")

        client_kwargs: dict[str, Any] = extra_client_kwargs
        if api_key is not None:
            client_kwargs.setdefault("api_key", api_key)
        if base_url is not None:
            client_kwargs.setdefault("base_url", base_url)

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(_ANTHROPIC_INSTALL_MSG) from None
        self.client = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.default_kwargs = default_kwargs or {}
        self.llm_concurrency_lock = (
            asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()
        )
        self._create_completion: Callable[
            [list[dict[str, Any]], dict[str, Any]], Awaitable[Any]
        ] = self._create_completion_no_retry
        if retry_policy is not None:
            self._create_completion = wrap_async(
                self._create_completion_no_retry,
                policy=retry_policy,
                op_name="anthropic.messages.create",
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
    ) -> Any:
        logger.debug(
            "Submitting LLM completion provider=anthropic model=%s messages=%s",
            self.model,
            len(messages),
        )
        system = _extract_system(messages)
        anthropic_messages = _convert_messages(messages)
        tools = payload.pop("tools", None)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            **payload,
        }
        if system:
            create_kwargs["system"] = system
        if tools:
            create_kwargs["tools"] = _convert_tools(tools)

        async with self.llm_concurrency_lock:
            return await self.client.messages.create(**create_kwargs)

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        resp = await self._create_completion(message_list, payload)
        usage = _convert_usage(resp.usage)
        logger.debug(
            "LLM completion finished provider=anthropic model=%s "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            self.model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        return _convert_response(resp), usage
