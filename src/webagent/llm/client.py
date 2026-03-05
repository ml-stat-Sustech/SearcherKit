"""Warpped LLM client

Controllable LLM call concorrency
Parse from and to `ChatMessage`

TODO:
- [ ] support openai compatible engines
- [ ] support non server LLM engines
"""

import asyncio
import json
from typing import Any, Dict, Iterable, Mapping, Optional

from chat_types import ChatMessage, TextPart, ToolCall


NUM_CONCURRENT_QUERIES = 32

sem = asyncio.Semaphore(NUM_CONCURRENT_QUERIES)


def _part_to_openai(part: TextPart) -> dict[str, Any]:
    return {"type": "text", "text": part.text}


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, Any]:
    arguments = tool_call.arguments
    serialized_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": serialized_arguments,
        },
    }


def _to_openai_messages(messages: Iterable[ChatMessage | Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(messages)
    if not materialized:
        return []
    if isinstance(materialized[0], ChatMessage):
        out: list[dict[str, Any]] = []
        for message in materialized:
            item: dict[str, Any] = {"role": message.role}
            if message.content is None:
                item["content"] = None
            elif isinstance(message.content, str):
                item["content"] = message.content
            else:
                item["content"] = [_part_to_openai(part) for part in message.content]
            if message.name:
                item["name"] = message.name
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [_tool_call_to_openai(tc) for tc in message.tool_calls]
            if message.extensions.get("openai") and isinstance(message.extensions["openai"], Mapping):
                item.update(dict(message.extensions["openai"]))
            out.append(item)
        return out
    return [dict(message) for message in materialized]  # type: ignore[misc]


class OpenAIChatClient:
    """Wrapper around the `openai` Python SDK."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_kwargs: Dict[str, object] = {},
    ) -> None:
        from openai import AsyncOpenAI, RateLimitError
        import backoff
        self.client = AsyncOpenAI(
            base_url = base_url,
            api_key = api_key
        )
        self.model = model
        self.query = backoff.on_exception(backoff.expo, RateLimitError)(self.client.chat.completions.create)
        
        self.default_kwargs = default_kwargs

    async def complete(self, messages: Iterable[ChatMessage | Mapping[str, Any]], **kwargs) -> ChatMessage:
        openai_messages = _to_openai_messages(messages)
        payload = {**self.default_kwargs, **kwargs}
        async with sem:
            response = await self.query(
                model=self.model,
                messages=openai_messages,
                **payload,
            )
        msg = response.choices[0].message
        # TODO: parse to ChatMessages
    
    @classmethod
    def from_conf(cls, conf):
        pass
        
