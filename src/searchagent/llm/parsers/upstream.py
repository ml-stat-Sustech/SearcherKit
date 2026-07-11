"""Provider-native structured message parser."""

from __future__ import annotations

import json
from typing import Any, Iterable

from searchagent.common.messages import ChatMessage, ToolCall, assistant, system, user
from searchagent.common.utils import get_first_or_default, get_or_default
from searchagent.llm.parsers.base import Parser, ParsingError


class UpstreamParser(Parser):
    """Parse provider-native messages with structured reasoning and tool calls."""

    @property
    def uses_provider_tools(self) -> bool:
        return True

    def to_model(self, messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                out.append({"role": "system", "content": message.content or ""})
            elif message.role == "user":
                out.append({"role": "user", "content": message.content or ""})
            elif message.role == "assistant":
                out.append(self._assistant_to_model(message))
            elif message.role == "tool":
                for tool_call_id, response in message.tool_responses.items():
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": response,
                        }
                    )
            else:
                raise ValueError(f"Invalid ChatMessage role: {message.role}")

        return out

    def _assistant_to_model(self, message: ChatMessage) -> dict[str, Any]:
        item: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if isinstance(message.thinking, str):
            item["reasoning"] = message.thinking
            item["reasoning_content"] = message.thinking
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in message.tool_calls
            ]
        return item

    def from_model(self, messages: Iterable[dict[str, Any]]) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                out.append(self.from_user(message))
            elif role == "assistant":
                out.append(self.from_assistant(message))
            elif role == "system":
                out.append(self.from_system(message))
            else:
                raise ValueError(f"Invalid upstream message: {message}")
        return out

    def from_system(self, message: dict[str, Any]) -> ChatMessage:
        return system(message.get("content") or "")

    def from_user(self, message: dict[str, Any]) -> ChatMessage:
        return user(get_or_default(message, "content", ""))

    def from_assistant(self, message: dict[str, Any]) -> ChatMessage:
        thinking = get_first_or_default(message, "reasoning", "reasoning_content")
        tool_calls: list[ToolCall] = []
        for tool_call in get_or_default(message, "tool_calls", []):
            try:
                function = tool_call["function"]
                arguments_raw = function["arguments"]
            except (KeyError, TypeError) as exc:
                raise ParsingError(f"Invalid upstream tool_call payload: {tool_call!r}") from exc

            try:
                arguments = json.loads(arguments_raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ParsingError(
                    f"Invalid JSON payload for upstream tool_call arguments: {arguments_raw!r}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ParsingError(
                    f"Invalid upstream tool_call arguments object: {arguments_raw!r}"
                )

            tool_calls.append(
                ToolCall(
                    id=str(get_or_default(tool_call, "id", "call_tool")),
                    name=str(get_or_default(function, "name", "")),
                    arguments=arguments,
                )
            )
        return assistant(
            get_or_default(message, "content"),
            thinking=thinking if isinstance(thinking, str) else None,
            tool_calls=tool_calls,
        )
