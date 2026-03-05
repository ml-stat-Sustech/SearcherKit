"""Chat Template Parser

TODO:
- [ ] Support Qwen3 Format
- [ ] Support ASearcher Format
- [ ] Support OpenAI Format
- [ ] Support upstream parsing (tool calls / thinking parsed by LLM engine, should not be parsed, but accessed through `message.tool_call` etc.)
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from chat_types import ChatMessage, TextPart, ToolCall


def _render_content(content: str | list[TextPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def _parse_arguments(arguments: Mapping[str, Any] | str) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _render_tool_calls_raw(tool_calls: list[ToolCall]) -> str:
    lines: list[str] = []
    for tc in tool_calls:
        payload = {
            "name": tc.name,
            "arguments": _parse_arguments(tc.arguments),
        }
        lines.append("<tool_call>")
        lines.append(json.dumps(payload, ensure_ascii=False))
        lines.append("</tool_call>")
    return "\n".join(lines)


def _qwen_tools_block(tools: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for tool in tools:
        lines.append(json.dumps(tool, ensure_ascii=False))
    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
        ]
    )
    return "\n".join(lines)


def to_gpt(
    messages: Iterable[ChatMessage],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Convert internal ChatMessage objects into structured GPT message JSON.

    Intended for Harmony/OpenAI libraries that accept message JSON and perform
    model-specific rendering internally.
    """

    out: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role}
        content = _render_content(message.content)
        item["content"] = content

        if message.name:
            item["name"] = message.name
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id

        if message.role == "assistant":
            thinking = message.extensions.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                item["reasoning"] = thinking

            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(_parse_arguments(tc.arguments), ensure_ascii=False)
                            if not isinstance(tc.arguments, str)
                            else tc.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

        if message.extensions.get("openai") and isinstance(message.extensions["openai"], Mapping):
            item.update(dict(message.extensions["openai"]))

        out.append(item)

    payload: dict[str, Any] = {"messages": out}
    if tools:
        payload["tools"] = list(tools)
    return payload


def to_qwen(
    messages: Iterable[ChatMessage],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """
    Convert internal ChatMessage objects to Qwen-style JSON role/content messages.

    Thinking and tool calls are embedded as raw tags in assistant/user content.
    """

    systems: list[str] = []
    out: list[dict[str, str]] = []

    for message in messages:
        content = _render_content(message.content)

        if message.role == "system":
            if content:
                systems.append(content)
            continue

        if message.role == "tool":
            wrapped = f"<tool_response>\n{content}\n</tool_response>"
            out.append({"role": "user", "content": wrapped})
            continue

        if message.role == "assistant":
            parts: list[str] = []
            thinking = message.extensions.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                parts.append(f"<think>\n{thinking}\n</think>")
            if content:
                parts.append(content)
            if message.tool_calls:
                parts.append(_render_tool_calls_raw(message.tool_calls))
            out.append({"role": "assistant", "content": "\n\n".join(parts).strip()})
            continue

        out.append({"role": "user", "content": content})

    if tools:
        systems.append(_qwen_tools_block(tools))

    if systems:
        out.insert(0, {"role": "system", "content": "\n\n".join(systems).strip()})

    return out


__all__ = ["to_gpt", "to_qwen"]
