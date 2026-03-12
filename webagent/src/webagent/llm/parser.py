"""Chat Template Parser

TODO:
- [ ] Support ASearcher Format
- [ ] Support OpenAI Format
- [x] Support Qwen3 Format
- [x] Support upstream parsing (tool calls / thinking parsed by LLM engine, should not be parsed, but accessed through `message.tool_call` etc.)
"""
from __future__ import annotations

import re
import abc
import json
from typing import Any, Iterable, Mapping

from webagent.llm.chat_types import ChatMessage
from webagent.llm.chat_types import ToolCall
from webagent.llm.chat_types import assistant, system, user
from webagent.log import get_logger
from webagent.utils.mapping import get_first_or_default, get_or_default

logger = get_logger(__name__)

class ParsingError(Exception):
    """Raised when model message payload cannot be parsed into `ChatMessage`."""

class Parser:
    @abc.abstractmethod
    def from_model(self, messages: Iterable[dict[str, Any]]) -> Iterable[ChatMessage]:
        pass

    @abc.abstractmethod
    def to_model(self, messages: Iterable[ChatMessage]) -> Iterable[dict[str, Any]]:
        pass

# def to_gpt(
#     messages: Iterable[ChatMessage],
#     *,
#     tools: Sequence[Mapping[str, Any]] | None = None,
# ) -> dict[str, Any]:
#     """
#     Convert internal ChatMessage objects into structured GPT message JSON.

#     Intended for Harmony/OpenAI libraries that accept message JSON and perform
#     model-specific rendering internally.
#     """

#     out: list[dict[str, Any]] = []
#     for message in messages:
#         item: dict[str, Any] = {"role": message.role}
#         content = message.content
#         item["content"] = content

#         if message.name:
#             item["name"] = message.name
#         if message.tool_call_id:
#             item["tool_call_id"] = message.tool_call_id

#         if message.role == "assistant":
#             thinking = message.extensions.get("thinking")
#             if isinstance(thinking, str) and thinking.strip():
#                 item["reasoning"] = thinking

#             if message.tool_calls:
#                 item["tool_calls"] = [
#                     {
#                         "id": tc.id,
#                         "type": "function",
#                         "function": {
#                             "name": tc.name,
#                             "arguments": json.dumps(_parse_arguments(tc.arguments), ensure_ascii=False)
#                             if not isinstance(tc.arguments, str)
#                             else tc.arguments,
#                         },
#                     }
#                     for tc in message.tool_calls
#                 ]

#         if message.extensions.get("openai") and isinstance(message.extensions["openai"], Mapping):
#             item.update(dict(message.extensions["openai"]))

#         out.append(item)

#     payload: dict[str, Any] = {"messages": out}
#     if tools:
#         payload["tools"] = list(tools)
#     return payload

class QwenParser(Parser):
    """Parse `ChatMessage` to Qwen format in the form of OpenAI SDK message types
    """
    def __init__(self, upstream_parsed = False, drop_thinking = True) -> None:
        """_
        Args:
            upstream_parsed (bool, optional): Whether the `thinking` and `tool_call` content are already been parsed. Parser would attempt to read from and write to `reasoning`,`reasoning_content` and `tool_calls` fields of the message dict if set to `True`. Defaults to False.
            drop_thinking (bool, optional): Whether to format `thinking` content into model context (ineffective when using `upstream_parsing`). Defaults to True. 
        """        
        super().__init__()
        self.upstream_parsed = upstream_parsed
        self.drop_thinking = drop_thinking
    
    def to_model(
        self,
        messages: Iterable[ChatMessage],
    ) -> list[dict[str, Any]]:
        """Convert internal `ChatMessage` objects into Qwen/OpenAI-style message dicts.

        Args:
            messages: Iterable of internal chat messages.
            drop_thinking: Whether to omit assistant thinking/reasoning content.
            availiable_tools: Optional tools used to build a system tool-spec message.

        Raises:
            ValueError: If a message has an unsupported role.
        """
        out: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                content = message.content or ""
                if self.upstream_parsed:
                    item: dict[str, Any] = {"role": "system", "content": content}
                    if message.tools:
                        item["tools"] = self.as_openai_tools(message.tools)
                    out.append(item)
                else:
                    out.append({"role": "system", "content": content + (self.qwen_tools_block(message.tools) if message.tools else "")})
            elif message.role == "user":
                out.append({"role": "user", "content": message.content or ""})
            elif message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant"}
                if self.upstream_parsed:
                    item["content"] = message.content
                    if isinstance(message.thinking, str):
                        # ignore drop_thinking flag, let upstream decide
                        item["reasoning"] = message.thinking
                        item["reasoning_content"] = message.thinking
                    if message.tool_calls:
                        item["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]
                else:
                    parts: list[str] = []
                    if not self.drop_thinking and isinstance(message.thinking, str):
                        parts.append(f"<think>{message.thinking}</think>")
                    if message.content is not None:
                        parts.append(message.content)
                    if message.tool_calls:
                        parts.append(self.render_tool_calls(message.tool_calls))
                    item["content"] = "".join(parts)
                out.append(item)
            elif message.role == "tool":
                out.append({"role": "user", "content":"\n".join([f"<tool_response>{resp}</tool_response>" for resp in message.tool_responses])})
            else:
                raise ValueError(f"Invalid ChatMessage role: {message.role}")

        return out

    def from_model(self, messages: Iterable[dict[str, Any]]) -> list[ChatMessage]:
        """Parse Qwen/OpenAI-style message dicts into internal `ChatMessage` objects.

        Args:
            messages: Iterable of model message dicts with a `role` field.

        Raises:
            ValueError: If a message role is not supported or assistant content format is invalid.
            ParsingError: If a `<tool_call>` JSON payload is malformed or misses required fields.
        """
        out = []
        for message in messages:
            role = message.get("role")
            if role == "user":
                out.append(self.from_user(message))
            elif role == "assistant":
                out.append(self.from_assistant(message))
            elif role == "system":
                out.append(self.from_system(message))
            else:
                raise ValueError(f"Invalid Qwen3 message: {message}")
        return out
                
    def from_system(self, message: dict[str,Any]) -> ChatMessage:
        return system(message.get("content") or "")
        
    def from_assistant(self, message: dict[str,Any]) -> ChatMessage:
        if self.upstream_parsed:
            thinking = get_first_or_default(message, "reasoning", "reasoning_content")
            tool_calls: list[ToolCall] = []
            for tc in get_or_default(message, "tool_calls", []):
                if not isinstance(tc, Mapping):
                    continue
                function = get_or_default(tc, "function", {})
                if not isinstance(function, Mapping):
                    function = {}
                arguments = get_or_default(function, "arguments", {})
                if not isinstance(arguments, Mapping):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=str(get_or_default(tc, "id", "call_tool")),
                        name=str(get_or_default(function, "name", "")),
                        arguments=arguments,
                    )
                )
            return assistant(
                get_or_default(message, "content"),
                thinking=thinking if isinstance(thinking, str) else None,
                tool_calls=tool_calls,
            )

        content = get_or_default(message, "content", "")
        if not isinstance(content, str):
            raise ValueError(f"Invalid Qwen assistant message content: {content!r}")

        message_pattern = re.compile(
            r"^(?:<think>(?P<thinking>.*?)</think>)?"
            r"(?P<out>(?:(?!<think>|</think>|<tool_call>|</tool_call>).)*)"
            r"(?P<tool_calls>(?:<tool_call>.*?</tool_call>)*)$",
            re.DOTALL,
        )
        tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        parsed = message_pattern.fullmatch(content)
        if not parsed:
            raise ParsingError(f"Invalid Qwen assistant message format: {content!r}")

        thinking = parsed.group("thinking")
        out_raw = parsed.group("out")
        out_content = out_raw if out_raw != "" else None
        tool_calls_raw = parsed.group("tool_calls")

        tool_calls: list[ToolCall] = []
        cnt = 0
        for payload_raw in re.findall(tool_call_pattern, tool_calls_raw):
            # Notice: Keep tool call order
            try:
                tc = json.loads(payload_raw)
                tool_calls.append(
                    ToolCall(
                        id="chatcmpl-tool-"+str(cnt),
                        name=tc["name"],
                        arguments=tc["arguments"],
                    )
                )
                cnt += 1
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ParsingError(f"Invalid <tool_call> payload: {payload_raw!r}") from exc

        return assistant(
            out_content,
            thinking=thinking,
            tool_calls=tool_calls,
        )
    
    def from_user(self, message: dict[str, Any]) -> ChatMessage:
        return user(get_or_default(message, "content", ""))
            
    def render_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        lines: list[str] = []
        for tc in tool_calls:
            payload = {
                "name": tc.name,
                "arguments": tc.arguments,
            }
            lines.append("<tool_call>")
            lines.append(json.dumps(payload, ensure_ascii=False))
            lines.append("</tool_call>")
        return "\n".join(lines)


    def qwen_tools_block(self, tools: Iterable[Mapping[str, Any]]) -> str:
        lines = [
            "# Tools",
            "",
            "You may call one or more functions to assist with the user query.",
            "",
            "You are provided with function signatures within <tools></tools> XML tags:",
            "<tools>",
        ]
        for tool in self.as_openai_tools(tools):
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

    def as_openai_tools(self, tools: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            if "type" in tool and "function" in tool:
                out.append(dict(tool))
                continue
            name = get_or_default(tool, "name", "")
            description = get_or_default(tool, "description", "")
            parameters = tool.get("arguments_schema", tool.get("parameters", {})) or {}
            if not description:
                logger.warning("Tool %s has no description", name)
            if not parameters:
                logger.warning("Tool %s has no arguments schema", name)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return out
        
def get_parser_cls(name: str):
    if "qwen" in name.lower():
        return QwenParser
    raise ValueError(f"Cannot infer parser type from name: {name}")

__all__ = [ "ParsingError", "QwenParser", "get_parser_cls" ]
