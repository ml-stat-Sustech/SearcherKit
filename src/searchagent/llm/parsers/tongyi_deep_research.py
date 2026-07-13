"""Tongyi Deep Research chat-template parser."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, overload

from searchagent.common.messages import ChatMessage, ToolCall, assistant, system, user
from searchagent.common.messages import Tool as ToolType
from searchagent.common.log import get_logger
from searchagent.common.utils import get_or_default
from searchagent.llm.parsers.base import LiveDeltaPart, LiveDeltaSplitter, Parser, ParserConfig, ParsingError
from searchagent.tools import to_openai_tool

logger = get_logger(__name__)

class TongyiDeepRessearchLiveDeltaSplitter:
    """Interpret Tongyi-DeepResearch-style chat-template tags in raw streaming content.

    This splitter is for transient live display only. The complete assistant
    message is still parsed by :meth:`QwenParser.from_model` for agent semantics.
    """

    _TAGS = (
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "content"

    def feed(self, text: str) -> list[LiveDeltaPart]:
        self._buffer += text
        out: list[LiveDeltaPart] = []
        while self._buffer:
            tag_start = self._buffer.find("<")
            if tag_start > 0:
                self._emit(out, self._buffer[:tag_start])
                self._buffer = self._buffer[tag_start:]
                continue
            if tag_start == -1:
                self._emit(out, self._buffer)
                self._buffer = ""
                break

            matched = False
            for tag in self._TAGS:
                if self._buffer.startswith(tag):
                    self._apply_tag(tag)
                    self._buffer = self._buffer[len(tag):]
                    matched = True
                    break
            if matched:
                continue

            if any(tag.startswith(self._buffer) for tag in self._TAGS):
                break

            self._emit(out, "<")
            self._buffer = self._buffer[1:]
        return out

    def flush(self) -> list[LiveDeltaPart]:
        if not self._buffer:
            return []
        out: list[LiveDeltaPart] = []
        self._emit(out, self._buffer)
        self._buffer = ""
        return out

    def _apply_tag(self, tag: str) -> None:
        if tag == "<think>":
            self._mode = "thinking"
        elif tag == "</think>":
            self._mode = "content"
        elif tag == "<tool_call>":
            self._mode = "suppressed"
        elif tag == "</tool_call>":
            self._mode = "content"

    def _emit(self, out: list[LiveDeltaPart], text: str) -> None:
        if not text or self._mode == "suppressed":
            return
        field = "thinking" if self._mode == "thinking" else "content"
        if out and out[-1].field == field:
            out[-1] = LiveDeltaPart(field=field, text=out[-1].text + text)
            return
        out.append(LiveDeltaPart(field=field, text=text))


class TongyiDeepResearchParser(Parser):
    """Parse `ChatMessage` objects to and from Tongyi Deep Research messages."""
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, config: ParserConfig | Mapping[str, Any]) -> None: ...

    def __init__(self, *, config: ParserConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()

    def create_live_delta_splitter(self) -> LiveDeltaSplitter:
        return TongyiDeepRessearchLiveDeltaSplitter()

    def to_model(self, messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                content = message.content or ""
                tools_block = self.qwen_tools_block(message.tools) if message.tools else ""
                out.append({"role": "system", "content": content + tools_block})
            elif message.role == "user":
                out.append({"role": "user", "content": message.content or ""})
            elif message.role == "assistant":
                out.append(self._assistant_to_model(message))
            elif message.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": "\n".join(
                            f"<tool_response>{response}</tool_response>"
                            for response in message.tool_responses.values()
                        ),
                    }
                )
            else:
                raise ValueError(f"Invalid ChatMessage role: {message.role}")

        return out

    def _assistant_to_model(self, message: ChatMessage) -> dict[str, Any]:
        item: dict[str, Any] = {"role": "assistant"}
        parts: list[str] = []
        if isinstance(message.thinking, str):
            parts.append(f"<think>{message.thinking}</think>")
        if message.content is not None:
            parts.append(message.content)
        if message.tool_calls:
            parts.append(self.render_tool_calls(message.tool_calls))
        item["content"] = "".join(parts)
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
                raise ValueError(f"Invalid Qwen message: {message}")
        return out

    def from_system(self, message: dict[str, Any]) -> ChatMessage:
        return system(message.get("content") or "")

    def from_assistant(self, message: dict[str, Any]) -> ChatMessage:
        content = get_or_default(message, "content", "")
        if not isinstance(content, str):
            raise ValueError(f"Invalid Qwen assistant message content: {content!r}")

        message_pattern = re.compile(
            r"^(?:<think>(?P<thinking>.*?)</think>|</think>)?"
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
        for index, payload_raw in enumerate(re.findall(tool_call_pattern, tool_calls_raw)):
            try:
                payload = json.loads(payload_raw.strip())
                tool_calls.append(
                    ToolCall(
                        id=f"chatcmpl-tool-{index}",
                        name=payload["name"],
                        arguments=payload["arguments"],
                    )
                )
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
        for tool_call in tool_calls:
            payload = {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            lines.append("<tool_call>")
            lines.append(json.dumps(payload, ensure_ascii=False))
            lines.append("</tool_call>")
        return "\n".join(lines)

    def qwen_tools_block(self, tools: Iterable[ToolType]) -> str:
        lines = [
            "# Tools",
            "",
            "You may call one or more functions to assist with the user query.",
            "",
            "You are provided with function signatures within <tools></tools> XML tags:",
            "<tools>",
        ]
        for tool in tools:
            description = tool.description or ""
            parameters = tool.inputSchema or {}
            if not description:
                logger.warning("Tool %s has no description", tool.name)
            if not parameters:
                logger.warning("Tool %s has no arguments schema", tool.name)
            lines.append(
                json.dumps(
                    to_openai_tool(tool.name, tool.description, tool.inputSchema),
                    ensure_ascii=False,
                )
            )

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
