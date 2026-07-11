"""WebSailor prompt-format parser."""

from __future__ import annotations

import json
from typing import Any, Iterable

from searchagent.common.messages import ChatMessage
from searchagent.common.messages import Tool as ToolMsgType
from searchagent.llm.parsers.qwen import QwenParser


class WebSailorParser(QwenParser):
    """Qwen-compatible parser with WebSailor-style prompt assembly."""

    def __init__(
        self,
        *,
        first_user_prefix: str | None = None,
        user_role_prefix: str = "User: ",
    ) -> None:
        super().__init__(drop_thinking=False)
        self.first_user_prefix = first_user_prefix or self.default_first_user_prefix
        self.user_role_prefix = user_role_prefix

    @property
    def default_first_user_prefix(self) -> str:
        return (
            "A conversation between User and Assistant. The user asks a question, "
            "and the assistant solves it by calling one or more of the following tools.\n"
            "{tools_block}\n\n"
            "The assistant starts with one or more cycles of (thinking about which tool to use "
            "-> performing tool call -> waiting for tool response), and ends with "
            "(thinking about the answer -> answer of the question). The thinking processes, "
            "tool calls, tool responses, and answer are enclosed within their tags. "
            "There could be multiple thinking processes, tool calls, tool call parameters "
            "and tool response parameters.\n\n"
            "Example response:\n"
            "<think> thinking process here </think>\n"
            "<tool_call>\n"
            '{{"name": "tool name here", "arguments": {{"parameter name here": "parameter value here", "another parameter name here": "another parameter value here"}}}}\n'
            "</tool_call>\n"
            "<tool_response>\n"
            "tool_response here\n"
            "</tool_response>\n"
            "<think> thinking process here </think>\n"
            "<answer> answer here </answer>\n\n"
        )

    def to_model(self, messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
        message_list = list(messages)
        out: list[dict[str, Any]] = []

        available_tools: list[ToolMsgType] | None = None
        for message in message_list:
            if message.role == "system" and message.tools:
                available_tools = message.tools
                break

        first_user_seen = False
        for message in message_list:
            if message.role == "system":
                out.append({"role": "system", "content": message.content or ""})
            elif message.role == "user":
                content = message.content or ""
                if not first_user_seen:
                    first_user_seen = True
                    content = self.render_first_user_prompt(content, available_tools)
                out.append({"role": "user", "content": content})
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

    def render_first_user_prompt(
        self,
        query: str,
        tools: Iterable[ToolMsgType] | None,
    ) -> str:
        tools_block = self.render_tools_block(tools or [])
        prefix = self.first_user_prefix.format(tools_block=tools_block)
        return prefix + self.user_role_prefix + query

    def render_tools_block(self, tools: Iterable[ToolMsgType]) -> str:
        rendered_tools: list[str] = []
        for tool in tools:
            rendered_tools.append(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        inner = ",\n".join(rendered_tools) if rendered_tools else ""
        return "<tools>\n" + inner + "\n</tools>"
