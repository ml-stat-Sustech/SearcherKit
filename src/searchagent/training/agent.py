from __future__ import annotations

import json
from collections.abc import Iterable

from searchagent.agent import SearchAgent
from searchagent.agent.search_agent import LLMOutputError
from searchagent.common.messages import ToolCall


class TooManyToolCallsError(LLMOutputError):
    """Raised when the model issues too many parallel tool calls."""


class RepeatedToolCallError(LLMOutputError):
    """Raised when the model repeats a tool call with identical arguments."""


class SearchAgentTraining(SearchAgent):
    def __init__(self, raise_repeat_tool_call: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.raise_repeat_tool_call = raise_repeat_tool_call
        self.previous_tool_queries: set[tuple[str, str]] = set()

    def reset(self):
        super().reset()
        self.previous_tool_queries = set()

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[str]:
        tool_calls_list = list(tool_calls)
        if len(tool_calls_list) > 1:
            raise TooManyToolCallsError("Too many parallel tool calls")
        for tc in tool_calls_list:
            if tc.name not in self.tool_dict:
                raise LLMOutputError(f"Tool {tc.name} not found")
            argument_str = json.dumps(tc.arguments, sort_keys=True)
            if (tc.name, argument_str) in self.previous_tool_queries:
                if self.raise_repeat_tool_call:
                    raise RepeatedToolCallError(
                        f"Query {tc.name} has repeated arguments {argument_str}"
                    )
            else:
                self.previous_tool_queries.add((tc.name, argument_str))
        return await super().call_tools(tool_calls_list)
