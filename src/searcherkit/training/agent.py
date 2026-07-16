from __future__ import annotations

import json
from collections.abc import Iterable
from typing import overload

from searcherkit.agent.search_agent import LLMOutputError, SearchAgent
from searcherkit.common.messages import ToolCall
from searcherkit.common.retry import RetryPolicy
from searcherkit.llm.base import Client, get_client
from searcherkit.llm.parsers import get_parser
from searcherkit.sources import add_source_cfg
from searcherkit.tools import build_tool
from searcherkit.training.config import AgentConfig
from searcherkit.training.rewards import normalize_query


class TooManyToolCallsError(LLMOutputError):
    """Raised when the model issues too many parallel tool calls."""


class RepeatedToolCallError(LLMOutputError):
    """Raised when the model repeats a tool call with identical arguments."""


class SearchAgentTraining(SearchAgent):
    @overload
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm_client: Client | None = None,
    ) -> None: ...

    @overload
    def __init__(self, raise_repeat_tool_call: bool = False, **kwargs) -> None: ...

    def __init__(
        self,
        raise_repeat_tool_call: bool = False,
        *,
        config: AgentConfig | None = None,
        llm_client: Client | None = None,
        **kwargs,
    ):
        if config is not None:
            client = llm_client or get_client(config.llm_client)
            parser = get_parser(config.parser)
            for source_cfg in config.sources:
                add_source_cfg(source_cfg.name, source_cfg)
            tools = [build_tool(tool_cfg) for tool_cfg in config.tools]
            llm_retry_policy = (
                RetryPolicy(config=config.llm_retry_config)
                if config.llm_retry_config
                else None
            )
            tool_retry_policy = (
                RetryPolicy(config=config.tool_retry_config)
                if config.tool_retry_config
                else None
            )
            super().__init__(
                llm_client=client,
                parser=parser,
                tools=tools,
                system_prompt=config.system_prompt,
                query_prompt=config.query_prompt,
                max_turn=config.max_turn,
                max_turn_prompt=config.max_turn_prompt,
                max_tokens=config.max_tokens,
                max_tokens_prompt=config.max_tokens_prompt,
                max_tokens_prompt_margin=config.max_tokens_prompt_margin,
                run_timeout_seconds=config.run_timeout_seconds,
                run_timeout_prompt=config.run_timeout_prompt,
                run_timeout_prompt_margin_seconds=config.run_timeout_prompt_margin_seconds,
                llm_retry_policy=llm_retry_policy,
                tool_retry_policy=tool_retry_policy,
            )
            self.raise_repeat_tool_call = config.raise_repeat_tool_call
            self.previous_tool_queries: set[tuple[str, str]] = set()
            return
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
            if isinstance(tc.arguments, dict) and "query" in tc.arguments:
                argument_str = normalize_query(tc.arguments["query"])
            else:
                argument_str = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
            if (tc.name, argument_str) in self.previous_tool_queries:
                if self.raise_repeat_tool_call:
                    raise RepeatedToolCallError(
                        f"Query {tc.name} has repeated arguments {argument_str}"
                    )
            else:
                self.previous_tool_queries.add((tc.name, argument_str))
        return await super().call_tools(tool_calls_list)
