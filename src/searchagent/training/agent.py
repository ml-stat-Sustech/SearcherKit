from __future__ import annotations

import json
import re
import string
from collections.abc import Iterable, Mapping
from typing import overload

from searchagent.agent.search_agent import LLMOutputError, SearchAgent
from searchagent.common.messages import ToolCall
from searchagent.common.retry import RetryPolicy
from searchagent.llm.base import Client, get_client
from searchagent.llm.parsers import get_parser
from searchagent.sources import add_source_cfg
from searchagent.tools import build_tool
from searchagent.training.config import AgentConfig


class TooManyToolCallsError(LLMOutputError):
    """Raised when the model issues too many parallel tool calls."""


class RepeatedToolCallError(LLMOutputError):
    """Raised when the model repeats a tool call with identical arguments."""


class TooManyQueriesError(LLMOutputError):
    """Raised when a search tool call contains too many queries."""


class InvalidSearchQueryListError(LLMOutputError):
    """Raised when a search tool call does not provide a valid query_list."""


_QUERY_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b")
_QUERY_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_search_query(query: str) -> str:
    text = query.lower().translate(_QUERY_PUNCTUATION_TABLE)
    text = _QUERY_ARTICLE_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _extract_search_queries(arguments: Mapping[str, object]) -> list[object]:
    query_list = arguments.get("query_list")
    if not query_list or not isinstance(query_list, list):
        raise InvalidSearchQueryListError(
            "Search tool call must provide a non-empty query_list"
        )
    return query_list


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
        max_queries_per_tool_call: int | None = None,
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
                llm_retry_policy=llm_retry_policy,
                tool_retry_policy=tool_retry_policy,
            )
            self.raise_repeat_tool_call = config.raise_repeat_tool_call
            self.max_queries_per_tool_call = config.max_queries_per_tool_call
            self.previous_tool_queries: set[tuple[str, str]] = set()
            return
        super().__init__(**kwargs)
        self.raise_repeat_tool_call = raise_repeat_tool_call
        self.max_queries_per_tool_call = max_queries_per_tool_call
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
            queries = _extract_search_queries(tc.arguments) if tc.name == "search" else None
            if queries is not None:
                if (
                    self.max_queries_per_tool_call is not None
                    and len(queries) > self.max_queries_per_tool_call
                ):
                    raise TooManyQueriesError(
                        f"Tool {tc.name} received {len(queries)} queries; "
                        f"maximum is {self.max_queries_per_tool_call}"
                    )
                for query in queries:
                    try:
                        query_key = (tc.name, _normalize_search_query(query))
                    except (AttributeError, TypeError) as exc:
                        raise InvalidSearchQueryListError(
                            "Search query_list items must be normalizable strings"
                        ) from exc
                    if query_key in self.previous_tool_queries:
                        if self.raise_repeat_tool_call:
                            raise RepeatedToolCallError(
                                f"Query {tc.name} has repeated query {query!r}"
                            )
                    else:
                        self.previous_tool_queries.add(query_key)
                continue
            argument_str = json.dumps(tc.arguments, sort_keys=True)
            if (tc.name, argument_str) in self.previous_tool_queries:
                if self.raise_repeat_tool_call:
                    raise RepeatedToolCallError(
                        f"Query {tc.name} has repeated arguments {argument_str}"
                    )
            else:
                self.previous_tool_queries.add((tc.name, argument_str))
        return await super().call_tools(tool_calls_list)
