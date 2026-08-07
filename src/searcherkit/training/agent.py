from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import overload, Any, TYPE_CHECKING

from searcherkit.agent.search_agent import SearchAgent
from searcherkit.common.messages import ToolCall
from searcherkit.common.retry import RetryPolicy
from searcherkit.common.errors import LLMError
from searcherkit.llm.base import Client, get_client
from searcherkit.llm.parsers import get_parser
from searcherkit.sources import add_source_cfg
from searcherkit.tools import build_tool
from searcherkit.tools.base import map_arguments
from searcherkit.tools.summarizer import Summarizer
from searcherkit.training.config import AgentConfig
from searcherkit.training.rewards import normalize_query

if TYPE_CHECKING:
    from searcherkit.common.messages import ChatMessage


class TooManyToolCallsError(LLMError):
    """Raised when the model issues too many parallel tool calls."""
    pass


class RepeatedToolCallError(LLMError):
    """Raised when the model repeats a tool call with identical arguments."""
    pass


class LLMContextError(LLMError):
    """Raised when the LLM request exceeds the context limit."""
    pass


class VisitNotSearchedError(LLMError):
    """Raised when the model visits a document absent from prior search results."""


_AREAL_CONTEXT_LENGTH_ERROR = re.compile(
    r"^len of prompt tokens \d+ exceeds max_total_tokens \d+$"
)
_AREAL_NON_POSITIVE_MAX_NEW_TOKENS_ERROR = re.compile(
    r"^max_new_tokens \(-?\d+\) is non-positive! "
    r"max_tokens=\d+, prompt_len=\d+, max_new_tokens=\d+\.$"
)
_AREAL_GENERATE_BAD_REQUEST = re.compile(
    r"^Failed after \d+ retries each\. .*Endpoint: /generate\. "
    r"Last error: ClientResponseError\(.*status=400, message='Bad Request'",
    re.DOTALL,
)


def _is_areal_context_length_error(exc: ValueError | RuntimeError) -> bool:
    message = str(exc)
    return (
        _AREAL_CONTEXT_LENGTH_ERROR.fullmatch(message) is not None
        or _AREAL_NON_POSITIVE_MAX_NEW_TOKENS_ERROR.fullmatch(message) is not None
        # AReaL discards the SGLang response body when /generate returns 400,
        # leaving this wrapper as the only context-overflow signal available.
        or _AREAL_GENERATE_BAD_REQUEST.match(message) is not None
    )


class SearchAgentTraining(SearchAgent):
    @overload
    def __init__(
        self,
        *,
        config: AgentConfig,
        llm_client: Client | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        raise_repeat_tool_call: bool = False,
        check_visit_in_search_results: bool = True,
        **kwargs,
    ) -> None: ...

    def __init__(
        self,
        raise_repeat_tool_call: bool = False,
        check_visit_in_search_results: bool = True,
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
                llm_retry_policy=llm_retry_policy,
                tool_retry_policy=tool_retry_policy,
            )
            for tool_config in config.tools:
                summarizer_config = tool_config.summarizer
                if summarizer_config is None:
                    continue
                tool = self.tool_dict[tool_config.name]
                tool.configure_summarizer(
                    summarizer=Summarizer(
                        client=client,
                        max_chars=summarizer_config.max_chars,
                        timeout=summarizer_config.timeout,
                        max_concurrency=summarizer_config.max_concurrency,
                        default_kwargs=summarizer_config.default_kwargs,
                        retry_policy=(
                            RetryPolicy(config=summarizer_config.retry_config)
                            if summarizer_config.retry_config is not None
                            else None
                        ),
                    )
                )
            self.raise_repeat_tool_call = config.raise_repeat_tool_call
            self.check_visit_in_search_results = config.check_visit_in_search_results
            self.previous_tool_queries: set[tuple[str, str]] = set()
            self.searched_documents: set[str] = set()
            return
        super().__init__(**kwargs)
        self.raise_repeat_tool_call = raise_repeat_tool_call
        self.check_visit_in_search_results = check_visit_in_search_results
        self.previous_tool_queries: set[tuple[str, str]] = set()
        self.searched_documents: set[str] = set()

    def reset(self):
        super().reset()
        self.previous_tool_queries = set()
        self.searched_documents = set()

    async def parse_and_call_llm(self, history: list[ChatMessage]):
        try:
            return await super().parse_and_call_llm(history)
        except (ValueError, RuntimeError) as exc:
            if not _is_areal_context_length_error(exc):
                raise
            raise LLMContextError(str(exc)) from exc

    async def call_tools(self, tool_calls: Iterable[ToolCall]) -> list[tuple[str, Any]]:
        tool_calls_list = list(tool_calls)
        if len(tool_calls_list) > 10:
            raise TooManyToolCallsError("Too many parallel tool calls")
        for tc in tool_calls_list:
            if tc.name not in self.tool_dict:
                raise LLMError(f"Tool {tc.name} not found")
            if tc.name == "visit" and self.check_visit_in_search_results:
                # ToolCall arguments use model-visible names. Apply the same
                # mapping as BaseTool.run before enforcing the visit policy.
                mapped_arguments = map_arguments(
                    tc.arguments,
                    self.tool_dict[tc.name].argument_mapping,
                )
                document = mapped_arguments.get("document_id")
                documents = document if isinstance(document, list) else [document]
                not_searched = [
                    item
                    for item in documents
                    if not isinstance(item, str) or item not in self.searched_documents
                ]
                if not_searched:
                    raise VisitNotSearchedError(
                        f"Visit targets were not returned by a prior search: {not_searched!r}"
                    )
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
        results = await super().call_tools(tool_calls_list)
        for tool_call, (_, extensions) in zip(tool_calls_list, results, strict=True):
            if tool_call.name != "search":
                continue
            documents = extensions.get("documents", [])
            if not isinstance(documents, list):
                continue
            for document in documents:
                if not isinstance(document, dict):
                    continue
                for key in ("id", "url"):
                    value = document.get(key)
                    if isinstance(value, str):
                        self.searched_documents.add(value)
        return results
