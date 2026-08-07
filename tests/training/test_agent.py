from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from searcherkit.agent.search_agent import SearchAgent
from searcherkit.common.messages import ToolCall
from searcherkit.training.agent import (
    LLMContextError,
    SearchAgentTraining,
    VisitNotSearchedError,
)
from searcherkit.training.config import AgentConfig


def test_agent_config_enables_visit_search_result_check_by_default() -> None:
    assert AgentConfig().check_visit_in_search_results is True


def test_parse_and_call_llm_converts_areal_context_length_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ValueError(
        "len of prompt tokens 32768 exceeds max_total_tokens 32768"
    )

    async def raise_context_length_error(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(SearchAgent, "parse_and_call_llm", raise_context_length_error)
    agent = SearchAgentTraining.__new__(SearchAgentTraining)

    async def exercise() -> None:
        with pytest.raises(LLMContextError) as exc_info:
            await agent.parse_and_call_llm([])
        assert exc_info.value.__cause__ is error

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "message",
    [
        (
            "max_new_tokens (0) is non-positive! max_tokens=32768, "
            "prompt_len=32768, max_new_tokens=8151."
        ),
        (
            "Failed after 3 retries each. Payload: {'input_ids': [1, 2], "
            "'sampling_params': {'max_new_tokens': 8151}}. Addr: localhost:8000. "
            "Endpoint: /generate. Last error: ClientResponseError("
            "RequestInfo(...), (), status=400, message='Bad Request', headers=None)"
        ),
    ],
)
def test_parse_and_call_llm_converts_areal_runtime_context_errors(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    error = RuntimeError(message)

    async def raise_context_length_error(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(SearchAgent, "parse_and_call_llm", raise_context_length_error)
    agent = SearchAgentTraining.__new__(SearchAgentTraining)

    async def exercise() -> None:
        with pytest.raises(LLMContextError) as exc_info:
            await agent.parse_and_call_llm([])
        assert exc_info.value.__cause__ is error

    asyncio.run(exercise())


def test_parse_and_call_llm_preserves_unrelated_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ValueError("messages cannot be empty")

    async def raise_unrelated_value_error(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(SearchAgent, "parse_and_call_llm", raise_unrelated_value_error)
    agent = SearchAgentTraining.__new__(SearchAgentTraining)

    async def exercise() -> None:
        with pytest.raises(ValueError) as exc_info:
            await agent.parse_and_call_llm([])
        assert exc_info.value is error

    asyncio.run(exercise())


def test_parse_and_call_llm_preserves_unrelated_areal_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError(
        "Failed after 3 retries each. Endpoint: /generate. "
        "Last error: ClientResponseError(status=503, message='Service Unavailable')"
    )

    async def raise_unrelated_error(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(SearchAgent, "parse_and_call_llm", raise_unrelated_error)
    agent = SearchAgentTraining.__new__(SearchAgentTraining)

    async def exercise() -> None:
        with pytest.raises(RuntimeError) as exc_info:
            await agent.parse_and_call_llm([])
        assert exc_info.value is error

    asyncio.run(exercise())


def test_call_tools_allows_documents_returned_by_prior_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SearchAgentTraining.__new__(SearchAgentTraining)
    agent.tool_dict = {
        "search": SimpleNamespace(argument_mapping={}),
        "visit": SimpleNamespace(argument_mapping={"link": "document_id"}),
    }
    agent.previous_tool_queries = set()
    agent.searched_documents = set()
    agent.check_visit_in_search_results = True

    async def call_tools(
        self: SearchAgent,
        tool_calls: list[ToolCall],
    ) -> list[tuple[str, dict[str, object]]]:
        if tool_calls[0].name == "search":
            return [
                (
                    "result",
                    {
                        "documents": [
                            {"id": "doc-1", "url": "https://example.com/doc-1"}
                        ]
                    },
                )
            ]
        return [("visited", {})]

    monkeypatch.setattr(SearchAgent, "call_tools", call_tools)

    async def exercise() -> None:
        await agent.call_tools(
            [ToolCall(name="search", arguments={"query": "example"})]
        )
        result = await agent.call_tools(
            [
                ToolCall(
                    name="visit",
                    arguments={"link": "https://example.com/doc-1"},
                )
            ]
        )
        assert result == [("visited", {})]

    asyncio.run(exercise())


def test_call_tools_rejects_visit_absent_from_prior_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SearchAgentTraining.__new__(SearchAgentTraining)
    agent.tool_dict = {
        "visit": SimpleNamespace(argument_mapping={"link": "document_id"})
    }
    agent.previous_tool_queries = set()
    agent.searched_documents = set()
    agent.check_visit_in_search_results = True
    called = False

    async def call_tools(
        self: SearchAgent,
        tool_calls: list[ToolCall],
    ) -> list[tuple[str, dict[str, object]]]:
        nonlocal called
        called = True
        return [("visited", {})]

    monkeypatch.setattr(SearchAgent, "call_tools", call_tools)

    async def exercise() -> None:
        with pytest.raises(VisitNotSearchedError, match="not returned"):
            await agent.call_tools(
                [
                    ToolCall(
                        name="visit",
                        arguments={"link": "https://example.com/unsearched"},
                    )
                ]
            )

    asyncio.run(exercise())
    assert called is False


def test_call_tools_allows_unsearched_visit_when_check_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SearchAgentTraining.__new__(SearchAgentTraining)
    agent.tool_dict = {
        "visit": SimpleNamespace(argument_mapping={"link": "document_id"})
    }
    agent.previous_tool_queries = set()
    agent.searched_documents = set()
    agent.check_visit_in_search_results = False

    async def call_tools(
        self: SearchAgent,
        tool_calls: list[ToolCall],
    ) -> list[tuple[str, dict[str, object]]]:
        return [("visited", {})]

    monkeypatch.setattr(SearchAgent, "call_tools", call_tools)

    async def exercise() -> None:
        result = await agent.call_tools(
            [
                ToolCall(
                    name="visit",
                    arguments={"link": "https://example.com/unsearched"},
                )
            ]
        )
        assert result == [("visited", {})]

    asyncio.run(exercise())


def test_reset_clears_searched_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SearchAgentTraining.__new__(SearchAgentTraining)
    agent.searched_documents = {"https://example.com/doc-1"}
    agent.previous_tool_queries = {("search", "example")}

    monkeypatch.setattr(SearchAgent, "reset", lambda self: None)

    agent.reset()

    assert agent.searched_documents == set()
    assert agent.previous_tool_queries == set()
