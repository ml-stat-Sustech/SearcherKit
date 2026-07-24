from __future__ import annotations

import asyncio

import pytest

from searcherkit.agent.search_agent import SearchAgent
from searcherkit.training.agent import LLMContextError, SearchAgentTraining


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
