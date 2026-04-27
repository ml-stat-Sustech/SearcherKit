from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from searchagent.agent.search_agent import SearchAgent
from searchagent.common.messages import ChatMessage, assistant
from searchagent.llm.parsers import Parser
from searchagent.tools import BaseTool


class TrackingTool(BaseTool):
    def __init__(self, name: str = "tracking") -> None:
        super().__init__(name=name)
        self.init_count = 0
        self.close_count = 0

    async def init(self, *args: Any, **kwargs: Any) -> None:
        self.init_count += 1

    async def close(self) -> None:
        self.close_count += 1

    async def _run(self, **kwargs: Any) -> str:
        return "ok"


class FinalAnswerParser(Parser):
    def from_model(self, messages: Iterable[dict[str, Any]]) -> Iterable[ChatMessage]:
        return [assistant("done")]

    def to_model(self, messages: Iterable[ChatMessage]) -> Iterable[dict[str, Any]]:
        return [{"role": message.role} for message in messages]


class FinalAnswerClient:
    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        return {"role": "assistant", "content": "done"}, SimpleNamespace(total_tokens=1)


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeProviderError(Exception):
    def __init__(self, message: str, response: FakeResponse | None = None) -> None:
        super().__init__(message)
        self.response = response


def test_search_agent_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        SearchAgent(
            llm_client=FinalAnswerClient(),
            parser=FinalAnswerParser(),
            tools=[TrackingTool("search"), TrackingTool("search")],
        )


def test_search_agent_run_closes_tools() -> None:
    async def run_agent() -> None:
        tool = TrackingTool()
        agent = SearchAgent(
            llm_client=FinalAnswerClient(),
            parser=FinalAnswerParser(),
            tools=[tool],
        )

        await agent.run("hello")

        assert tool.init_count == 1
        assert tool.close_count == 1

    asyncio.run(run_agent())


def test_context_length_detection_requires_explicit_marker() -> None:
    assert not SearchAgent._is_context_length_error(
        FakeProviderError("upstream returned 500", response=FakeResponse(text="internal error"))
    )
    assert SearchAgent._is_context_length_error(
        FakeProviderError(
            "bad request",
            response=FakeResponse(
                {
                    "error": {
                        "message": "This model's maximum context length was exceeded.",
                    }
                }
            ),
        )
    )
