from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from openai import APITimeoutError

sys.path.append(str(Path(__file__).resolve().parents[1] / "llm"))
from mock_endpoints import upstream_parsed_endpoint  # noqa: E402

from searchagent.agent import SearchAgent, SearchAgentConfig
from searchagent.agent.search_agent import LLMContextError
from searchagent.common.messages import ToolMessage
from searchagent.common.retry import RetryPolicy
from searchagent.llm.base import ClientConfig, OpenAIConfig
from searchagent.llm.openai import OpenAIClient
from searchagent.llm.parsers import ParserConfig
from searchagent.llm.parsers.upstream import UpstreamParser
from searchagent.tools import BaseTool


class RetryableToolError(RuntimeError):
    pass


class TrackingTool(BaseTool):
    def __init__(
        self,
        *,
        name: str = upstream_parsed_endpoint.TOOL_NAME,
        failures_before_success: int = 0,
        always_fail: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            description="Look up test information.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        self.failures_before_success = failures_before_success
        self.always_fail = always_fail
        self.init_count = 0
        self.close_count = 0
        self.calls: list[dict[str, Any]] = []

    async def init(self, *args: Any, **kwargs: Any) -> None:
        self.init_count += 1

    async def close(self) -> None:
        self.close_count += 1

    async def _run(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        if self.always_fail or len(self.calls) <= self.failures_before_success:
            raise RetryableToolError("temporary tool failure")
        return f"lookup result for {kwargs['query']}"


def _parser() -> UpstreamParser:
    return UpstreamParser()


def _client() -> OpenAIClient:
    return OpenAIClient(
        model="gpt-test",
        api_key="test-key",
        base_url=upstream_parsed_endpoint.BASE_URL,
        max_retries=0,
    )


def _agent(
    *,
    tools: list[BaseTool] | None = None,
    llm_retry_policy: RetryPolicy | None = None,
    tool_retry_policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> SearchAgent:
    return SearchAgent(
        llm_client=_client(),
        parser=_parser(),
        tools=tools or [],
        llm_retry_policy=llm_retry_policy,
        tool_retry_policy=tool_retry_policy,
        **kwargs,
    )


def _config_agent() -> SearchAgent:
    return SearchAgent(
        config=SearchAgentConfig(
            llm_client=ClientConfig(
                type="openai",
                model="gpt-test",
                openai=OpenAIConfig(
                    api_key="test-key",
                    base_url=upstream_parsed_endpoint.BASE_URL,
                    extra_client_kwargs={"max_retries": 0},
                ),
            ),
            parser=ParserConfig(
                type="upstream",
            ),
            system_prompt="Config system.",
            query_prompt="Config query: {query}",
        )
    )


def _retry_policy(*exceptions: type[Exception]) -> RetryPolicy:
    return RetryPolicy(
        max_tries=2,
        exceptions=exceptions,
        jitter=None,
        factor=0.0,
    )


@pytest.mark.parametrize("agent_factory", [_agent, _config_agent])
def test_final_answer(agent_factory: Any) -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(outcomes=("final",)):
            agent = agent_factory()

            history = await agent.run("Ping?", session_id=0)

        assert [message.role for message in history] == ["system", "user", "assistant"]
        assert history[-1].content == upstream_parsed_endpoint.FINAL_CONTENT
        assert all(client.is_closed() for client in agent.client.clients)

    asyncio.run(run())


def test_run_sends_prompt_session_and_provider_tools() -> None:
    async def run() -> None:
        captured_payload: dict[str, Any] = {}
        tool = TrackingTool()
        with upstream_parsed_endpoint.context(captured_payload, outcomes=("final",)):
            agent = _agent(
                tools=[tool],
                system_prompt="System prompt.",
                query_prompt="Question: {query}",
            )

            history = await agent.run("Ping?", session_id=0)

        assert tool.init_count == 1
        assert tool.close_count == 1
        assert history[0].content == "System prompt."
        assert history[1].content == "Question: Ping?"
        assert captured_payload["messages"] == [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Question: Ping?"},
        ]
        assert captured_payload["tools"] == [tool.as_openai_tool()]
        assert captured_payload["model"] == "gpt-test"

    asyncio.run(run())


def test_tool_call_appends_tool_result() -> None:
    async def run() -> None:
        tool = TrackingTool()
        with upstream_parsed_endpoint.context(outcomes=("tool_call", "final")):
            agent = _agent(tools=[tool])

            history = await agent.run("Use a tool.", session_id=0)

        assert tool.calls == [upstream_parsed_endpoint.TOOL_ARGUMENTS]
        assert [message.role for message in history] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        tool_message = history[3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_responses == {
            upstream_parsed_endpoint.TOOL_CALL["id"]: "lookup result for next"
        }
        assert history[-1].content == upstream_parsed_endpoint.FINAL_CONTENT

    asyncio.run(run())


def test_unknown_tool_returns_error_response() -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(outcomes=("tool_call", "final")):
            agent = _agent()

            history = await agent.run("Use a missing tool.", session_id=0)

        tool_message = history[3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_responses[upstream_parsed_endpoint.TOOL_CALL["id"]] == "Error: Tool lookup not found"

    asyncio.run(run())


def test_max_turn_prompt_replaces_first_tool_result() -> None:
    async def run() -> None:
        tool = TrackingTool()
        with upstream_parsed_endpoint.context(outcomes=("tool_call", "final")):
            agent = _agent(
                tools=[tool],
                max_turn=1,
                max_turn_prompt="Answer now.",
            )

            history = await agent.run("Use a tool.", session_id=0)

        tool_message = history[3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_responses[upstream_parsed_endpoint.TOOL_CALL["id"]] == "Answer now."
        assert history[-1].content == upstream_parsed_endpoint.FINAL_CONTENT

    asyncio.run(run())


def test_max_tokens_prompt_replaces_first_tool_result() -> None:
    async def run() -> None:
        tool = TrackingTool()
        with upstream_parsed_endpoint.context(outcomes=("tool_call_high_usage", "final")):
            agent = _agent(
                tools=[tool],
                max_tokens=140,
                max_tokens_prompt_margin=20,
                max_tokens_prompt="Context answer now.",
            )

            history = await agent.run("Use a tool.", session_id=0)

        tool_message = history[3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_responses[upstream_parsed_endpoint.TOOL_CALL["id"]] == "Context answer now."
        assert history[-1].content == upstream_parsed_endpoint.FINAL_CONTENT

    asyncio.run(run())


def test_context_length_error_surfaces_as_agent_context_error() -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(
            outcomes=("context_length",),
            assert_all_called=False,
        ):
            agent = _agent()

            with pytest.raises(LLMContextError):
                await agent.run("Too long.", session_id=0)

    asyncio.run(run())


def test_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        _agent(tools=[TrackingTool(), TrackingTool()])


def test_llm_retry_success() -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(
            outcomes=("timeout", "final"),
            assert_all_called=False,
        ) as router:
            agent = _agent(llm_retry_policy=_retry_policy(APITimeoutError))

            history = await agent.run("Retry.", session_id=0)

            assert len(router.calls) == 2
            assert history[-1].content == upstream_parsed_endpoint.FINAL_CONTENT

    asyncio.run(run())


def test_llm_retry_failure() -> None:
    async def run() -> None:
        with upstream_parsed_endpoint.context(
            outcomes=("timeout", "timeout"),
            assert_all_called=False,
        ) as router:
            agent = _agent(llm_retry_policy=_retry_policy(APITimeoutError))

            with pytest.raises(APITimeoutError):
                await agent.run("Retry.", session_id=0)

            assert len(router.calls) == 2
            assert all(client.is_closed() for client in agent.client.clients)

    asyncio.run(run())


def test_tool_retry_success() -> None:
    async def run() -> None:
        tool = TrackingTool(failures_before_success=1)
        with upstream_parsed_endpoint.context(outcomes=("tool_call", "final")):
            agent = _agent(
                tools=[tool],
                tool_retry_policy=_retry_policy(RetryableToolError),
            )

            history = await agent.run("Retry tool.", session_id=0)

        assert len(tool.calls) == 2
        tool_message = history[3]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_responses[upstream_parsed_endpoint.TOOL_CALL["id"]] == "lookup result for next"

    asyncio.run(run())


def test_tool_retry_failure() -> None:
    async def run() -> None:
        tool = TrackingTool(always_fail=True)
        with upstream_parsed_endpoint.context(
            outcomes=("tool_call",),
            assert_all_called=False,
        ):
            agent = _agent(
                tools=[tool],
                tool_retry_policy=_retry_policy(RetryableToolError),
            )

            with pytest.raises(RetryableToolError):
                await agent.run("Retry tool.", session_id=0)

        assert len(tool.calls) == 2
        assert tool.close_count == 1

    asyncio.run(run())
