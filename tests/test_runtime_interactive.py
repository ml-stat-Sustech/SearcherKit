from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from searcherkit.agent.search_agent import SearchAgentConfig
from searcherkit.common.messages import ToolCall, assistant, tool
from searcherkit.llm.base import ClientConfig, LLMStreamChunk
from searcherkit.llm.parsers import QwenParser
from searcherkit.common.errors import RecoverableError
from searcherkit.runtime.interactive import (
    InteractiveQueryConfig,
    InteractiveQueryRunner,
    resolve_record_dir,
)
from searcherkit.common.live_events import LiveEvent, emit_live_event
from searcherkit.sources import SourceConfig
from searcherkit.tools.base import ToolConfig


class SuccessfulAgent:
    async def run(
        self,
        query: str,
        session_id: int | None = None,
        extra: dict[str, Any] | None = None,
        live_event_sink: Any = None,
    ) -> list[Any]:
        await emit_live_event(live_event_sink, LiveEvent(kind="user_message", message=query))
        await emit_live_event(
            live_event_sink,
            LiveEvent(
                kind="tool_call_started",
                message='search({"query": "demo"})',
                data={"id": "call-1", "name": "search", "arguments": {"query": "demo"}},
            ),
        )
        call = ToolCall(name="search", arguments={"query": "demo"}, id="call-1")
        await emit_live_event(
            live_event_sink,
            LiveEvent(
                kind="tool_result",
                message="search -> result",
                data={"id": "call-1", "name": "search", "result": "result", "status": "completed"},
            ),
        )
        return [
            assistant("thinking", tool_calls=[call]),
            tool({"call-1": "result"}),
            assistant("done"),
        ]


class FailingAgent:
    async def run(
        self,
        query: str,
        session_id: int | None = None,
        extra: dict[str, Any] | None = None,
        live_event_sink: Any = None,
    ) -> list[Any]:
        await emit_live_event(live_event_sink, LiveEvent(kind="user_message", message=query))
        raise RecoverableError("boom")


class StreamingClient:
    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete_with_usage(self, messages, session_id=None, **kwargs):
        self.complete_calls += 1
        return {"role": "assistant", "content": "complete"}, {"total_tokens": 11}

    async def stream_complete_with_usage(self, messages, session_id=None, **kwargs):
        self.stream_calls += 1
        yield LLMStreamChunk(content_delta="hel")
        yield LLMStreamChunk(content_delta="lo")
        yield LLMStreamChunk(
            message={"role": "assistant", "content": "hello"},
            usage={"total_tokens": 7},
            done=True,
        )


class TaggedStreamingClient:
    async def complete_with_usage(self, messages, session_id=None, **kwargs):
        return {"role": "assistant", "content": "complete"}, None

    async def stream_complete_with_usage(self, messages, session_id=None, **kwargs):
        yield LLMStreamChunk(content_delta="<think>hid")
        yield LLMStreamChunk(content_delta="den</think><answer>Paris</answer>")
        yield LLMStreamChunk(
            message={"role": "assistant", "content": "<think>hidden</think><answer>Paris</answer>"},
            done=True,
        )


class CancellingAgent:
    async def run(
        self,
        query: str,
        session_id: int | None = None,
        extra: dict[str, Any] | None = None,
        live_event_sink: Any = None,
    ) -> list[Any]:
        await emit_live_event(live_event_sink, LiveEvent(kind="user_message", message=query))
        raise asyncio.CancelledError()


class ProgrammingErrorAgent:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def run(
        self,
        query: str,
        session_id: int | None = None,
        extra: dict[str, Any] | None = None,
        live_event_sink: Any = None,
    ) -> list[Any]:
        raise self._error


def test_resolve_record_dir_prefers_override_then_output_path_then_default(tmp_path, monkeypatch) -> None:
    assert resolve_record_dir(InteractiveQueryConfig(record_dir=str(tmp_path / "manual"))) == tmp_path / "manual"
    assert resolve_record_dir(InteractiveQueryConfig(output_path=str(tmp_path / "out"))) == tmp_path / "out" / "interactive"
    monkeypatch.chdir(tmp_path)
    assert resolve_record_dir(InteractiveQueryConfig()) == Path("outputs/interactive")


def test_interactive_query_runner_persists_completed_record_and_events(tmp_path) -> None:
    async def run_case() -> None:
        events: list[LiveEvent] = []
        result = await InteractiveQueryRunner(
            config=InteractiveQueryConfig(record_dir=str(tmp_path)),
            build_agent=SuccessfulAgent,
        ).run_query("hello", live_event_sink=events.append)

        record = json.loads(result.record_path.read_text(encoding="utf-8"))
        assert result.status == "completed"
        assert record["kind"] == "interactive_query_run"
        assert record["status"] == "completed"
        assert record["input"] == "hello"
        assert record["history"] is not None
        assert record["stats"] == {"turns": 2, "tool_calls": 1, "tool_messages": 1}
        assert [event["kind"] for event in record["events"]] == [
            "run_started",
            "user_message",
            "tool_call_started",
            "tool_result",
            "run_completed",
        ]
        assert record["events"][0]["data"]["active_source"] == {"state": "none"}
        assert record["events"][0]["data"]["active_model"] == {"provider": "openai", "model": "", "base_url": None}
        assert [event.kind for event in events] == [
            "run_started",
            "user_message",
            "tool_call_started",
            "tool_result",
            "run_completed",
        ]
        assert (tmp_path / "run.log").exists()
        assert (tmp_path / "traces" / f"{record['trace_id']}.log").exists()

    asyncio.run(run_case())


def test_interactive_query_runner_enables_streaming_for_default_agent(tmp_path) -> None:
    async def run_case() -> None:
        from searcherkit.agent.search_agent import SearchAgent
        from searcherkit.runtime.interactive import InteractiveQueryConfig

        client = StreamingClient()
        events: list[LiveEvent] = []
        agent_holder: list[Any] = []

        def build_agent():
            agent = SearchAgent(
                llm_client=client,
                parser=QwenParser(),
                tools=[],
                stream_llm=True,
            )
            agent_holder.append(agent)
            return agent

        result = await InteractiveQueryRunner(
            config=InteractiveQueryConfig(record_dir=str(tmp_path)),
            build_agent=build_agent,
        ).run_query("hello", live_event_sink=events.append)

        record = json.loads(result.record_path.read_text(encoding="utf-8"))
        assert result.status == "completed"
        assert client.stream_calls == 1
        assert client.complete_calls == 0
        assert [event.kind for event in events] == [
            "run_started",
            "user_message",
            "assistant_turn_started",
            "assistant_delta",
            "assistant_delta",
            "assistant_message",
            "run_completed",
        ]
        assert [event["kind"] for event in record["events"]] == [event.kind for event in events]
        assert record["history"][-1]["content"] == "hello"
        assert agent_holder[0].context_token_size == 7

    asyncio.run(run_case())


def test_interactive_query_runner_streaming_uses_parser_for_live_deltas_without_answer_parsing(tmp_path) -> None:
    async def run_case() -> None:
        from searcherkit.agent.search_agent import SearchAgent

        events: list[LiveEvent] = []
        result = await InteractiveQueryRunner(
            config=InteractiveQueryConfig(record_dir=str(tmp_path)),
            build_agent=lambda: SearchAgent(llm_client=TaggedStreamingClient(), parser=QwenParser(), tools=[], stream_llm=True),
        ).run_query("hello", live_event_sink=events.append)

        assistant_deltas = [event for event in events if event.kind == "assistant_delta"]
        record = json.loads(result.record_path.read_text(encoding="utf-8"))

        assert [(event.data["field"], event.data["delta"]) for event in assistant_deltas] == [
            ("thinking", "hid"),
            ("thinking", "den"),
            ("content", "<answer>Paris</answer>"),
        ]
        assert record["history"][-1]["thinking"] == "hidden"
        assert record["history"][-1]["content"] == "<answer>Paris</answer>"

    asyncio.run(run_case())


def test_interactive_query_runner_records_active_model_from_run_config(tmp_path) -> None:
    async def run_case() -> None:
        config = InteractiveQueryConfig(
            record_dir=str(tmp_path),
            agent=SearchAgentConfig(
                llm_client=ClientConfig(
                    type="openai",
                    model="llama3.2:1b",
                    base_url="http://127.0.0.1:11434/v1",
                    api_key="secret",
                ),
            ),
        )
        result = await InteractiveQueryRunner(
            config=config,
            build_agent=SuccessfulAgent,
        ).run_query("hello")

        record = json.loads(result.record_path.read_text(encoding="utf-8"))

        assert record["events"][0]["kind"] == "run_started"
        assert record["events"][0]["data"]["active_model"] == {
            "provider": "openai",
            "model": "llama3.2:1b",
            "base_url": "http://127.0.0.1:11434/v1",
        }
        assert "api_key" not in record["events"][0]["data"]["active_model"]
        assert "active_model" not in record

    asyncio.run(run_case())


def test_interactive_query_runner_records_active_source_from_run_config(tmp_path) -> None:
    async def run_case() -> None:
        config = InteractiveQueryConfig(
            record_dir=str(tmp_path),
            agent=SearchAgentConfig(
                sources=[
                    SourceConfig(
                        type="file",
                        name="wiki",
                        root_path=str(tmp_path),
                    )
                ],
                tools=[
                    ToolConfig(type="search", name="search", source=["wiki"]),
                    ToolConfig(type="visit", name="visit", source=["wiki"]),
                ],
            ),
        )
        result = await InteractiveQueryRunner(
            config=config,
            build_agent=SuccessfulAgent,
        ).run_query("hello")

        record = json.loads(result.record_path.read_text(encoding="utf-8"))

        assert record["events"][0]["kind"] == "run_started"
        assert record["events"][0]["data"]["active_source"] == {
            "state": "active",
            "name": "wiki",
            "type": "file",
            "tool_sources": ["wiki"],
        }
        assert "active_source" not in record

    asyncio.run(run_case())


def test_interactive_query_runner_persists_failed_record(tmp_path) -> None:
    async def run_case() -> None:
        result = await InteractiveQueryRunner(
            config=InteractiveQueryConfig(record_dir=str(tmp_path)),
            build_agent=FailingAgent,
        ).run_query("hello")

        record = json.loads(result.record_path.read_text(encoding="utf-8"))
        assert result.status == "failed"
        assert record["status"] == "failed"
        assert record["history"] is None
        assert record["stats"] is None
        assert record["error"]["type"] == "RecoverableError"
        assert [event["kind"] for event in record["events"]] == ["run_started", "user_message", "run_failed"]

    asyncio.run(run_case())


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, KeyError])
def test_interactive_query_runner_propagates_programming_errors(tmp_path, error_type) -> None:
    async def run_case() -> None:
        with pytest.raises(error_type):
            await InteractiveQueryRunner(
                config=InteractiveQueryConfig(record_dir=str(tmp_path)),
                build_agent=lambda: ProgrammingErrorAgent(error_type("bug")),
            ).run_query("hello")

        assert not list(tmp_path.glob("*.json"))

    asyncio.run(run_case())


def test_interactive_query_runner_persists_cancelled_record(tmp_path) -> None:
    async def run_case() -> None:
        result = await InteractiveQueryRunner(
            config=InteractiveQueryConfig(record_dir=str(tmp_path)),
            build_agent=CancellingAgent,
        ).run_query("hello")

        record = json.loads(result.record_path.read_text(encoding="utf-8"))
        assert result.status == "cancelled"
        assert record["status"] == "cancelled"
        assert record["history"] is None
        assert record["stats"] is None
        assert [event["kind"] for event in record["events"]] == ["run_started", "user_message", "run_cancelled"]

    asyncio.run(run_case())
