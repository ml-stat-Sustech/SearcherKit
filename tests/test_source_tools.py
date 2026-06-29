from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from searchagent.common.retry import RetryConfig
from searchagent.errors import RecoverableError
from searchagent.sources import SourceConfig, add_source_cfg
from searchagent.sources.base import Document
from searchagent.tools import SearchTool, SummarizerConfig, ToolConfig, VisitTool, build_tool


class FakeCompletions:
    def __init__(self, client: "FakeSummaryClient") -> None:
        self.client = client

    async def create(self, **payload: Any) -> SimpleNamespace:
        self.client.calls.append(payload)
        value = self.client.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=str(value)),
                )
            ]
        )


class FakeChat:
    def __init__(self, client: "FakeSummaryClient") -> None:
        self.completions = FakeCompletions(client)


class FakeSummaryClient:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.chat = FakeChat(self)


class FakeOpenAIModule:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def AsyncOpenAI(self, **kwargs: Any) -> FakeSummaryClient:
        self.created.append(kwargs)
        return FakeSummaryClient(['{"evidence": "config evidence", "summary": "config summary"}'])


def _summary_retry_config(max_tries: int) -> RetryConfig:
    return RetryConfig(
        max_tries=max_tries,
        exceptions=["pkg://searchagent.tools.summarizer:SummaryError"],
        jitter=None,
        factor=0,
    )


def _memory_source_cfg(name: str = "summary-memory") -> SourceConfig:
    return SourceConfig(
        name=name,
        type="memory",
        documents=[
            Document(
                id="doc-1",
                title="SearchAgent",
                text="SearchAgent wires tools to data sources.",
                url="https://example.test/searchagent",
            ),
            Document(
                id="doc-2",
                title="Summary Tools",
                text="Summary tools combine full document text before asking the summary model.",
                url="https://example.test/summary",
            ),
        ],
    )


def test_source_backed_search_and_visit_tools_share_named_source() -> None:
    async def run_tools() -> None:
        add_source_cfg(
            "memory",
            SourceConfig(
                name="memory",
                type="memory",
                documents=[
                    Document(
                        id="doc-1",
                        title="SearchAgent",
                        text="SearchAgent wires tools to data sources.",
                    ),
                    Document(
                        id="doc-2",
                        title="Other",
                        text="Unrelated content.",
                    ),
                ],
            ),
        )

        search_tool = build_tool(
            ToolConfig(type="search", name="search", source="memory"),
        )
        visit_tool = build_tool(
            ToolConfig(type="visit", name="visit", source="memory"),
        )

        search_payload = await search_tool.run(query="wires sources", top_k=1)
        assert "[SearchAgent]" in search_payload
        assert "SearchAgent wires tools to data sources." in search_payload

        visit_payload = await visit_tool.run(document_id="doc-1")
        assert "[SearchAgent]" in visit_payload
        assert "SearchAgent wires tools to data sources." in visit_payload

        visit_with_goal = await visit_tool.run(document_id="doc-1", goal="confirm wiring")
        assert "SearchAgent wires tools to data sources." in visit_with_goal

    asyncio.run(run_tools())


def test_build_tool_uses_registered_source_config() -> None:
    add_source_cfg(
        "configured-memory",
        SourceConfig(
            name="configured-memory",
            type="memory",
            documents=[
                Document(
                    id="doc-1",
                    title="Configured object",
                    text="Existing source configs can be reused.",
                )
            ],
        ),
    )

    tool = build_tool(ToolConfig(type="search", name="search", source="configured-memory"))
    assert tool.source.documents[0].id == "doc-1"


def test_factory_keeps_mcp_type_separate_from_model_visible_name() -> None:
    tool = build_tool(
        ToolConfig(
            type="mcp",
            name="search",
            mcp_tool_name="remote_search",
            endpoint="http://127.0.0.1:8100/mcp",
        )
    )

    assert tool.name == "search"
    assert getattr(tool, "mcp_tool_name") == "remote_search"


def test_search_tool_summary_uses_all_result_text() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("summary-all-text")
        add_source_cfg("summary-all-text", source_cfg)
        summary_client = FakeSummaryClient(
            ['{"evidence": "combined evidence", "summary": "combined summary"}']
        )
        tool = SearchTool(
            build_tool(ToolConfig(type="search", name="search", source="summary-all-text")).source,
            summarizer=SummarizerConfig(model="summary-model"),
            summary_client=summary_client,
        )

        payload = await tool.run(query="SearchAgent Summary", top_k=2)

        assert "combined evidence" in payload
        assert "combined summary" in payload
        prompt = summary_client.calls[0]["messages"][0]["content"]
        assert "SearchAgent wires tools to data sources." in prompt
        assert "Summary tools combine full document text" in prompt

    asyncio.run(run_tool())


def test_search_tool_summary_retries_once_then_succeeds() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("summary-retry")
        add_source_cfg("summary-retry", source_cfg)
        summary_client = FakeSummaryClient(
            [
                "not json at all",
                '{"evidence": "after retry", "summary": "ok"}',
            ]
        )
        tool = SearchTool(
            build_tool(ToolConfig(type="search", name="search", source="summary-retry")).source,
            summarizer=SummarizerConfig(
                model="summary-model",
                retry_config=_summary_retry_config(max_tries=2),
            ),
            summary_client=summary_client,
        )

        payload = await tool.run(query="SearchAgent", top_k=1)

        assert len(summary_client.calls) == 2
        assert "after retry" in payload

    asyncio.run(run_tool())


def test_search_tool_summary_failure_surfaces_recoverable_error() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("summary-failure")
        add_source_cfg("summary-failure", source_cfg)
        tool = SearchTool(
            build_tool(ToolConfig(type="search", name="search", source="summary-failure")).source,
            summarizer=SummarizerConfig(
                model="summary-model",
                retry_config=_summary_retry_config(max_tries=1),
            ),
            summary_client=FakeSummaryClient(['{"evidence": "", "summary": ""}']),
        )

        with pytest.raises(RecoverableError):
            await tool.run(query="SearchAgent", top_k=1)

    asyncio.run(run_tool())


def test_visit_tool_summary_uses_goal() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("visit-summary-goal")
        add_source_cfg("visit-summary-goal", source_cfg)
        summary_client = FakeSummaryClient(
            ['{"evidence": "goal evidence", "summary": "goal summary"}']
        )
        tool = VisitTool(
            build_tool(ToolConfig(type="visit", name="visit", source="visit-summary-goal")).source,
            summarizer=SummarizerConfig(model="summary-model"),
            summary_client=summary_client,
        )

        payload = await tool.run(document_id="doc-1", goal="confirm wiring")

        assert "goal evidence" in payload
        assert "goal summary" in payload
        prompt = summary_client.calls[0]["messages"][0]["content"]
        assert "confirm wiring" in prompt

    asyncio.run(run_tool())


def test_visit_tool_summary_retries_once_then_succeeds() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("visit-summary-retry")
        add_source_cfg("visit-summary-retry", source_cfg)
        summary_client = FakeSummaryClient(
            [
                "not json at all",
                '{"evidence": "visit retry evidence", "summary": "visit retry summary"}',
            ]
        )
        tool = VisitTool(
            build_tool(ToolConfig(type="visit", name="visit", source="visit-summary-retry")).source,
            summarizer=SummarizerConfig(
                model="summary-model",
                retry_config=_summary_retry_config(max_tries=2),
            ),
            summary_client=summary_client,
        )

        payload = await tool.run(document_id="doc-1", goal="confirm wiring")

        assert len(summary_client.calls) == 2
        assert "visit retry evidence" in payload
        assert "visit retry summary" in payload

    asyncio.run(run_tool())


def test_visit_tool_summary_failure_surfaces_recoverable_error() -> None:
    async def run_tool() -> None:
        source_cfg = _memory_source_cfg("visit-summary-failure")
        add_source_cfg("visit-summary-failure", source_cfg)
        tool = VisitTool(
            build_tool(ToolConfig(type="visit", name="visit", source="visit-summary-failure")).source,
            summarizer=SummarizerConfig(
                model="summary-model",
                retry_config=_summary_retry_config(max_tries=1),
            ),
            summary_client=FakeSummaryClient(['{"evidence": "", "summary": ""}']),
        )

        with pytest.raises(RecoverableError):
            await tool.run(document_id="doc-1", goal="confirm wiring")
        with pytest.raises(RecoverableError):
            await tool.run(document_id="missing", goal="confirm wiring")

    asyncio.run(run_tool())


def test_build_tool_creates_search_and_visit_tools_with_summary_from_registered_source_config() -> None:
    add_source_cfg("factory-summary", _memory_source_cfg("factory-summary"))

    search_tool = build_tool(
        ToolConfig(
            type="search",
            name="search",
            source="factory-summary",
            summarizer=SummarizerConfig(model="summary-model", api_key="key"),
        )
    )
    visit_tool = build_tool(
        ToolConfig(
            type="visit",
            name="visit",
            source="factory-summary",
            summarizer=SummarizerConfig(model="summary-model", api_key="key"),
        )
    )

    assert isinstance(search_tool, SearchTool)
    assert isinstance(visit_tool, VisitTool)
    assert search_tool.summary_enabled
    assert visit_tool.summary_enabled


def test_summary_tool_config_initializes_summary_client_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_tool() -> None:
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        add_source_cfg("config-summary-client", _memory_source_cfg("config-summary-client"))
        injected_client = FakeSummaryClient(['{"evidence": "unused", "summary": "unused"}'])

        tool = SearchTool(
            config=ToolConfig(
                type="search",
                name="search",
                source="config-summary-client",
                summarizer=SummarizerConfig(
                    model="config-model",
                    api_key="config-key",
                    base_url="https://summary.test/v1",
                ),
            ),
            summary_client=injected_client,
        )

        payload = await tool.run(query="SearchAgent", top_k=1)

        assert "unused" in payload
        assert fake_openai.created == []
        assert len(injected_client.calls) == 1

    asyncio.run(run_tool())


def test_summary_tools_require_model_and_api_key_without_injected_client() -> None:
    add_source_cfg("summary-validation", _memory_source_cfg("summary-validation"))
    source = build_tool(ToolConfig(type="search", name="search", source="summary-validation")).source

    with pytest.raises(ValueError, match="summarizer.model"):
        SearchTool(source, summarizer=SummarizerConfig(api_key="key"))
    with pytest.raises(ValueError, match="summarizer.api_key"):
        VisitTool(source, summarizer=SummarizerConfig(model="summary-model"))


def test_summary_tool_types_are_not_registered() -> None:
    with pytest.raises(ValueError, match="Unknown tool type"):
        build_tool(ToolConfig(type="search_summary", name="search_summary", source="memory"))
    with pytest.raises(ValueError, match="Unknown tool type"):
        build_tool(ToolConfig(type="visit_summary", name="visit_summary", source="memory"))
