from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from searchagent.common.errors import RecoverableError
from searchagent.sources import Document, SourceConfig, add_source_cfg
from searchagent.sources.memory import MemorySource
from searchagent.tools import ToolConfig, build_tool
from searchagent.tools.multi_source_search import MultiSourceSearchTool
from searchagent.tools.multi_source_visit import MultiSourceVisitTool


SearchFactory = Callable[..., MultiSourceSearchTool]
VisitFactory = Callable[..., MultiSourceVisitTool]


def _runtime_documents() -> list[Document]:
    return [
        Document(
            id="runtime-doc",
            title="Runtime Source",
            text="Runtime source explains pluggable source-backed execution.",
            url="https://example.test/runtime",
        ),
        Document(
            id="runtime-other",
            title="Runtime Other",
            text="Other runtime material.",
            url="https://example.test/runtime-other",
        ),
    ]


def _summary_documents() -> list[Document]:
    return [
        Document(
            id="summary-doc",
            title="Summary Source",
            text="Summary source explains evidence extraction.",
            url="https://example.test/summary",
            metadata={"kind": "summary"},
        ),
    ]


def _source_names(test_name: str) -> tuple[str, str]:
    return (
        f"tools-multi-runtime-{test_name}",
        f"tools-multi-summary-{test_name}",
    )


def _add_sources(test_name: str) -> tuple[str, str]:
    runtime_name, summary_name = _source_names(test_name)
    add_source_cfg(
        runtime_name,
        SourceConfig(
            type="memory",
            name=runtime_name,
            documents=_runtime_documents(),
        ),
    )
    add_source_cfg(
        summary_name,
        SourceConfig(
            type="memory",
            name=summary_name,
            documents=_summary_documents(),
        ),
    )
    return runtime_name, summary_name


def _direct_search_tool(*, test_name: str) -> MultiSourceSearchTool:
    runtime_name, summary_name = _source_names(test_name)
    return MultiSourceSearchTool(
        {
            runtime_name: MemorySource(documents=_runtime_documents()),
            summary_name: MemorySource(documents=_summary_documents()),
        },
        name="search",
    )


def _config_search_tool(*, test_name: str) -> MultiSourceSearchTool:
    source_names = list(_add_sources(test_name))
    tool = build_tool(
        ToolConfig(
            type="multi_source_search",
            name="search",
            source=source_names,
        )
    )
    assert isinstance(tool, MultiSourceSearchTool)
    return tool


def _direct_visit_tool(*, test_name: str) -> MultiSourceVisitTool:
    runtime_name, summary_name = _source_names(test_name)
    return MultiSourceVisitTool(
        {
            runtime_name: MemorySource(documents=_runtime_documents()),
            summary_name: MemorySource(documents=_summary_documents()),
        },
        name="visit",
    )


def _config_visit_tool(*, test_name: str) -> MultiSourceVisitTool:
    source_names = list(_add_sources(test_name))
    tool = build_tool(
        ToolConfig(
            type="multi_source_visit",
            name="visit",
            source=source_names,
        )
    )
    assert isinstance(tool, MultiSourceVisitTool)
    return tool


@pytest.mark.parametrize("tool_factory", [_direct_search_tool, _config_search_tool])
def test_multi_source_search_run(tool_factory: SearchFactory) -> None:
    async def run() -> None:
        test_name = "search-run"
        runtime_name, summary_name = _source_names(test_name)
        tool = tool_factory(test_name=test_name)

        runtime_result = await tool.run(
            query="pluggable execution",
            source=runtime_name,
        )
        summary_result = await tool.run(
            query="evidence extraction",
            source=summary_name,
        )

        assert "Runtime source explains pluggable source-backed execution." in runtime_result
        assert "Summary source explains evidence extraction." not in runtime_result
        assert "Summary source explains evidence extraction." in summary_result

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_search_tool, _config_search_tool])
def test_multi_source_search_unknown_source(
    tool_factory: SearchFactory,
) -> None:
    async def run() -> None:
        test_name = "search-unknown"
        runtime_name, summary_name = _source_names(test_name)
        tool = tool_factory(test_name=test_name)

        with pytest.raises(RecoverableError) as exc_info:
            await tool.run(query="pluggable execution", source="missing-source")

        message = str(exc_info.value)
        assert "unknown source 'missing-source'" in message
        assert runtime_name in message
        assert summary_name in message

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_visit_tool, _config_visit_tool])
def test_multi_source_visit_run(tool_factory: VisitFactory) -> None:
    async def run() -> None:
        test_name = "visit-run"
        runtime_name, summary_name = _source_names(test_name)
        tool = tool_factory(test_name=test_name)

        runtime_result = await tool.run(
            document_id="runtime-doc",
            source=runtime_name,
            goal="confirm runtime",
        )
        summary_result = await tool.run(
            document_id="summary-doc",
            source=summary_name,
            goal="confirm summary",
        )

        assert "[Runtime Source](https://example.test/runtime)" in runtime_result
        assert "Runtime source explains pluggable source-backed execution." in runtime_result
        assert "[Summary Source](https://example.test/summary)" in summary_result
        assert "Metadata: {'kind': 'summary'}" in summary_result

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_visit_tool, _config_visit_tool])
def test_multi_source_visit_unknown_source(tool_factory: VisitFactory) -> None:
    async def run() -> None:
        test_name = "visit-unknown"
        runtime_name, summary_name = _source_names(test_name)
        tool = tool_factory(test_name=test_name)

        with pytest.raises(RecoverableError) as exc_info:
            await tool.run(document_id="runtime-doc", source="missing-source")

        message = str(exc_info.value)
        assert "unknown source 'missing-source'" in message
        assert runtime_name in message
        assert summary_name in message

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_visit_tool, _config_visit_tool])
def test_multi_source_visit_missing_document(tool_factory: VisitFactory) -> None:
    async def run() -> None:
        test_name = "visit-missing-document"
        runtime_name, _summary_name = _source_names(test_name)
        tool = tool_factory(test_name=test_name)

        with pytest.raises(RecoverableError, match="Document not found: missing"):
            await tool.run(document_id="missing", source=runtime_name)

    asyncio.run(run())
