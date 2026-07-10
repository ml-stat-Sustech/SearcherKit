from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from searchagent.common.errors import RecoverableError
from searchagent.sources import SourceConfig, add_source_cfg
from searchagent.sources.local_file import LocalFileSource
from searchagent.tools import ToolConfig, build_tool
from searchagent.tools.multi_source_search import MultiSourceSearchTool
from searchagent.tools.multi_source_visit import MultiSourceVisitTool


SearchFactory = Callable[..., MultiSourceSearchTool]
VisitFactory = Callable[..., MultiSourceVisitTool]
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "source_files"
RUNTIME_ROOT = FIXTURE_ROOT / "runtime"
SUMMARY_ROOT = FIXTURE_ROOT / "summary"
RUNTIME_DOC_ID = "runtime-doc.md"
SUMMARY_DOC_ID = "summary-doc.md"


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
            type="local_file",
            name=runtime_name,
            root_path=str(RUNTIME_ROOT),
        ),
    )
    add_source_cfg(
        summary_name,
        SourceConfig(
            type="local_file",
            name=summary_name,
            root_path=str(SUMMARY_ROOT),
        ),
    )
    return runtime_name, summary_name


def _direct_search_tool(*, test_name: str) -> MultiSourceSearchTool:
    runtime_name, summary_name = _source_names(test_name)
    return MultiSourceSearchTool(
        {
            runtime_name: LocalFileSource(root_path=RUNTIME_ROOT),
            summary_name: LocalFileSource(root_path=SUMMARY_ROOT),
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
            runtime_name: LocalFileSource(root_path=RUNTIME_ROOT),
            summary_name: LocalFileSource(root_path=SUMMARY_ROOT),
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
            query="runtime-doc",
            source=runtime_name,
        )
        summary_result = await tool.run(
            query="summary-doc",
            source=summary_name,
        )

        runtime_content, runtime_extensions = runtime_result
        summary_content, summary_extensions = summary_result
        assert RUNTIME_DOC_ID in runtime_content
        assert (
            "Summary source explains evidence extraction."
            not in runtime_content
        )
        assert SUMMARY_DOC_ID in summary_content
        assert runtime_extensions == {"searched_ids": [RUNTIME_DOC_ID]}
        assert summary_extensions == {"searched_ids": [SUMMARY_DOC_ID]}

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
            document_id=RUNTIME_DOC_ID,
            source=runtime_name,
            goal="confirm runtime",
        )
        summary_result = await tool.run(
            document_id=SUMMARY_DOC_ID,
            source=summary_name,
            goal="confirm summary",
        )

        runtime_content, runtime_extensions = runtime_result
        summary_content, summary_extensions = summary_result
        assert f"[{RUNTIME_DOC_ID}](None)" in runtime_content
        assert (
            "Runtime source explains pluggable source-backed execution."
            in runtime_content
        )
        assert f"[{SUMMARY_DOC_ID}](None)" in summary_content
        assert "Summary source explains evidence extraction." in summary_content
        assert runtime_extensions == {}
        assert summary_extensions == {}

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

        with pytest.raises(RecoverableError, match="local file document not found"):
            await tool.run(document_id="missing", source=runtime_name)

    asyncio.run(run())
