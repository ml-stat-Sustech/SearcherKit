from __future__ import annotations

import asyncio
import json

from searchagent.sources import build_source, SourceConfig
from searchagent.sources.base import Document
from searchagent.sources.memory import MemorySource
from searchagent.tools import ToolConfig, build_tool


def test_source_backed_search_and_visit_tools_share_named_source() -> None:
    async def run_tools() -> None:
        source = build_source(
            SourceConfig(
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
            )
        )

        sources = {"memory": source}

        search_tool = build_tool(
            ToolConfig(type="search", name="search", source="memory"),
            sources=sources,
        )
        visit_tool = build_tool(
            ToolConfig(type="visit", name="visit", source="memory"),
            sources=sources,
        )

        search_payload = json.loads(await search_tool.run(query="wires sources", top_k=1))
        assert search_payload[0]["document"]["id"] == "doc-1"
        assert "text" not in search_payload[0]["document"]

        visit_payload = json.loads(await visit_tool.run(document_id="doc-1"))
        assert visit_payload["title"] == "SearchAgent"
        assert visit_payload["text"] == "SearchAgent wires tools to data sources."

    asyncio.run(run_tools())


def test_build_source_accepts_mapping() -> None:
    source = build_source(
        SourceConfig(
            type="memory",
            documents=[
                Document(
                    id="doc-1",
                    title="Configured object",
                    text="Existing source instances can be reused.",
                )
            ],
        )
    )

    assert source is not None


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


def test_factory_infers_legacy_endpoint_tools_as_mcp() -> None:
    tool = build_tool(
        ToolConfig(
            name="search",
            mcp_tool_name="remote_search",
            endpoint="http://127.0.0.1:8100/mcp",
        )
    )

    assert tool.name == "search"
    assert getattr(tool, "mcp_tool_name") == "remote_search"
