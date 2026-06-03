from __future__ import annotations

import asyncio
from searchagent.sources import SourceConfig, add_source_cfg
from searchagent.sources.base import Document
from searchagent.tools import ToolConfig, build_tool


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

        search_payload = await search_tool.run(query_list=["wires sources"], top_k=1)
        assert "[SearchAgent]" in search_payload
        assert "SearchAgent wires tools to data sources." in search_payload

        visit_payload = await visit_tool.run(document_id="doc-1")
        assert "[SearchAgent]" in visit_payload
        assert "SearchAgent wires tools to data sources." in visit_payload

        visit_with_goal = await visit_tool.run(document_id="doc-1", goal="confirm wiring")
        assert "SearchAgent wires tools to data sources." in visit_with_goal

    asyncio.run(run_tools())


def test_source_backed_search_accepts_multiple_queries() -> None:
    async def run_tool() -> None:
        add_source_cfg(
            "multi-query-memory",
            SourceConfig(
                name="multi-query-memory",
                type="memory",
                documents=[
                    Document(
                        id="doc-1",
                        title="Alpha",
                        text="Alpha topic evidence.",
                    ),
                    Document(
                        id="doc-2",
                        title="Beta",
                        text="Beta topic evidence.",
                    ),
                ],
            ),
        )

        search_tool = build_tool(
            ToolConfig(type="search", name="search", source="multi-query-memory"),
        )

        search_payload = await search_tool.run(query_list=["alpha topic", "beta topic"], top_k=1)

        assert search_payload.count("\n-*-*-\n") == 1
        alpha_payload, beta_payload = search_payload.split("\n-*-*-\n")
        assert "[Alpha]" in alpha_payload
        assert "Alpha topic evidence." in alpha_payload
        assert "[Beta]" in beta_payload
        assert "Beta topic evidence." in beta_payload

    asyncio.run(run_tool())


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


