from __future__ import annotations

import asyncio

from searchagent.common.messages import ToolCall
from searchagent.llm.parsers import ParserConfig, get_parser
from searchagent.llm.parsers.openseeker import OpenSeekerParser
from searchagent.sources import SourceConfig, add_source_cfg
from searchagent.sources.base import Document
from searchagent.tools import ToolConfig, build_tool


def test_openseeker_tools_accept_batched_search_and_url_visit() -> None:
    async def run_tools() -> None:
        add_source_cfg(
            "openseeker-memory",
            SourceConfig(
                name="openseeker-memory",
                type="memory",
                documents=[
                    Document(
                        id="localwiki://example",
                        title="Example Book",
                        url="localwiki://example",
                        text="Example Book was written by Ada Example.",
                    ),
                    Document(
                        id="localwiki://other",
                        title="Other",
                        url="localwiki://other",
                        text="Unrelated content.",
                    ),
                ],
            ),
        )

        search_tool = build_tool(
            ToolConfig(type="openseeker_search", name="search", source="openseeker-memory")
        )
        visit_tool = build_tool(
            ToolConfig(type="openseeker_visit", name="visit", source="openseeker-memory")
        )

        assert search_tool.inputSchema["properties"]["query"]["type"] == "array"
        assert visit_tool.inputSchema["properties"]["url"]["type"] == ["string", "array"]

        search_payload = await search_tool.run(query=["Example Book author"], top_k=1)
        assert "A Local wiki search for 'Example Book author' found 1 results:" in search_payload
        assert "[Example Book](localwiki://example)" in search_payload

        visit_payload = await visit_tool.run(
            url="localwiki://example",
            goal="identify the author",
        )
        assert "# [Example Book](localwiki://example)" in visit_payload
        assert "Ada Example" in visit_payload

    asyncio.run(run_tools())


def test_openseeker_parser_accepts_tool_calls_wrapper() -> None:
    parser = OpenSeekerParser(drop_thinking=False)

    parsed = parser.from_assistant(
        {
            "role": "assistant",
            "content": (
                "<think>Need evidence.</think>\n"
                "<tool_calls_begin>\n"
                "<tool_call>\n"
                '{"name":"search","arguments":{"query":["Example Book author"]}}\n'
                "</tool_call>\n"
                "</tool_calls_end>"
            ),
        }
    )

    assert parsed.thinking == "Need evidence."
    assert parsed.content is None
    assert parsed.tool_calls == [
        ToolCall(
            id="chatcmpl-tool-0",
            name="search",
            arguments={"query": ["Example Book author"]},
        )
    ]


def test_openseeker_parser_renders_wrapper_for_history_tool_calls() -> None:
    parser = OpenSeekerParser(drop_thinking=False)

    rendered = parser.render_tool_calls(
        [
            ToolCall(
                id="call-1",
                name="visit",
                arguments={"url": "localwiki://example", "goal": "identify author"},
            )
        ]
    )

    assert "<tool_calls_begin>" not in rendered
    assert "<tool_call>" in rendered
    assert '"name": "visit"' in rendered


def test_parser_factory_builds_openseeker_parser() -> None:
    parser = get_parser(ParserConfig(type="openseeker"))

    assert isinstance(parser, OpenSeekerParser)
