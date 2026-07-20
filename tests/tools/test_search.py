from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from searcherkit.common.retry import RetryConfig, RetryPolicy
from searcherkit.common.errors import RecoverableError
from searcherkit.sources import SourceConfig, add_source_cfg
from searcherkit.sources.file import FileSource
from searcherkit.tools import SummarizerConfig, ToolConfig, build_tool
from searcherkit.tools.search import SearchTool
from searcherkit.tools.summarizer import Summarizer


BASE_URL = "https://example.test/v1"
SUMMARY_EVIDENCE = "SearcherKit evidence"
SUMMARY_TEXT = "SearcherKit summary"

ToolFactory = Callable[..., SearchTool]
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "files"
DOC_ID = "source_files.md"


def _expected_search_extensions(query: str = "source-backed-tools") -> dict[str, object]:
    return {
        "searched_ids": [DOC_ID],
        "documents": [
            {
                "id": DOC_ID,
                "title": DOC_ID,
                "url": None,
                "query": query,
            }
        ],
    }


def _summary_response(*, evidence: str, summary: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-summary-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "fake-summary-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"evidence": evidence, "summary": summary}
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _invalid_summary_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-summary-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "fake-summary-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "not json at all"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@contextmanager
def _summary_context(
    captured_payload: dict[str, Any] | None = None,
    *,
    assert_all_called: bool = True,
    outcomes: list[str] | tuple[str, ...] = ("success",),
) -> Iterator[Any]:
    remaining = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        if captured_payload is not None:
            captured_payload.clear()
            captured_payload.update(json.loads(request.content))
        outcome = remaining.pop(0) if remaining else outcomes[-1]
        if outcome == "success":
            return _summary_response(
                evidence=SUMMARY_EVIDENCE,
                summary=SUMMARY_TEXT,
            )
        if outcome == "retry_success":
            return _summary_response(evidence="after retry", summary="ok")
        if outcome == "invalid":
            return _invalid_summary_response()
        raise ValueError(f"Unknown mock outcome: {outcome}")

    with respx.mock(assert_all_called=assert_all_called) as router:
        route = router.post(f"{BASE_URL}/chat/completions").mock(side_effect=handler)
        yield route


def _source_name(test_name: str) -> str:
    return f"tools-search-{test_name}"


def _add_file_source(name: str) -> None:
    add_source_cfg(
        name,
        SourceConfig(type="file", name=name, root_path=str(SOURCE_ROOT)),
    )


def _retry_config(max_tries: int) -> RetryConfig:
    return RetryConfig(
        max_tries=max_tries,
        exceptions=["pkg://searcherkit.tools.summarizer:SummaryError"],
        jitter=None,
        factor=0.0,
    )


def _direct_tool(
    *,
    test_name: str,
    with_summary: bool = False,
    retry_config: RetryConfig | None = None,
) -> SearchTool:
    summarizer = None
    if with_summary:
        summarizer = Summarizer(
            model="fake-summary-model",
            api_key="test-key",
            base_url=BASE_URL,
            timeout=1,
            retry_policy=(
                RetryPolicy(config=retry_config)
                if retry_config is not None
                else None
            ),
        )

    return SearchTool(
        FileSource(root_path=SOURCE_ROOT),
        name="search",
        summarizer=summarizer,
    )


def _config_tool(
    *,
    test_name: str,
    with_summary: bool = False,
    retry_config: RetryConfig | None = None,
) -> SearchTool:
    source_name = _source_name(test_name)
    _add_file_source(source_name)
    tool = build_tool(
            ToolConfig(
                type="search",
                name="search",
                source=[source_name],
                summarizer=(
                SummarizerConfig(
                    model="fake-summary-model",
                    api_key="test-key",
                    base_url=BASE_URL,
                    timeout=1,
                    retry_config=retry_config,
                )
                if with_summary
                else None
            ),
            summary_goal_key="query",
        ),
    )
    assert isinstance(tool, SearchTool)
    return tool


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_run(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        tool = tool_factory(test_name="run")
        assert "source" not in tool.inputSchema["properties"]

        result = await tool.run(query="source-backed-tools", top_k=1)
        content, extensions = result

        assert f"1. [{DOC_ID}](None)" in content
        assert "Summary Tools" not in content
        assert extensions == _expected_search_extensions()

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_summary(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        captured_payload: dict[str, Any] = {}
        with _summary_context(captured_payload) as route:
            tool = tool_factory(test_name="summary", with_summary=True)
            assert tool.summary_enabled

            result = await tool.run(query="source-backed-tools", top_k=1)

        assert len(route.calls) == 1, result
        content, extensions = result
        assert captured_payload["model"] == "fake-summary-model"
        assert captured_payload["response_format"] == {"type": "json_object"}
        prompt = captured_payload["messages"][0]["content"]
        assert "source-backed-tools" in prompt
        assert DOC_ID in prompt
        assert content == (
            "The useful information for query source-backed-tools as follows:\n\n"
            "Evidence in page:\n"
            f"{SUMMARY_EVIDENCE}\n\n"
            "Summary:\n"
            f"{SUMMARY_TEXT}"
        )
        assert extensions == _expected_search_extensions()

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_summary_retry_success(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        with _summary_context(outcomes=("invalid", "retry_success")) as route:
            tool = tool_factory(
                test_name="summary-retry-success",
                with_summary=True,
                retry_config=_retry_config(max_tries=2),
            )
            assert tool.summary_enabled

            result = await tool.run(query="source-backed-tools", top_k=1)

        assert len(route.calls) == 2
        content, extensions = result
        assert "after retry" in content
        assert "ok" in content
        assert extensions == _expected_search_extensions()

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_summary_retry_failure(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        with _summary_context(
            outcomes=("invalid", "invalid"),
            assert_all_called=False,
        ) as route:
            tool = tool_factory(
                test_name="summary-retry-failure",
                with_summary=True,
                retry_config=_retry_config(max_tries=2),
            )
            assert tool.summary_enabled

            with pytest.raises(RecoverableError):
                await tool.run(query="source-backed-tools", top_k=1)

        assert len(route.calls) == 2

    asyncio.run(run())
