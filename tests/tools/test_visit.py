from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import respx

from searchagent.common.retry import RetryConfig, RetryPolicy
from searchagent.common.errors import RecoverableError
from searchagent.sources import Document, SourceConfig, add_source_cfg
from searchagent.sources.memory import MemorySource
from searchagent.tools import SummarizerConfig, ToolConfig, build_tool
from searchagent.tools.summarizer import Summarizer
from searchagent.tools.visit import VisitTool


BASE_URL = "https://example.test/v1"
SUMMARY_EVIDENCE = "Visit evidence"
SUMMARY_TEXT = "Visit summary"

ToolFactory = Callable[..., VisitTool]


def _summary_response(*, evidence: str, summary: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-visit-summary-test",
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
            "id": "chatcmpl-visit-summary-test",
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


def _documents() -> list[Document]:
    return [
        Document(
            id="doc-1",
            title="SearchAgent Runtime",
            text="SearchAgent provides a pluggable runtime for source-backed tools.",
            url="https://example.test/runtime",
        ),
        Document(
            id="doc-2",
            title="Summary Tools",
            text="Summary tools produce evidence and concise answers from search output.",
            url="https://example.test/summary",
        ),
    ]


def _source_name(test_name: str) -> str:
    return f"tools-visit-{test_name}"


def _add_memory_source(name: str) -> None:
    add_source_cfg(
        name,
        SourceConfig(type="memory", name=name, documents=_documents()),
    )


def _retry_config(max_tries: int) -> RetryConfig:
    return RetryConfig(
        max_tries=max_tries,
        exceptions=["pkg://searchagent.tools.summarizer:SummaryError"],
        jitter=None,
        factor=0.0,
    )


def _direct_tool(
    *,
    test_name: str,
    with_summary: bool = False,
    retry_config: RetryConfig | None = None,
) -> VisitTool:
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

    return VisitTool(
        MemorySource(documents=_documents()),
        name="visit",
        summarizer=summarizer,
    )


def _config_tool(
    *,
    test_name: str,
    with_summary: bool = False,
    retry_config: RetryConfig | None = None,
) -> VisitTool:
    source_name = _source_name(test_name)
    _add_memory_source(source_name)
    tool = build_tool(
            ToolConfig(
                type="visit",
                name="visit",
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
            summary_goal_key="goal",
        ),
    )
    assert isinstance(tool, VisitTool)
    return tool


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_run(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        tool = tool_factory(test_name="run")

        result = await tool.run(document_id="doc-1", goal="confirm runtime wiring")

        assert "[SearchAgent Runtime](https://example.test/runtime)" in result
        assert "pluggable runtime for source-backed tools" in result

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_missing_document_surfaces_recoverable_error(
    tool_factory: ToolFactory,
) -> None:
    async def run() -> None:
        tool = tool_factory(test_name="missing-document")

        with pytest.raises(RecoverableError, match="Document not found: missing"):
            await tool.run(document_id="missing", goal="confirm runtime wiring")

    asyncio.run(run())


@pytest.mark.parametrize("tool_factory", [_direct_tool, _config_tool])
def test_summary(tool_factory: ToolFactory) -> None:
    async def run() -> None:
        captured_payload: dict[str, Any] = {}
        with _summary_context(captured_payload) as route:
            tool = tool_factory(test_name="summary", with_summary=True)
            assert tool.summary_enabled

            result = await tool.run(document_id="doc-1", goal="confirm runtime wiring")

        assert len(route.calls) == 1, result
        prompt = captured_payload["messages"][0]["content"]
        assert "confirm runtime wiring" in prompt
        assert "SearchAgent provides a pluggable runtime for source-backed tools." in prompt
        assert result == (
            "The useful information for query confirm runtime wiring as follows:\n\n"
            "Evidence in page:\n"
            f"{SUMMARY_EVIDENCE}\n\n"
            "Summary:\n"
            f"{SUMMARY_TEXT}"
        )

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

            result = await tool.run(document_id="doc-1", goal="confirm runtime wiring")

        assert len(route.calls) == 2
        assert "after retry" in result
        assert "ok" in result

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
                await tool.run(document_id="doc-1", goal="confirm runtime wiring")

        assert len(route.calls) == 2

    asyncio.run(run())
