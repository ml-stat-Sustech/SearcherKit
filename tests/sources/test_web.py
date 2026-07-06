import asyncio
from collections.abc import Callable

import pytest
from aioresponses import aioresponses
from yarl import URL

from searchagent.common.retry import RetryConfig
from searchagent.errors import SourceError
from searchagent.sources import SourceConfig, build_source
from searchagent.sources.base import DataSource
from searchagent.sources.web import WebSource


SERPER_URL = "https://google.serper.dev/search"
JINA_URL = "https://r.jina.ai/https://example.test/searchagent"
RETRY_CONFIG = RetryConfig(
    max_tries=2,
    exceptions=["pkg://builtins:TimeoutError"],
    jitter=None,
    factor=0.0,
)


def _search_payload() -> dict[str, object]:
    return {
        "organic": [
            {
                "title": "SearchAgent",
                "link": "https://example.test/searchagent",
                "snippet": "A pluggable search-agent runtime.",
                "position": 1,
            }
        ]
    }


def _direct_source(*, retry_config: RetryConfig | None = None) -> DataSource:
    return WebSource(
        serper_api_key="serper-key",
        serper_endpoint=SERPER_URL,
        jina_api_key="jina-key",
        jina_base_url="https://r.jina.ai/",
        web_retry_config=retry_config,
    )


def _config_source(*, retry_config: RetryConfig | None = None) -> DataSource:
    return build_source(
        config=SourceConfig(
            type="web",
            serper_api_key="serper-key",
            serper_endpoint=SERPER_URL,
            jina_api_key="jina-key",
            jina_base_url="https://r.jina.ai/",
            web_retry_config=retry_config,
        )
    )


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_search(source_factory: Callable[..., DataSource]) -> None:
    async def run() -> None:
        with aioresponses() as mocked:
            mocked.post(SERPER_URL, payload=_search_payload())
            source = source_factory()

            results = await source.search("searchagent", top_k=2)
            await source.close()

        request = mocked.requests[("POST", URL(SERPER_URL))][0]
        assert request.kwargs["headers"]["X-API-KEY"] == "serper-key"
        assert request.kwargs["headers"]["Content-Type"] == "application/json"
        assert request.kwargs["json"] == {"q": "searchagent", "num": 2}
        assert request.kwargs["timeout"].total == 30
        assert results[0].document.id == "https://example.test/searchagent"
        assert results[0].document.title == "SearchAgent"
        assert results[0].document.url == "https://example.test/searchagent"
        assert results[0].snippet == "A pluggable search-agent runtime."
        assert results[0].document.metadata == {"position": 1}
        assert results[0].metadata == {"source": "serper"}

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_fetch(source_factory: Callable[..., DataSource]) -> None:
    async def run() -> None:
        with aioresponses() as mocked:
            mocked.get(JINA_URL, body="Full page text from Jina Reader.")
            source = source_factory()

            document = await source.fetch("https://example.test/searchagent")
            await source.close()

        request = mocked.requests[("GET", URL(JINA_URL))][0]
        assert request.kwargs["headers"]["Authorization"] == "Bearer jina-key"
        assert request.kwargs["headers"]["Accept"] == "text/plain"
        assert request.kwargs["timeout"].total == 60
        assert document.id == "https://example.test/searchagent"
        assert document.url == "https://example.test/searchagent"
        assert document.text == "Full page text from Jina Reader."
        assert document.metadata == {"source": "jina"}

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_retry_success(source_factory: Callable[..., DataSource]) -> None:
    async def run() -> None:
        source = source_factory(retry_config=RETRY_CONFIG)
        try:
            with aioresponses() as mocked:
                mocked.post(SERPER_URL, exception=TimeoutError("serper timeout"))
                mocked.post(SERPER_URL, payload=_search_payload())

                results = await source.search("searchagent", top_k=2)

                assert len(mocked.requests[("POST", URL(SERPER_URL))]) == 2
                assert results[0].document.title == "SearchAgent"
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_retry_failure(source_factory: Callable[..., DataSource]) -> None:
    async def run() -> None:
        source = source_factory(retry_config=RETRY_CONFIG)
        try:
            with aioresponses() as mocked:
                mocked.get(JINA_URL, exception=TimeoutError("jina timeout"))
                mocked.get(JINA_URL, exception=TimeoutError("jina timeout"))

                with pytest.raises(SourceError, match="failed to fetch web document"):
                    await source.fetch("https://example.test/searchagent")

                assert len(mocked.requests[("GET", URL(JINA_URL))]) == 2
        finally:
            await source.close()

    asyncio.run(run())
