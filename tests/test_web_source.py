from __future__ import annotations

import asyncio
from typing import Any

import searchagent.sources.web as web_module
from searchagent.sources import SourceConfig, build_source
from searchagent.sources.web import WebSource


class FakeResponse:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.raise_for_status_called = False

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True

    async def json(self) -> Any:
        return self.result

    async def text(self) -> str:
        return str(self.result)


class FakeClientSession:
    instances: list["FakeClientSession"] = []

    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.closed = False
        self.close_count = 0
        FakeClientSession.instances.append(self)

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.post_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "organic": [
                    {
                        "title": "SearchAgent",
                        "link": "https://example.test/searchagent",
                        "snippet": "A pluggable search-agent runtime.",
                        "position": 1,
                        "date": "Jul 2, 2026",
                    }
                ]
            }
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.get_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return FakeResponse("Full page text from Jina Reader.")

    async def close(self) -> None:
        self.close_count += 1
        self.closed = True


def test_web_source_search_and_fetch(monkeypatch: Any) -> None:
    FakeClientSession.instances = []
    monkeypatch.setattr(web_module, "ClientSession", FakeClientSession)

    async def run_source() -> None:
        source = WebSource(
            serper_api_key="serper-key",
            serper_timeout=12,
            jina_api_key="jina-key",
            jina_base_url="https://r.jina.ai/",
            jina_timeout=20,
        )

        results = await source.search("searchagent", top_k=3)
        client = FakeClientSession.instances[0]

        assert results[0].document.id == "https://example.test/searchagent"
        assert results[0].document.title == "SearchAgent"
        assert results[0].snippet == "A pluggable search-agent runtime."
        assert results[0].score is None
        assert results[0].document.metadata["date"] == "Jul 2, 2026"
        assert client.post_calls[0]["url"] == "https://google.serper.dev/search"
        assert client.post_calls[0]["headers"]["X-API-KEY"] == "serper-key"
        assert client.post_calls[0]["json"] == {
            "q": "searchagent",
            "num": 3,
        }
        assert client.post_calls[0]["timeout"].total == 12

        document = await source.fetch("https://example.test/searchagent")

        assert document.id == "https://example.test/searchagent"
        assert document.url == "https://example.test/searchagent"
        assert document.text == "Full page text from Jina Reader."
        assert client.get_calls[0]["url"] == "https://r.jina.ai/https://example.test/searchagent"
        assert client.get_calls[0]["headers"]["Authorization"] == "Bearer jina-key"
        assert client.get_calls[0]["headers"]["Accept"] == "text/plain"
        assert client.get_calls[0]["timeout"].total == 20

        await source.close()
        assert client.closed

    asyncio.run(run_source())


def test_web_source_reuses_owned_session(monkeypatch: Any) -> None:
    FakeClientSession.instances = []
    monkeypatch.setattr(web_module, "ClientSession", FakeClientSession)

    async def run_source() -> None:
        source = WebSource(serper_api_key="serper-key")

        await source.search("first")
        await source.search("second")

        assert len(FakeClientSession.instances) == 1
        client = FakeClientSession.instances[0]
        assert len(client.post_calls) == 2

        await source.close()
        assert client.closed
        assert client.close_count == 1

    asyncio.run(run_source())


def test_web_source_builds_from_source_config() -> None:
    source = build_source(
        config=SourceConfig(
            type="web",
            serper_api_key="serper-key",
        )
    )

    assert isinstance(source, WebSource)

    asyncio.run(source.close())
