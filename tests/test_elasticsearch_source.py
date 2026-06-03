from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from searchagent.common.retry import RetryConfig
from searchagent.errors import RecoverableError
from searchagent.sources.elasticsearch import ElasticsearchSource, SummaryError
from searchagent.tools import SearchTool, VisitTool


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.documents = [
            {
                "_id": "1",
                "_index": "browsecomp_hybrid",
                "_score": 3.5,
                "_source": {
                    "title": "BrowseComp Plus",
                    "text": "BrowseComp Plus is a benchmark corpus stored in Elasticsearch.",
                    "url": "https://example.test/bcp",
                    "links": [{"text": "Related", "url": "https://example.test/related"}],
                },
                "highlight": {"text": ["BrowseComp Plus benchmark corpus"]},
            }
        ]
        self.search_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_calls.append({"index": index, "body": body})
        term = body.get("query", {}).get("term")
        if isinstance(term, dict):
            field, spec = next(iter(term.items()))
            value = spec.get("value") if isinstance(spec, dict) else spec
            hits = [
                doc
                for doc in self.documents
                if doc["_source"].get(field) == value
            ]
        else:
            hits = self.documents[: body.get("size", 10)]
        return {"hits": {"hits": hits}}

    async def get(self, *, index: str, id: str) -> dict[str, Any]:
        for doc in self.documents:
            if doc["_id"] == id:
                return doc
        raise KeyError(id)

    async def close(self) -> None:
        pass


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


class FakeEmbeddings:
    def __init__(self, client: "FakeEmbeddingClient") -> None:
        self.client = client

    async def create(self, **payload: Any) -> SimpleNamespace:
        self.client.calls.append(payload)
        value = self.client.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    embedding=value,
                )
            ]
        )


class FakeEmbeddingClient:
    def __init__(self, responses: list[list[float] | BaseException]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.embeddings = FakeEmbeddings(self)


def _summary_retry_config(max_tries: int) -> RetryConfig:
    return RetryConfig(
        max_tries=max_tries,
        exceptions=["pkg://searchagent.sources.elasticsearch:SummaryError"],
        jitter=None,
        factor=0,
    )


def _summary_source(
    *,
    summary_client: FakeSummaryClient | None = None,
    client: FakeElasticsearchClient | None = None,
    fetch_by_url: bool = False,
    search_summary_enabled: bool = True,
    fetch_summary_enabled: bool = True,
    summary_max_tokens: int | None = None,
    retry_config: RetryConfig | None = None,
) -> ElasticsearchSource:
    kwargs: dict[str, Any] = {}
    if fetch_by_url:
        kwargs.update(document_id_field="url", fetch_field="url")
    return ElasticsearchSource(
        client=client or FakeElasticsearchClient(),
        index="browsecomp_hybrid",
        search_summary_enabled=search_summary_enabled,
        fetch_summary_enabled=fetch_summary_enabled,
        summary_model="summary-model",
        summary_api_key="key",
        summary_max_tokens=summary_max_tokens,
        summary_client=summary_client,
        summary_retry_config=retry_config,
        **kwargs,
    )


def test_elasticsearch_source_search_and_fetch() -> None:
    async def run_source() -> None:
        client = FakeElasticsearchClient()
        source = ElasticsearchSource(client=client, index="browsecomp_hybrid")

        results = await source.search("benchmark", top_k=1)
        assert results[0].document.id == "1"
        assert results[0].snippet == "BrowseComp Plus benchmark corpus"
        assert results[0].score == 3.5

        document = await source.fetch("1")
        assert document.title == "BrowseComp Plus"
        assert document.url == "https://example.test/bcp"

        url_source = ElasticsearchSource(
            client=client,
            index="browsecomp_hybrid",
            search_fields=["title^2", "text"],
            document_id_field="url",
            fetch_field="url",
            highlight_fragment_size=128,
            snippet_chars=512,
            metadata_fields=["links"],
        )

        url_results = await url_source.search("benchmark", top_k=1)
        assert url_results[0].document.id == "https://example.test/bcp"
        assert url_results[0].snippet == "BrowseComp Plus benchmark corpus"

        url_document = await url_source.fetch("https://example.test/bcp")
        assert url_document.title == "BrowseComp Plus"
        assert url_document.url == "https://example.test/bcp"
        assert url_document.metadata["links"] == [
            {"text": "Related", "url": "https://example.test/related"}
        ]
        assert client.search_calls[-1]["body"]["query"]["term"]["url"]["value"] == "https://example.test/bcp"

    asyncio.run(run_source())


def test_elasticsearch_hybrid_vector_search_uses_embedding_and_keeps_highlight() -> None:
    async def run_source() -> None:
        client = FakeElasticsearchClient()
        embedding_client = FakeEmbeddingClient([[0.1, 0.2, 0.3]])
        source = ElasticsearchSource(
            client=client,
            index="browsecomp_hybrid",
            vector_search_mode="hybrid",
            vector_field="text_vector",
            embedding_model="embedding-model",
            embedding_api_key="key",
            embedding_prefix="query: ",
            embedding_client=embedding_client,
            embedding_default_kwargs={"encoding_format": "float"},
        )

        results = await source.search("benchmark", top_k=2)

        body = client.search_calls[-1]["body"]
        assert results[0].document.id == "1"
        assert body["query"]["multi_match"]["query"] == "benchmark"
        assert body["knn"] == {
            "field": "text_vector",
            "query_vector": [0.1, 0.2, 0.3],
            "k": 2,
            "num_candidates": 2,
        }
        assert "highlight" in body
        assert embedding_client.calls[0]["model"] == "embedding-model"
        assert embedding_client.calls[0]["input"] == "query: benchmark"
        assert embedding_client.calls[0]["encoding_format"] == "float"

    asyncio.run(run_source())


def test_elasticsearch_pure_vector_search_omits_text_query() -> None:
    async def run_source() -> None:
        client = FakeElasticsearchClient()
        source = ElasticsearchSource(
            client=client,
            index="browsecomp_hybrid",
            vector_search_mode="vector",
            highlight=False,
            embedding_model="embedding-model",
            embedding_api_key="key",
            embedding_client=FakeEmbeddingClient([[0.1, 0.2]]),
        )

        await source.search("benchmark", top_k=3)

        body = client.search_calls[-1]["body"]
        assert "query" not in body
        assert "highlight" not in body
        assert body["knn"]["k"] == 3
        assert body["knn"]["num_candidates"] == 3

    asyncio.run(run_source())


def test_elasticsearch_highlight_rejects_pure_vector_search() -> None:
    with pytest.raises(ValueError, match="highlight"):
        ElasticsearchSource(
            client=FakeElasticsearchClient(),
            index="browsecomp_hybrid",
            vector_search_mode="vector",
            highlight=True,
            embedding_model="embedding-model",
            embedding_api_key="key",
            embedding_client=FakeEmbeddingClient([[0.1, 0.2]]),
        )


def test_elasticsearch_search_summary_disabled_returns_normal_results() -> None:
    async def run_source() -> None:
        summary_client = FakeSummaryClient(
            ['{"evidence": "unused evidence", "summary": "unused summary"}']
        )
        source = ElasticsearchSource(
            client=FakeElasticsearchClient(),
            index="browsecomp_hybrid",
            summary_model="summary-model",
            summary_api_key="key",
            summary_client=summary_client,
        )

        results = await source.search("benchmark", top_k=1)

        assert len(results) == 1
        assert results[0].document.id == "1"
        assert results[0].metadata == {"index": "browsecomp_hybrid"}
        assert summary_client.calls == []

    asyncio.run(run_source())


def test_elasticsearch_search_summary_enabled_uses_all_result_text() -> None:
    async def run_source() -> None:
        client = FakeElasticsearchClient()
        client.documents.append(
            {
                "_id": "2",
                "_index": "browsecomp_hybrid",
                "_score": 2.0,
                "_source": {
                    "title": "Second Result",
                    "text": "Second full document text for summary.",
                    "url": "https://example.test/second",
                },
            }
        )
        summary_client = FakeSummaryClient(
            ['{"evidence": "combined evidence", "summary": "combined summary"}']
        )
        source = _summary_source(
            client=client,
            summary_client=summary_client,
            fetch_summary_enabled=False,
        )

        results = await source.search("benchmark", top_k=2)

        assert len(results) == 1
        assert results[0].metadata["summary"] is True
        assert "combined evidence" in (results[0].snippet or "")
        assert "combined summary" in results[0].document.text
        prompt = summary_client.calls[0]["messages"][0]["content"]
        assert "BrowseComp Plus is a benchmark corpus stored in Elasticsearch." in prompt
        assert "Second full document text for summary." in prompt

    asyncio.run(run_source())


def test_elasticsearch_summary_truncates_by_token_limit() -> None:
    class FakeTokenizer:
        def encode(self, content: str, *, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [ord(ch) for ch in content]

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
            assert skip_special_tokens is True
            return "".join(chr(token_id) for token_id in token_ids)

    source = _summary_source(summary_max_tokens=5)
    source._summary_tokenizer = FakeTokenizer()

    assert source._truncate_summary_content("abcdefghij") == "abcde"


def test_elasticsearch_search_summary_retries_once_then_succeeds() -> None:
    async def run_source() -> None:
        summary_client = FakeSummaryClient(
            [
                "not json at all",
                '{"evidence": "after retry", "summary": "ok"}',
            ]
        )
        source = _summary_source(
            summary_client=summary_client,
            fetch_summary_enabled=False,
            retry_config=_summary_retry_config(max_tries=2),
        )

        results = await source.search("benchmark", top_k=1)

        assert len(summary_client.calls) == 2
        assert "after retry" in (results[0].snippet or "")

    asyncio.run(run_source())


def test_elasticsearch_search_summary_failure_surfaces_recoverable_error() -> None:
    async def run_source() -> None:
        source = _summary_source(
            summary_client=FakeSummaryClient(['{"evidence": "", "summary": ""}']),
            fetch_summary_enabled=False,
            retry_config=_summary_retry_config(max_tries=1),
        )
        search_tool = SearchTool(source)

        with pytest.raises(RecoverableError):
            await search_tool._run(query="benchmark", top_k=1)

    asyncio.run(run_source())


def test_elasticsearch_visit_summary_disabled_returns_normal_document() -> None:
    async def run_source() -> None:
        summary_client = FakeSummaryClient(
            ['{"evidence": "unused evidence", "summary": "unused summary"}']
        )
        source = ElasticsearchSource(
            client=FakeElasticsearchClient(),
            index="browsecomp_hybrid",
            document_id_field="url",
            fetch_field="url",
            summary_model="summary-model",
            summary_api_key="key",
            summary_client=summary_client,
        )

        document = await source.fetch("https://example.test/bcp", goal="find benchmark details")

        assert document.title == "BrowseComp Plus"
        assert document.text == "BrowseComp Plus is a benchmark corpus stored in Elasticsearch."
        assert summary_client.calls == []

    asyncio.run(run_source())


def test_elasticsearch_visit_summary_enabled_uses_goal() -> None:
    async def run_source() -> None:
        summary_client = FakeSummaryClient(
            ['{"evidence": "goal evidence", "summary": "goal summary"}']
        )
        source = _summary_source(
            summary_client=summary_client,
            fetch_by_url=True,
            search_summary_enabled=False,
        )

        visit_tool = VisitTool(source)
        payload = await visit_tool.run(
            document_id="https://example.test/bcp",
            goal="find benchmark details",
        )

        assert "goal evidence" in payload
        assert "goal summary" in payload
        prompt = summary_client.calls[0]["messages"][0]["content"]
        assert "find benchmark details" in prompt

    asyncio.run(run_source())


def test_elasticsearch_visit_summary_retries_once_then_succeeds() -> None:
    async def run_source() -> None:
        summary_client = FakeSummaryClient(
            [
                "not json at all",
                '{"evidence": "visit retry evidence", "summary": "visit retry summary"}',
            ]
        )
        source = _summary_source(
            summary_client=summary_client,
            fetch_by_url=True,
            search_summary_enabled=False,
            retry_config=_summary_retry_config(max_tries=2),
        )

        document = await source.fetch("https://example.test/bcp", goal="find benchmark details")

        assert len(summary_client.calls) == 2
        assert "visit retry evidence" in document.text
        assert "visit retry summary" in document.text

    asyncio.run(run_source())


def test_elasticsearch_visit_summary_failure_surfaces_recoverable_error() -> None:
    async def run_source() -> None:
        source = _summary_source(
            summary_client=FakeSummaryClient(['{"evidence": "", "summary": ""}']),
            fetch_by_url=True,
            search_summary_enabled=False,
            retry_config=_summary_retry_config(max_tries=1),
        )
        visit_tool = VisitTool(source)

        with pytest.raises(RecoverableError):
            await visit_tool._run(
                document_id="https://example.test/bcp",
                goal="find benchmark details",
            )

    asyncio.run(run_source())


def test_elasticsearch_source_summary_requires_model_and_key() -> None:
    with pytest.raises(ValueError, match="summary_model"):
        ElasticsearchSource(
            client=FakeElasticsearchClient(),
            index="browsecomp_hybrid",
            search_summary_enabled=True,
            summary_api_key="key",
        )

    with pytest.raises(ValueError, match="summary_api_key"):
        ElasticsearchSource(
            client=FakeElasticsearchClient(),
            index="browsecomp_hybrid",
            fetch_summary_enabled=True,
            summary_model="summary-model",
        )
