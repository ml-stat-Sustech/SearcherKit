from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from searchagent.sources.elasticsearch import ElasticsearchSource


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
