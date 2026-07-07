import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
import respx
from aioresponses import aioresponses
from yarl import URL

from searchagent.common.retry import RetryConfig
from searchagent.sources import DataSource, SourceConfig, SourceError, build_source
from searchagent.sources.elasticsearch import ElasticsearchSource


BASE_URL = "http://localhost:9200"
EMBEDDING_BASE_URL = "https://embedding.example.test/v1"
INDEX = "docs"
EMBEDDINGS_URL = f"{EMBEDDING_BASE_URL}/embeddings"
EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_tries=2,
    exceptions=["pkg://openai:APITimeoutError"],
    jitter=None,
    factor=0.0,
)


def _direct_source() -> DataSource:
    return ElasticsearchSource(
        hosts=BASE_URL,
        index=INDEX,
    )


def _config_source() -> DataSource:
    return build_source(
        config=SourceConfig(
            type="elasticsearch",
            hosts=BASE_URL,
            index=INDEX,
        )
    )


def _direct_url_source() -> DataSource:
    return ElasticsearchSource(
        hosts=BASE_URL,
        index=INDEX,
        document_id_field="url",
        fetch_field="url",
        metadata_fields=["links"],
    )


def _config_url_source() -> DataSource:
    return build_source(
        config=SourceConfig(
            type="elasticsearch",
            hosts=BASE_URL,
            index=INDEX,
            document_id_field="url",
            fetch_field="url",
            metadata_fields=["links"],
        )
    )


def _direct_hybrid_source(*, retry_config: RetryConfig | None = None) -> DataSource:
    return ElasticsearchSource(
        hosts=BASE_URL,
        index=INDEX,
        vector_search_mode="hybrid",
        vector_field="text_vector",
        embedding_model="embedding-model",
        embedding_api_key="test-key",
        embedding_base_url=EMBEDDING_BASE_URL,
        embedding_prefix="query: ",
        embedding_default_kwargs={"encoding_format": "float"},
        embedding_retry_config=retry_config,
    )


def _config_hybrid_source(*, retry_config: RetryConfig | None = None) -> DataSource:
    return build_source(
        config=SourceConfig(
            type="elasticsearch",
            hosts=BASE_URL,
            index=INDEX,
            vector_search_mode="hybrid",
            vector_field="text_vector",
            embedding_model="embedding-model",
            embedding_api_key="test-key",
            embedding_base_url=EMBEDDING_BASE_URL,
            embedding_prefix="query: ",
            embedding_default_kwargs={"encoding_format": "float"},
            embedding_retry_config=retry_config,
        )
    )


def _direct_vector_source() -> DataSource:
    return ElasticsearchSource(
        hosts=BASE_URL,
        index=INDEX,
        vector_search_mode="vector",
        highlight=False,
        vector_field="text_vector",
        embedding_model="embedding-model",
        embedding_api_key="test-key",
        embedding_base_url=EMBEDDING_BASE_URL,
    )


def _config_vector_source() -> DataSource:
    return build_source(
        config=SourceConfig(
            type="elasticsearch",
            hosts=BASE_URL,
            index=INDEX,
            vector_search_mode="vector",
            highlight=False,
            vector_field="text_vector",
            embedding_model="embedding-model",
            embedding_api_key="test-key",
            embedding_base_url=EMBEDDING_BASE_URL,
        )
    )


def _embedding_response() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        "model": "embedding-model",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def _search_response() -> dict[str, object]:
    return {
        "hits": {
            "hits": [
                {
                    "_index": INDEX,
                    "_id": "doc-1",
                    "_score": 1.5,
                    "_source": {
                        "title": "SearchAgent",
                        "text": "pluggable runtime",
                        "url": "https://example.test/doc-1",
                        "links": ["https://example.test/next"],
                    },
                    "highlight": {"text": ["pluggable <em>runtime</em>"]},
                }
            ]
        }
    }


def _fetch_response() -> dict[str, object]:
    return {
        "_index": INDEX,
        "_id": "doc-1",
        "_source": {
            "title": "Fetched",
            "text": "full document text",
            "url": "https://example.test/fetched",
        },
    }


def _elastic_headers() -> dict[str, str]:
    return {"x-elastic-product": "Elasticsearch"}


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_search(source_factory: Callable[[], DataSource]) -> None:
    search_url = f"{BASE_URL}/{INDEX}/_search"

    async def run() -> None:
        source = source_factory()
        try:
            with aioresponses() as mocked:
                mocked.post(search_url, payload=_search_response(), headers=_elastic_headers())

                results = await source.search("runtime", top_k=3)

                request = mocked.requests[("POST", URL(search_url))][0]
                body = json.loads(request.kwargs["data"])
                assert body["size"] == 3
                assert body["query"]["multi_match"]["query"] == "runtime"
                assert body["highlight"] == {
                    "fields": {"text": {"fragment_size": 256, "number_of_fragments": 5}}
                }
                assert results[0].document.id == "doc-1"
                assert results[0].document.title == "SearchAgent"
                assert results[0].document.metadata == {"links": ["https://example.test/next"]}
                assert results[0].snippet == "pluggable runtime"
                assert results[0].score == 1.5
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_url_source, _config_url_source])
def test_fetch_by_field(source_factory: Callable[[], DataSource]) -> None:
    search_url = f"{BASE_URL}/{INDEX}/_search"
    document_url = "https://example.test/doc-1"

    async def run() -> None:
        source = source_factory()
        try:
            with aioresponses() as mocked:
                mocked.post(search_url, payload=_search_response(), headers=_elastic_headers())

                document = await source.fetch(document_url)

                request = mocked.requests[("POST", URL(search_url))][0]
                body = json.loads(request.kwargs["data"])
                assert body == {
                    "size": 1,
                    "query": {"term": {"url": {"value": document_url}}},
                }
                assert document.id == document_url
                assert document.url == document_url
                assert document.title == "SearchAgent"
                assert document.metadata == {"links": ["https://example.test/next"]}
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_hybrid_source, _config_hybrid_source])
def test_hybrid_search_with_highlight(source_factory: Callable[..., DataSource]) -> None:
    search_url = f"{BASE_URL}/{INDEX}/_search"

    async def run() -> None:
        source = source_factory()
        try:
            with (
                respx.mock(assert_all_called=True) as embedding_router,
                aioresponses() as mocked,
            ):
                embedding_route = embedding_router.post(EMBEDDINGS_URL).mock(
                    return_value=httpx.Response(200, json=_embedding_response())
                )
                mocked.post(search_url, payload=_search_response(), headers=_elastic_headers())

                results = await source.search("runtime", top_k=2)

                embedding_payload = json.loads(embedding_route.calls[0].request.content)
                assert embedding_payload["model"] == "embedding-model"
                assert embedding_payload["input"] == "query: runtime"
                assert embedding_payload["encoding_format"] == "float"

                request = mocked.requests[("POST", URL(search_url))][0]
                body = json.loads(request.kwargs["data"])
                assert body["query"]["multi_match"]["query"] == "runtime"
                assert body["knn"] == {
                    "field": "text_vector",
                    "query_vector": [0.1, 0.2, 0.3],
                    "k": 2,
                    "num_candidates": 2,
                }
                assert body["highlight"] == {
                    "fields": {"text": {"fragment_size": 256, "number_of_fragments": 5}}
                }
                assert results[0].document.id == "doc-1"
                assert results[0].document.title == "SearchAgent"
                assert results[0].snippet == "pluggable runtime"
                assert results[0].score == 1.5
        finally:
            await source.close()

    asyncio.run(run())


def test_vector_search_rejects_highlight() -> None:
    with pytest.raises(ValueError, match="highlight"):
        ElasticsearchSource(
            hosts=BASE_URL,
            index=INDEX,
            vector_search_mode="vector",
            highlight=True,
            embedding_model="embedding-model",
            embedding_api_key="test-key",
        )

    with pytest.raises(ValueError, match="highlight"):
        build_source(
            config=SourceConfig(
                type="elasticsearch",
                hosts=BASE_URL,
                index=INDEX,
                vector_search_mode="vector",
                highlight=True,
                embedding_model="embedding-model",
                embedding_api_key="test-key",
            )
        )


@pytest.mark.parametrize("source_factory", [_direct_source, _config_source])
def test_fetch(source_factory: Callable[[], DataSource]) -> None:
    fetch_url = f"{BASE_URL}/{INDEX}/_doc/doc-1"

    async def run() -> None:
        source = source_factory()
        try:
            with aioresponses() as mocked:
                mocked.get(fetch_url, payload=_fetch_response(), headers=_elastic_headers())

                document = await source.fetch("doc-1")

                assert ("GET", URL(fetch_url)) in mocked.requests
                assert document.id == "doc-1"
                assert document.title == "Fetched"
                assert document.text == "full document text"
                assert document.url == "https://example.test/fetched"
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_hybrid_source, _config_hybrid_source])
def test_retry_success(source_factory: Callable[..., DataSource]) -> None:
    search_url = f"{BASE_URL}/{INDEX}/_search"

    async def run() -> None:
        source = source_factory(retry_config=EMBEDDING_RETRY_CONFIG)
        try:
            with (
                respx.mock(assert_all_called=True) as embedding_router,
                aioresponses() as mocked,
            ):
                embedding_route = embedding_router.post(EMBEDDINGS_URL).mock(
                    side_effect=[
                        httpx.TimeoutException("embedding timeout"),
                        httpx.Response(200, json=_embedding_response()),
                    ]
                )
                mocked.post(search_url, payload=_search_response(), headers=_elastic_headers())

                results = await source.search("runtime", top_k=2)

                assert len(embedding_route.calls) == 2
                assert results[0].document.id == "doc-1"
        finally:
            await source.close()

    asyncio.run(run())


@pytest.mark.parametrize("source_factory", [_direct_hybrid_source, _config_hybrid_source])
def test_retry_failure(source_factory: Callable[..., DataSource]) -> None:
    async def run() -> None:
        source = source_factory(retry_config=EMBEDDING_RETRY_CONFIG)
        try:
            with respx.mock(assert_all_called=False) as embedding_router:
                embedding_route = embedding_router.post(EMBEDDINGS_URL).mock(
                    side_effect=[
                        httpx.TimeoutException("embedding timeout"),
                        httpx.TimeoutException("embedding timeout"),
                    ]
                )

                with pytest.raises(SourceError, match="failed to generate Elasticsearch query embedding"):
                    await source.search("runtime", top_k=2)

                assert len(embedding_route.calls) == 2
        finally:
            await source.close()

    asyncio.run(run())
