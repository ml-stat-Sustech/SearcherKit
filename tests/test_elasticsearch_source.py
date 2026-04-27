from __future__ import annotations

import asyncio
import json
from typing import Any

from searchagent.sources.elasticsearch import ElasticsearchSource
from searchagent.tools import ToolConfig, build_tool


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

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
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

    def get(self, *, index: str, id: str) -> dict[str, Any]:
        for doc in self.documents:
            if doc["_id"] == id:
                return doc
        raise KeyError(id)


def test_elasticsearch_source_search_and_fetch_by_es_id() -> None:
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

    asyncio.run(run_source())


def test_elasticsearch_source_can_be_configured_for_browsecomp_plus_visit_ids() -> None:
    async def run_source() -> None:
        client = FakeElasticsearchClient()
        source = ElasticsearchSource(
            client=client,
            index="browsecomp_hybrid",
            search_fields=["title^2", "text"],
            document_id_field="url",
            fetch_field="url",
            highlight_fragment_size=128,
            snippet_chars=512,
            metadata_fields=["links"],
        )

        search_tool = build_tool(
            ToolConfig(type="search", name="search", source="bcp"),
            sources={"bcp": source},
        )
        visit_tool = build_tool(
            ToolConfig(type="visit", name="visit", source="bcp"),
            sources={"bcp": source},
        )

        search_payload = json.loads(await search_tool.run(query="benchmark", top_k=1))
        assert search_payload[0]["document"]["id"] == "https://example.test/bcp"
        assert "text" not in search_payload[0]["document"]

        visit_payload = json.loads(
            await visit_tool.run(document_id="https://example.test/bcp")
        )
        assert visit_payload["title"] == "BrowseComp Plus"
        assert "benchmark corpus" in visit_payload["text"]

    asyncio.run(run_source())
