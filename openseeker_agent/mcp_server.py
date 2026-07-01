"""BCP search/visit FastMCP server for OpenSeeker-style SFT evaluation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from elasticsearch import (
    BadRequestError,
    ConflictError,
    Elasticsearch,
    NotFoundError,
    SerializationError,
    TransportError,
)
from fastmcp import FastMCP
from fastmcp.tools import tool
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from prompts import VISIT_SUMMARY_PROMPT

try:
    import json_repair
except ImportError:
    json_repair = None


ELASTICSEARCH_RUNTIME_ERRORS = (
    BadRequestError,
    SerializationError,
    ConflictError,
    NotFoundError,
    TransportError,
)

SUMMARY_MAX_LENGTH = 32000
SNIPPET_SIZE = 512 * 5


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _normalise_vector(values: list[float]) -> np.ndarray:
    embeddings = np.array(values, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norms + 1e-9)


def _safe_source_text(source: Mapping[str, Any], *, with_highlight: bool, hit: Mapping[str, Any]) -> str:
    if with_highlight:
        highlight = hit.get("highlight")
        if isinstance(highlight, Mapping):
            text_highlights = highlight.get("text")
            if isinstance(text_highlights, list):
                joined = "\n...\n".join(str(part) for part in text_highlights)
                if joined:
                    return joined.replace("<em>", "").replace("</em>", "")
        text = source.get("text")
        return str(text or "")[: 128 * 5]
    return str(source.get("text") or "")


def _format_hits(hits: Iterable[Mapping[str, Any]], *, with_highlight: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source")
        if not isinstance(source, Mapping):
            source = {}
        out.append(
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "title": source.get("title"),
                "text": _safe_source_text(source, with_highlight=with_highlight, hit=hit),
                "url": source.get("url"),
                "links": source.get("links", []),
            }
        )
    return out


class APIEmbeddingEncoder:
    def __init__(self, endpoint: str, model_name: str) -> None:
        self._client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY")
        self._model_name = model_name

    async def encode(self, text: str) -> np.ndarray:
        prefixed = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}"
        response = await self._client.embeddings.create(model=self._model_name, input=prefixed)
        return _normalise_vector(response.data[0].embedding)


@dataclass
class BCPSearch:
    es_host: str
    index: str
    embedding_endpoint: str
    model_name: str
    max_candidates: int = 10
    snippet_length: int | None = SNIPPET_SIZE
    with_highlighted_snippet: bool = True

    def __post_init__(self) -> None:
        self.es_client = Elasticsearch(self.es_host)
        self.encoder = APIEmbeddingEncoder(self.embedding_endpoint, self.model_name)

    def parse_search_results(self, results: list[tuple[str, list[dict[str, Any]]]]) -> str:
        blocks: list[str] = []
        for query, result_rows in results:
            if not result_rows:
                blocks.append(f"No relevant pages found for '{query}'.")
                continue
            entries: list[str] = []
            for index, row in enumerate(result_rows):
                title = row.get("title") or "[No Title]"
                url = row.get("url") or f"localwiki://{str(title).replace(' ', '_')}"
                entry_lines = [f"{index}. [{title}]({url})"]
                text = str(row.get("text") or "")[:512]
                if text:
                    entry_lines.append(text)
                entries.append("\n".join(entry_lines).strip())
            blocks.append(
                "\n".join(
                    [
                        f"A Local wiki search for '{query}' found {len(result_rows)} results:",
                        "",
                        "## Web Results",
                        "\n\n".join(entries),
                    ]
                ).strip()
            )
        return "\n\n=======\n\n".join(blocks)

    async def _search_one(self, query: str) -> list[dict[str, Any]]:
        vector = (await self.encoder.encode(query)).tolist()
        es_query: dict[str, Any] = {
            "size": self.max_candidates,
            "knn": {
                "field": "text_vector",
                "query_vector": vector,
                "k": self.max_candidates,
                "num_candidates": 100,
            },
        }
        if self.with_highlighted_snippet:
            es_query["query"] = {"multi_match": {"query": query, "fields": ["text", "title"]}}
            es_query["highlight"] = {"fields": {"text": {"fragment_size": 128}}}
        response = self.es_client.search(index=self.index, body=es_query)
        hits = response.get("hits", {}).get("hits", [])
        if not isinstance(hits, list):
            hits = []
        return _format_hits(hits, with_highlight=self.with_highlighted_snippet)

    @tool()
    async def search(self, query: str | list[str]) -> str:
        if isinstance(query, str):
            results = [(query, await self._search_one(query))]
        elif isinstance(query, list):
            queries = [item for item in query if isinstance(item, str)]
            batch_results = await asyncio.gather(*(self._search_one(item) for item in queries))
            results = list(zip(queries, batch_results))
        else:
            return "[Search] Invalid request format: query must be a string or array of strings"
        return self.parse_search_results(results)


class BCPVisit:
    def __init__(self, es_host: str, index: str, *, summary: bool = True) -> None:
        self.es_host = es_host
        self.index = index
        self.es_client = Elasticsearch(self.es_host)
        self.summary = summary
        self.summary_model = os.getenv("SUMMARY_MODEL") or os.getenv("SUMMARY_MODEL_NAME")
        self.summary_api_key = os.getenv("SUMMARY_API_KEY")
        self.summary_base_url = os.getenv("SUMMARY_BASE_URL")
        self.summary_max_retries = max(1, int(os.getenv("SUMMARY_MAX_RETRIES", "3")))
        self.summary_client: AsyncOpenAI | None = None
        if summary and self.summary_model and self.summary_api_key:
            kwargs: dict[str, str] = {"api_key": self.summary_api_key}
            if self.summary_base_url:
                kwargs["base_url"] = self.summary_base_url
            self.summary_client = AsyncOpenAI(**kwargs)

    @tool()
    async def visit(self, url: str | list[str], goal: str | None = None) -> str:
        if isinstance(url, str):
            return await self._visit_single(url, goal)
        if isinstance(url, list):
            urls = [item for item in url if isinstance(item, str)]
            results = await asyncio.gather(*(self._visit_single(item, goal) for item in urls))
            return "\n\n=======\n\n".join(results)
        return "[Visit] Invalid request format: url must be a string or array of strings"

    async def _visit_single(self, url: str, goal: str | None = None) -> str:
        query = {"query": {"term": {"url": {"value": url}}}}
        response = self.es_client.search(index=self.index, body=query)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            print(f"Page with url '{url}' was not found from {self.es_host} in index '{self.index}'.")
            return f"Page with url '{url}' was not found."
        first = hits[0]
        source = first.get("_source", {})
        if not isinstance(source, Mapping):
            source = {}
        title = str(source.get("title") or "[No Title]")
        text = str(source.get("text") or "")

        if self.summary and self.summary_client is not None:
            summary = await self._summarise_visit_page(title, url, goal or "", text)
            if summary:
                return summary
        return f"# [{title}]({url})\n\n{text}"

    async def _summarise_visit_page(self, title: str, url: str, goal: str, content: str) -> str | None:
        if self.summary_client is None or not self.summary_model:
            return None
        if len(content) > SUMMARY_MAX_LENGTH:
            print(f"Content too long, truncated to {SUMMARY_MAX_LENGTH} characters.")
        prompt = VISIT_SUMMARY_PROMPT.format(
            goal=goal or "N/A",
            webpage_content=content[:SUMMARY_MAX_LENGTH],
        )
        for _ in range(self.summary_max_retries):
            try:
                response = await self.summary_client.chat.completions.create(
                    model=self.summary_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            except (APIConnectionError, APIError, APITimeoutError):
                continue

            raw = response.choices[0].message.content if response.choices else None
            if not raw:
                continue
            data = _loads_json_object(raw)
            if not data:
                continue
            rational = str(data.get("rational", "")).strip()
            evidence = str(data.get("evidence", "")).strip()
            summary = str(data.get("summary", "")).strip()
            formatted = _format_visit_summary_block(url or title, goal, rational, evidence, summary)
            if formatted:
                return formatted
        return None


def _loads_json_object(raw: str) -> Mapping[str, Any] | None:
    if json_repair is not None:
        repaired = json_repair.loads(raw)
        if isinstance(repaired, Mapping):
            return repaired

    candidates = [raw]
    left = raw.find("{")
    right = raw.rfind("}")
    if left != -1 and right != -1 and left <= right:
        candidates.append(raw[left : right + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, Mapping):
            return data
    return None


def _format_visit_summary_block(
    source: str,
    goal: str,
    rational: str,
    evidence: str,
    summary: str,
) -> str:
    lines = [
        f"The useful information in {source or 'local wiki'} for user goal {goal or 'N/A'} as follows:",
        "",
        "Evidence in page:",
        evidence or "No evidence extracted.",
        "",
        "Summary:",
        summary or "No summary available.",
        "",
    ]
    return "\n".join(lines).strip()


def create_app() -> Any:
    es_host = _require_env("ELASTICSEARCH_HOST")
    index = _require_env("ELASTICSEARCH_INDEX")
    embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:8004/v1")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "/home/jovyan/Qwen3-Embedding-8B")

    searcher = BCPSearch(
        es_host=es_host,
        index=index,
        embedding_endpoint=embedding_endpoint,
        model_name=embedding_model_name,
        max_candidates=10,
        with_highlighted_snippet=True,
    )
    visitor = BCPVisit(es_host, index, summary=True)
    mcp = FastMCP()
    mcp.add_tool(searcher.search)
    mcp.add_tool(visitor.visit)
    return mcp.http_app(path="/mcp/", stateless_http=True)
