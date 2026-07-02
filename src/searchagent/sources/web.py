"""Live web search and page-fetch data source adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, overload
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout

from searchagent.common.retry import RetryConfig, RetryPolicy, retry_async
from searchagent.errors import SourceError

from .base import DataSource, Document, SearchResult, SourceConfig


_WEB_ERRORS = (ClientError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError)


class WebSource(DataSource):
    """Search the public web with Google Serper and fetch pages through Jina Reader."""

    @overload
    def __init__(
        self,
        *,
        config: SourceConfig,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        serper_api_key: str | None = None,
        serper_endpoint: str = "https://google.serper.dev/search",
        serper_timeout: float = 30,
        jina_api_key: str | None = None,
        jina_base_url: str = "https://r.jina.ai/",
        jina_timeout: float = 60,
        web_max_concurrency: int | None = None,
        web_retry_config: RetryConfig | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        serper_api_key: str | None = None,
        serper_endpoint: str = "https://google.serper.dev/search",
        serper_timeout: float = 30,
        jina_api_key: str | None = None,
        jina_base_url: str = "https://r.jina.ai/",
        jina_timeout: float = 60,
        web_max_concurrency: int | None = None,
        web_retry_config: RetryConfig | None = None,
        config: SourceConfig | None = None,
    ) -> None:
        if config is not None:
            serper_api_key = serper_api_key or config.serper_api_key
            serper_endpoint = config.serper_endpoint
            serper_timeout = config.serper_timeout
            jina_api_key = jina_api_key or config.jina_api_key
            jina_base_url = config.jina_base_url
            jina_timeout = config.jina_timeout
            if web_max_concurrency is None:
                web_max_concurrency = config.web_max_concurrency
            web_retry_config = web_retry_config or config.web_retry_config

        if not serper_api_key:
            raise ValueError("WebSource requires serper_api_key")
        if not serper_endpoint:
            raise ValueError("WebSource requires serper_endpoint")
        if not jina_base_url:
            raise ValueError("WebSource requires jina_base_url")
        if serper_timeout <= 0:
            raise ValueError("serper_timeout must be positive")
        if jina_timeout <= 0:
            raise ValueError("jina_timeout must be positive")
        if web_max_concurrency is not None and web_max_concurrency <= 0:
            raise ValueError("web_max_concurrency must be positive")

        self.serper_api_key = serper_api_key
        self.serper_endpoint = serper_endpoint
        self.serper_timeout = serper_timeout
        self.jina_api_key = jina_api_key
        self.jina_base_url = jina_base_url
        self.jina_timeout = jina_timeout
        self._semaphore = asyncio.Semaphore(web_max_concurrency) if web_max_concurrency else None
        self.retry_policy = (
            RetryPolicy(config=web_retry_config)
            if web_retry_config is not None
            else RetryPolicy(exceptions=(*_WEB_ERRORS, ValueError))
        )
        self._client: ClientSession | None = None

    async def close(self) -> None:
        if self._client is not None and not self._client.closed:
            await self._client.close()
        self._client = None

    async def _client_session(self) -> ClientSession:
        if self._client is None or self._client.closed:
            self._client = ClientSession()
        return self._client

    def _limit_request(self) -> Any:
        if self._semaphore is None:
            return nullcontext()
        return self._semaphore

    async def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if not query:
            raise ValueError("query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        try:
            response = await retry_async(
                self._request_search,
                query,
                top_k,
                policy=self.retry_policy,
                op_name="web.search",
            )
        except (*_WEB_ERRORS, ValueError) as exc:
            raise SourceError("failed to search web with Serper") from exc

        results: list[SearchResult] = []
        for item in response.get("organic", []):
            url = item.get("link", "") or ""
            snippet = item.get("snippet", "") or ""
            metadata = {}
            for key in ("position", "date", "sitelinks"):
                if key in item:
                    metadata[key] = item[key]
            results.append(
                SearchResult(
                    document=Document(
                        id=url,
                        title=item.get("title") or None,
                        text="",
                        url=url or None,
                        metadata=metadata,
                    ),
                    score=None,
                    snippet=snippet,
                    metadata={"source": "serper"},
                )
            )
        return results

    async def fetch(self, document_id: str, *, goal: str | None = None) -> Document:
        if not document_id:
            raise ValueError("document_id must be non-empty")
        try:
            text = await retry_async(
                self._request_jina_page,
                document_id,
                policy=self.retry_policy,
                op_name="web.fetch",
            )
        except (*_WEB_ERRORS, ValueError) as exc:
            raise SourceError(f"failed to fetch web document {document_id!r} with Jina") from exc
        return Document(
            id=document_id,
            text=text,
            url=document_id,
            metadata={"source": "jina"},
        )

    async def _request_search(self, query: str, top_k: int) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "q": query,
            "num": top_k,
        }

        async with self._limit_request():
            client = await self._client_session()
            async with client.post(
                self.serper_endpoint,
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=ClientTimeout(total=self.serper_timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()
        if not isinstance(result, Mapping):
            raise ValueError("web search response must be a JSON object")
        return result

    async def _request_jina_page(self, url: str) -> str:
        headers = {"Accept": "text/plain"}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        async with self._limit_request():
            client = await self._client_session()
            async with client.get(
                f"{self.jina_base_url.rstrip('/')}/{quote(url, safe=':/')}",
                headers=headers,
                timeout=ClientTimeout(total=self.jina_timeout),
            ) as response:
                response.raise_for_status()
                text = await response.text()
        if not text:
            raise ValueError("Jina returned empty document text")
        return text
