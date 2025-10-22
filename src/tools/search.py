from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, TYPE_CHECKING

import requests

from .base import BaseTool, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentState

DEFAULT_MAX_QUERIES = int(os.getenv("MAX_MULTIQUERY_NUM", 3))
SERPER_KEY = os.getenv("GOOGLE_SEARCH_KEY")
SERPER_ENDPOINT = "https://google.serper.dev/search"


class SearchTool(BaseTool):
    name = "search"
    description = (
        "Performs batched web searches. Provide an array field 'query'; the tool retrieves top results "
        "for each query in a single call."
    )
    arguments_schema = {"query": ["keyword or natural-language search request", "..."]}

    def run(self, call: ToolCall, state: "AgentState") -> str:
        if not SERPER_KEY:
            return "[Search] Missing GOOGLE_SEARCH_KEY environment variable."

        queries = call.arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            return "[Search] Expected {'query': ['keyword', ...]}."

        limit = min(len(queries), DEFAULT_MAX_QUERIES)
        queries = queries[:limit]

        with ThreadPoolExecutor(max_workers=limit) as pool:
            results = list(pool.map(self._search_single, queries))
        return "\n=======\n".join(results)

    def _search_single(self, query: str) -> str:
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        payload = {"q": query}
        try:
            response = requests.post(SERPER_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=20)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return f"[Search] Failed for '{query}': {exc}"

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            return f"[Search] Invalid JSON for '{query}': {exc}"

        organic = body.get("organic") or []
        if not organic:
            return f"A Google search for '{query}' found 0 results."

        lines: List[str] = [f"A Google search for '{query}' found {len(organic)} results:", "", "## Web Results"]
        for idx, page in enumerate(organic[:10], start=1):
            title = page.get("title", "Untitled")
            link = page.get("link", "")
            snippet = page.get("snippet", "")
            date = page.get("date")
            source = page.get("source")

            parts = [f"{idx}. [{title}]({link})"]
            if date:
                parts.append(f"Date: {date}")
            if source:
                parts.append(f"Source: {source}")
            block = "\n".join(parts)
            if snippet:
                block += f"\n{snippet}"
            lines.append(block)
            lines.append("")
        return "\n".join(lines).strip()
