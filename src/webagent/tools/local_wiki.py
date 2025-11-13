from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse, unquote

from elasticsearch import Elasticsearch, ConnectionError

from .base import BaseTool, ToolCall

try:  # pragma: no cover - optional dependency during local setup
    from src.local_wiki.retrievers.retrievers import BaseRetriever, build_retriever
    from src.local_wiki.retrievers.encoders import BaseEncoder, build_encoder, load_model
except Exception as exc:  # noqa: BLE001
    # The actual error is surfaced during tool execution.
    BaseRetriever = BaseEncoder = None  # type: ignore[assignment]
    build_retriever = build_encoder = load_model = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_MAX_QUERIES = max(1, int(os.getenv("MAX_MULTIQUERY_NUM", "3")))
DEFAULT_TOP_K = max(1, int(os.getenv("LOCAL_WIKI_SEARCH_TOP_K", "10")))
DEFAULT_MAX_LINKS = max(0, int(os.getenv("LOCAL_WIKI_MAX_LINKS", "100")))
DEFAULT_BODY_MAX_CHARS = max(1, int(os.getenv("LOCAL_WIKI_BODY_MAX_CHARS", "8000")))


@dataclass(frozen=True)
class LocalWikiContext:
    es: Elasticsearch
    index_name: str
    retriever: Optional[BaseRetriever]


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def use_local_wiki_tools() -> bool:
    return (
        _env_flag("WEBDANCER_USE_LOCAL_WIKI")
        or _env_flag("WEBWALKER_USE_LOCAL_WIKI")
        or _env_flag("RAG_USE_LOCAL_WIKI")
    )


DISABLE_ACTIONABLE_LINKS = _env_flag("LOCAL_WIKI_DISABLE_LINKS")


def _initialization_error(message: str) -> str:
    return f"[LocalWiki] {message}"


@functools.lru_cache(maxsize=1)
def _build_context() -> LocalWikiContext:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Failed to import local_wiki components: {_IMPORT_ERROR}. "
            "Ensure dependencies from requirements.txt are installed."
        )

    index_name = os.getenv("LOCAL_WIKI_INDEX", 'wiki20251001_qwen3-embedding-0.6b')
    if not index_name:
        raise RuntimeError("LOCAL_WIKI_INDEX environment variable is required for local wiki tools.")

    es_host = os.getenv("LOCAL_WIKI_ES_HOST", "http://192.168.77.12:9200")
    timeout = os.getenv("LOCAL_WIKI_ES_TIMEOUT")
    es_kwargs: Dict[str, Union[int, float]] = {}
    if timeout:
        try:
            es_kwargs["request_timeout"] = float(timeout)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise RuntimeError(f"Invalid LOCAL_WIKI_ES_TIMEOUT value: {timeout}") from exc

    try:
        es = Elasticsearch(es_host, **es_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to create Elasticsearch client for host {es_host}: {exc}") from exc

    try:
        es = Elasticsearch(es_host, request_timeout=100, retry_on_timeout=True, max_retries=3)
        info = es.info()
    except ConnectionError as e:
        print(f"Unable to connect to Elasticsearch. Error: {e}")
        exit(1)

    retriever_type = os.getenv("LOCAL_WIKI_RETRIEVER", "dense").lower()
    retriever: Optional[BaseRetriever] = None

    if retriever_type not in {"bm25", "dense", "hybrid"}:
        raise RuntimeError(f"Unsupported LOCAL_WIKI_RETRIEVER '{retriever_type}'. Choose bm25, dense, or hybrid.")

    encoder: Optional[BaseEncoder] = None
    if retriever_type in {"dense", "hybrid"}:
        model_name = os.getenv("LOCAL_WIKI_MODEL_NAME", "/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B")
        if not model_name:
            raise RuntimeError(
                f"LOCAL_WIKI_MODEL_NAME is required when LOCAL_WIKI_RETRIEVER={retriever_type}."
            )
        if load_model is None or build_encoder is None:
            raise RuntimeError("local_wiki encoder utilities are unavailable.")
        model = load_model(model_name)
        encoder = build_encoder(model_name, model)

    if build_retriever is None:
        raise RuntimeError("local_wiki retriever builder is unavailable.")

    retriever = build_retriever(
        retriever_type=retriever_type,
        es_client=es,
        index_name=index_name,
        encoder=encoder,
    )

    return LocalWikiContext(es=es, index_name=index_name, retriever=retriever)


def _coerce_queries(raw: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(raw, str):
        queries = [raw]
    elif isinstance(raw, Sequence):
        queries = [str(item) for item in raw if isinstance(item, (str, bytes))]
    else:
        return []
    # Strip whitespace and drop blanks
    return [query.strip() for query in queries if query and query.strip()]


def _safe_int(value: object, default: int, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
        if parsed > 0:
            return parsed
        if allow_zero and parsed == 0:
            return 0
        return default
    except Exception:  # _coerce_queriesnoqa: BLE001
        return default


class LocalWikiSearchTool(BaseTool):
    name = "search"
    description = (
        "Searches the local Wikipedia index for relevant pages. Provide one or more natural language queries. "
        "Use the visit tool with `url` equal to the returned page title to read full content."
    )
    arguments_schema = {
        "query": ["keyword or natural-language search request", "..."],
        "top_k": "optional integer limit on returned titles per query",
    }

    def run(self, call: ToolCall, state) -> str:  # type: ignore[override]
        if not use_local_wiki_tools():
            return "[LocalWiki Search] Local wiki tools are disabled. Set WEBDANCER_USE_LOCAL_WIKI=1 to enable."
        try:
            context = _build_context()
        except Exception as exc:  # noqa: BLE001
            return _initialization_error(str(exc))

        queries_raw = call.arguments.get("query")
        if queries_raw is None:
            return "[LocalWiki Search] Expected arguments like {'query': 'topic'} or {'query': ['topic', ...]}."

        queries = _coerce_queries(queries_raw)
        if not queries:
            return "[LocalWiki Search] Expected a non-empty query string or list of strings."

        limit = min(len(queries), DEFAULT_MAX_QUERIES)
        queries = queries[:limit]

        top_k = _safe_int(call.arguments.get("top_k"), DEFAULT_TOP_K)
        retriever = context.retriever
        if retriever is None:
            return _initialization_error("Retriever is not available.")

        outputs: List[str] = []
        for query in queries:
            try:
                results = retriever.search(query=query, top_k=top_k)
            except Exception as exc:  # noqa: BLE001
                outputs.append(f"[LocalWiki Search] Retrieval failed for '{query}': {exc}")
                continue

            if not results:
                outputs.append(f"No relevant pages found for '{query}'.")
                continue

            header = (
                f"Local wiki search for '{query}' returned {min(len(results), top_k)} candidate titles.\n"
                "Call the visit tool with `url` set to the chosen title to inspect its content."
            )
            lines = [header, ""]
            for idx, result in enumerate(results[:top_k], start=1):
                title = result.get("title") or "[No Title]"
                score = result.get("score")
                if score is not None:
                    lines.append(f"{idx}. {title} (score: {score:.3f})")
                else:
                    lines.append(f"{idx}. {title}")
            outputs.append("\n".join(lines).strip())

        return "\n=======\n".join(outputs)


class LocalWikiVisitTool(BaseTool):
    name = "visit"
    description = (
        "Retrieves full content for one or more local wiki pages by title. "
        "Pass the page title via the `url` field (matching Search output). Optionally include `goal` for context."
    )
    arguments_schema = {
        "url": ["Exact page title to open", "..."],
        "goal": "string; optional context for what you are looking for",
        "max_links": "optional integer limit on actionable links to display",
        "body_max_tokens": "optional integer limit on characters returned from the page body",
    }

    def run(self, call: ToolCall, state) -> str:  # type: ignore[override]
        if not use_local_wiki_tools():
            return "[LocalWiki Visit] Local wiki tools are disabled. Set WEBDANCER_USE_LOCAL_WIKI=1 to enable."
        try:
            context = _build_context()
        except Exception as exc:  # noqa: BLE001
            return _initialization_error(str(exc))

        raw_titles = call.arguments.get("url")
        if raw_titles is None:
            return "[LocalWiki Visit] Expected arguments like {'url': 'Page Title'} or {'url': ['Title', ...]}."

        titles = _coerce_queries(raw_titles)
        if not titles:
            return "[LocalWiki Visit] Provide at least one valid page title."
        processed_titles: List[str] = []
        for title in titles:
            trimmed = title.strip()
            parsed = urlparse(trimmed)
            if parsed.scheme and parsed.path:
                last_segment = parsed.path.rstrip("/").split("/")[-1]
                if last_segment:
                    trimmed = unquote(last_segment).replace("_", " ")
                else:
                    trimmed = unquote(parsed.path).replace("_", " ")
            processed_titles.append(trimmed)
        titles = processed_titles
        titles = titles[: min(len(titles), DEFAULT_MAX_QUERIES)]
        print(titles)

        max_links = _safe_int(call.arguments.get("max_links"), DEFAULT_MAX_LINKS, allow_zero=True)
        if DISABLE_ACTIONABLE_LINKS:
            max_links = 0
        body_limit = call.arguments.get("body_max_tokens")
        body_max = _safe_int(body_limit, DEFAULT_BODY_MAX_CHARS)
        goal = str(call.arguments.get("goal", "") or "")

        outputs: List[str] = []
        for title in titles:
            outputs.append(self._visit_single(context, title, goal, max_links, body_max))
        return "\n=======\n".join(outputs)

    def _visit_single(
        self,
        context: LocalWikiContext,
        title: str,
        goal: str,
        max_links: int,
        body_max: int,
    ) -> str:
        es = context.es
        query = {"size": 1, "query": {"match_phrase": {"title": title}}}
        try:
            response = es.search(index=context.index_name, body=query)
        except Exception as exc:  # noqa: BLE001
            return f"[LocalWiki Visit] Failed to fetch '{title}': {exc}"

        hits = (response.get("hits") or {}).get("hits", [])
        if not hits:
            return f"[LocalWiki Visit] Page titled '{title}' was not found in index '{context.index_name}'."
            # raise SystemExit(f"Page titled '{title}' not found")

        source = hits[0].get("_source", {})
        page_title = source.get("title") or title
        page_url = source.get("url", "N/A")
        page_text = source.get("text", "")
        links = source.get("links") or []

        if page_text and len(page_text) > body_max:
            page_text = f"{page_text[:body_max]}..."

        lines = [f"Title: {page_title}", f"URL: {page_url}"]
        if goal:
            lines.append(f"Goal: {goal}")
        lines.append("------------------")
        lines.append("Page Content:")
        lines.append(page_text or "No content available.")
        lines.append("------------------")

        if max_links == 0:
            return "\n".join(lines).strip()

        actionable_links = self._render_links(links, max_links)
        if actionable_links:
            lines.append("Actionable Links:")
            lines.extend(actionable_links)
        else:
            lines.append("Actionable Links: None available.")

        return "\n".join(lines).strip()

    @staticmethod
    def _render_links(links: Iterable[Dict[str, str]], max_links: int) -> List[str]:
        if max_links <= 0:
            return []
        prefixes_to_skip = ("File:", "Category:")
        actionable: List[Dict[str, str]] = []
        for link in links:
            target = (link.get("target") or "").strip()
            if not target or target.startswith(prefixes_to_skip):
                continue
            text = link.get("text") or target
            actionable.append({"text": text, "target": target})

        if not actionable:
            return []

        rendered = [
            f"{idx}. {item['text']} (call visit with url='{item['target']}')"
            for idx, item in enumerate(actionable[:max_links], start=1)
        ]
        if len(actionable) > max_links:
            rendered.append("...")
        return rendered


class WebWalkerLocalWikiVisitTool(LocalWikiVisitTool):
    """
    WebWalker-compatible wrapper around the local wiki visit tool.

    Keeps the legacy `visit_page` tool name while accepting either `button` or `url`
    arguments, so prompts written for the original crawler-based tool continue to work.
    """

    name = "visit_page"
    description = (
        "Fetches local wiki content. Provide the page title via either 'url' or 'button'. "
        "Optionally include 'goal' to describe what information you are looking for."
    )

    def run(self, call: ToolCall, state) -> str:  # type: ignore[override]
        normalised_args: Dict[str, object] = dict(call.arguments)
        button_value = normalised_args.pop("button", None)
        if "url" not in normalised_args and isinstance(button_value, str):
            normalised_args["url"] = button_value

        if "goal" not in normalised_args:
            user_query = getattr(getattr(state, "user_input", None), "query", None)
            if isinstance(user_query, str) and user_query.strip():
                normalised_args["goal"] = user_query.strip()

        proxy_call = ToolCall(name="visit", arguments=normalised_args)
        return super().run(proxy_call, state)


__all__ = [
    "LocalWikiSearchTool",
    "LocalWikiVisitTool",
    "WebWalkerLocalWikiVisitTool",
    "use_local_wiki_tools",
]
