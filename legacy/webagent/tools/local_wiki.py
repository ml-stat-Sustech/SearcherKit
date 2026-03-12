from __future__ import annotations

import functools
import os
import json
import asyncio
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse, unquote, quote

from elasticsearch import Elasticsearch, BadRequestError, SerializationError, ConflictError, NotFoundError, TransportError
from elasticsearch import ConnectionError as ESConnectionError
from openai import OpenAI, AsyncOpenAI

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
    
ELASTICSEARCH_RUMTIME_ERRORS = (
    BadRequestError,
    SerializationError,
    ConflictError,
    NotFoundError,
    TransportError,
    ESConnectionError,
)

DEFAULT_MAX_QUERIES = max(1, int(os.getenv("MAX_MULTIQUERY_NUM", "1")))
DEFAULT_TOP_K = max(1, int(os.getenv("LOCAL_WIKI_SEARCH_TOP_K", "10")))
DEFAULT_MAX_LINKS = max(0, int(os.getenv("LOCAL_WIKI_MAX_LINKS", "100")))
DEFAULT_BODY_MAX_CHARS = max(1, int(os.getenv("LOCAL_WIKI_BODY_MAX_CHARS", "8000")))
DEFAULT_SNIPPET_MAX_CHARS = max(50, int(os.getenv("LOCAL_WIKI_SNIPPET_MAX_CHARS", "240")))

SUMMARY_MODEL = (
    os.getenv("LOCAL_WIKI_SUMMARY_MODEL")
    or os.getenv("WEBDANCER_SUMMARY_MODEL")
    or os.getenv("WEBDANCER_VISIT_MODEL")
)
SUMMARY_API_KEY = (
    os.getenv("LOCAL_WIKI_SUMMARY_API_KEY")
    or os.getenv("WEBDANCER_SUMMARY_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
)
SUMMARY_BASE_URL = (
    os.getenv("LOCAL_WIKI_SUMMARY_BASE_URL")
    or os.getenv("WEBDANCER_SUMMARY_MODEL_SERVER")
    or os.getenv("DASHSCOPE_MODEL_SERVER")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
SUMMARY_MAX_RETRIES = max(1, int(os.getenv("LOCAL_WIKI_SUMMARY_MAX_RETRIES", "3")))

SNIPPET_PROMPT = """You are creating short snippets for local wiki search results.

## Page Title
{title}

## Search Query
{query}

## Page Content
{content}

Instructions:
1. Extract information that best answers or relates to the search query.
2. Write at most two concise sentences (<= 80 words total).
3. Avoid markdown links; quote concrete facts when possible.

Respond strictly in JSON format:
{{
  "snippet": "..."
}}
"""

VISIT_SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## User Goal
{goal}

## Task Guidelines
1. Locate the portions that directly support the goal.
2. Extract the most relevant evidence.
3. Provide a concise summary and judge usefulness.
4. Find and provide (if any) urls to other pages that may be relevant.

Respond strictly in JSON:
{{
  "rational": "...",
  "evidence": "...",
  "summary": "...",
  "urls": [
      {{
          "url": "...",
          "title": "...",
      }},
      {{...}}
  ]
}}

## Webpage Content
{webpage_content}
"""



@dataclass(frozen=True)
class LocalWikiContext:
    es: Elasticsearch
    index_name: str
    retriever: Optional[BaseRetriever]


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


DISABLE_ACTIONABLE_LINKS = _env_flag("LOCAL_WIKI_DISABLE_LINKS")
_SUMMARY_CLIENT: Optional[AsyncOpenAI] = None
_SUMMARY_CLIENT_FAILED = False


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
        "Searches the local Wikipedia index for relevant pages. Provide natural language queries. "
        "Returns candidate titles with optional relevance scores and snippets summarised via the local summary model."
    )
    arguments_schema = {
        "query": ["keyword or natural-language search request", "..."],
        "top_k": "optional integer limit on returned titles per query",
        "max_queries": "optional integer limit on how many query entries to execute in a single call",
    }

    async def run(self, call: ToolCall, state) -> str:  # type: ignore[override]
        # if not use_local_wiki_tools():
        #     return "[LocalWiki Search] Local wiki tools are disabled. Set WEBDANCER_USE_LOCAL_WIKI=1 to enable."
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

        max_queries = _safe_int(call.arguments.get("max_queries"), DEFAULT_MAX_QUERIES)
        limit = min(len(queries), max_queries)
        queries = queries[:limit]

        top_k = _safe_int(call.arguments.get("top_k"), DEFAULT_TOP_K)
        retriever = context.retriever
        if retriever is None:
            return _initialization_error("Retriever is not available.")

        snippet_client = _get_summary_client()
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

            result_count = min(len(results), top_k)
            entries: List[str] = []
            for idx, result in enumerate(results[:top_k], start=1):
                title = result.get("title") or "[No Title]"
                score = result.get("score")
                url = result.get("url") or f"localwiki://{title.replace(' ', '_')}"
                entry_lines = [f"{idx}. [{title}]({url})"]
                if score is not None:
                    entry_lines.append(f"Score: {score:.3f}")
                # snippet = self._build_snippet_block(result, query, snippet_client)
                snippet = result.get("text")[:512]
                if snippet:
                    entry_lines.append(f"Snippet: {snippet}")
                entries.append("\n".join(entry_lines).strip())

            block_lines = [
                f"A Local wiki search for '{query}' found {result_count} results:",
                "",
                "## Web Results",
                "\n\n".join(entries),
                "",
                "Call the visit tool with `url` set to the desired title to inspect full content.",
            ]
            outputs.append("\n".join(block_lines).strip())

        return "\n=======\n".join(outputs)

    def _build_snippet_block(
        self,
        result: Dict[str, object],
        query: str,
        client: Optional[OpenAI],
    ) -> str:
        text = str(result.get("text") or "").strip()
        if not text:
            return ""

        snippet: Optional[str] = None
        if client is not None and SUMMARY_MODEL:
            snippet = _summarise_snippet(
                content=text[:DEFAULT_BODY_MAX_CHARS],
                query=query,
                title=str(result.get("title") or ""),
                client=client,
            )
        if not snippet:
            snippet = _fallback_snippet(text)
        return snippet


class LocalWikiVisitTool(BaseTool):
    name = "visit"
    description = (
        "Retrieves full content for one or more local wiki pages by title. "
        "Pass the page url via the `url` field (matching Search output). Optionally include `goal` for context."
    )
    arguments_schema = {
        "url": ["Exact page url to open", "..."],
        "goal": "string; optional context for what you are looking for",
        "max_links": "optional integer limit on actionable links to display",
        # "body_max_tokens": "optional integer limit on characters returned from the page body",
    }

    async def run(self, call: ToolCall, state) -> str:  # type: ignore[override]
        # if not use_local_wiki_tools():
        #     return "[LocalWiki Visit] Local wiki tools are disabled. Set WEBDANCER_USE_LOCAL_WIKI=1 to enable."
        try:
            context = _build_context()
        except Exception as exc:  # noqa: BLE001
            return _initialization_error(str(exc))

        raw_titles = call.arguments.get("url")
        if raw_titles is None:
            return "[LocalWiki Visit] Expected arguments like {'url': 'https://...'} or {'url': ['https://...', ...]}."

        urls = _coerce_queries(raw_titles)
        if not urls:
            return "[LocalWiki Visit] Provide at least one valid page url."
        # processed_titles: List[str] = []
        # for title in titles:
        #     trimmed = title.strip()
        #     parsed = urlparse(trimmed)
        #     if parsed.scheme and parsed.path:
        #         last_segment = parsed.path.rstrip("/").split("/")[-1]
        #         if last_segment:
        #             trimmed = unquote(last_segment).replace("_", " ")
        #         else:
        #             trimmed = unquote(parsed.path).replace("_", " ")
        #     else:
        #         trimmed = trimmed.replace("_", " ")
        #     processed_titles.append(trimmed)
        # titles = processed_titles[: min(len(processed_titles), DEFAULT_MAX_QUERIES)]
        urls = urls[: min(len(urls), DEFAULT_MAX_QUERIES)]
        max_links = _safe_int(call.arguments.get("max_links"), DEFAULT_MAX_LINKS, allow_zero=True)
        if DISABLE_ACTIONABLE_LINKS:
            max_links = 0
        # body_limit = call.arguments.get("body_max_tokens")
        # body_max = _safe_int(body_limit, DEFAULT_BODY_MAX_CHARS)
        goal = str(call.arguments.get("goal", "") or "")
        summary_client = _get_summary_client()

        outputs: List[str] = []
        # for title in titles:
        #     outputs.append(self._visit_single(context, title, goal, max_links, body_max, summary_client))
        outputs = await asyncio.gather(*[self._visit_single_url(context, url, goal, max_links, summary_client) for url in urls])
        return "\n=======\n".join(outputs)
    
    async def _visit_single_url(
        self,
        context: LocalWikiContext,
        url: str,
        goal: str,
        max_links: int,
        summary_client: Optional[AsyncOpenAI],
    ) -> str:
        es = context.es
        query = {
            "query": {
                "term" : {
                    "url": {
                        "value": url
                    }
                }
            }
        }
        try:
            response = es.search(index=context.index_name, body=query)
        except ELASTICSEARCH_RUMTIME_ERRORS as exc:
            return f"[LocalWiki Visit] Failed to fetch '{url}': {exc}"
        
        hits = (response.get("hits") or {}).get("hits", [])
        if not hits:
            return f"[LocalWiki Visit] Page with url '{url}' was not found in index '{context.index_name}'."
        
        source = hits[0].get("_source", {})
        page_title = source.get("title")
        page_url = url
        page_text = source.get("text")
        links = source.get("links") or []
        
        annotated_text = _inject_inline_links(page_text, links, max_links, page_url)
        print(100*'=')
        
        if summary_client:
            return await self._render_visit_summary(page_title, page_url, goal, annotated_text, summary_client)
        else:
            return annotated_text
            

    async def _visit_single(
        self,
        context: LocalWikiContext,
        title: str,
        goal: str,
        max_links: int,
        body_max: int,
        summary_client: Optional[AsyncOpenAI],
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
        page_text = str(source.get("text", "") or "")
        links = source.get("links") or []

        annotated_text = _inject_inline_links(page_text, links, max_links, page_url)
        print(100*'=')

        if annotated_text and len(annotated_text) > body_max:
            annotated_text = f"{annotated_text[:body_max]}..."

        summary_block = await self._render_visit_summary(page_title, page_url, goal, annotated_text, summary_client)
        return summary_block

    async def _render_visit_summary(
        self,
        title: str,
        url: str,
        goal: str,
        content: str,
        client: Optional[AsyncOpenAI],
    ) -> str:
        goal_text = goal or "N/A"
        if client is not None and content:
            summary = await _summarise_visit_page(
                title=title,
                url=url,
                goal=goal_text,
                content=content,
                client=client,
            )
            if summary:
                return summary
        return self._fallback_visit_summary(title, url, goal_text, content)

    @staticmethod
    def _fallback_visit_summary(title: str, url: str, goal: str, content: str) -> str:
        lines = [f"Title: {title}", f"URL: {url}"]
        if goal and goal != "N/A":
            lines.append(f"Goal: {goal}")
        lines.append("------------------")
        lines.append("Page Content:")
        lines.append(content or "No content available.")
        lines.append("------------------")
        return "\n".join(lines).strip()

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
    # "use_local_wiki_tools",
]


def _get_summary_client() -> Optional[AsyncOpenAI]:
    global _SUMMARY_CLIENT, _SUMMARY_CLIENT_FAILED
    if _SUMMARY_CLIENT is not None:
        return _SUMMARY_CLIENT
    if _SUMMARY_CLIENT_FAILED:
        return None
    if not SUMMARY_MODEL or not SUMMARY_API_KEY:
        _SUMMARY_CLIENT_FAILED = True
        return None

    client_kwargs = {"api_key": SUMMARY_API_KEY}
    if SUMMARY_BASE_URL:
        client_kwargs["base_url"] = SUMMARY_BASE_URL
    try:
        _SUMMARY_CLIENT = AsyncOpenAI(**client_kwargs)
    except Exception:
        _SUMMARY_CLIENT_FAILED = True
        _SUMMARY_CLIENT = None
    return _SUMMARY_CLIENT


def _summarise_snippet(content: str, query: str, title: str, client: OpenAI) -> Optional[str]:
    prompt = SNIPPET_PROMPT.format(
        title=title or "Unknown Title",
        query=query,
        content=content,
    )
    for _ in range(SUMMARY_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception:
            continue

        raw = response.choices[0].message.content if response.choices else None
        if not raw:
            continue
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            continue
        snippet = str(data.get("snippet", "")).strip()
        if snippet:
            return snippet
    return None


def _fallback_snippet(content: str) -> str:
    compact = " ".join(content.split())
    snippet = compact[:DEFAULT_SNIPPET_MAX_CHARS].strip()
    if not snippet:
        return ""
    if len(compact) > DEFAULT_SNIPPET_MAX_CHARS:
        snippet = snippet.rstrip(".;,")
        snippet += "..."
    return snippet


async def _summarise_visit_page(
    title: str,
    url: str,
    goal: str,
    content: str,
    client: AsyncOpenAI,
) -> Optional[str]:
    prompt = VISIT_SUMMARY_PROMPT.format(
        goal=goal or "N/A",
        webpage_content=content,
    )
    for _ in range(SUMMARY_MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception:
            continue

        raw = response.choices[0].message.content if response.choices else None
        if not raw:
            continue
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            continue
        rational = str(data.get("rational", "")).strip()
        evidence = str(data.get("evidence", "")).strip()
        summary = str(data.get("summary", "")).strip()
        links = data.get("urls", [])
        formatted = _format_visit_summary_block(url or title, goal, rational, evidence, summary, links)
        if formatted:
            return formatted
    return None


def _format_visit_summary_block(
    source: str,
    goal: str,
    rational: str,
    evidence: str,
    summary: str,
    links: Optional[List[Dict[str, str]]] = None
) -> str:
    source_label = source or "local wiki"
    goal_label = goal or "N/A"
    lines = [
        f"The useful information in {source_label} for user goal {goal_label} as follows:",
        "",
    ]
    lines.append("Evidence in page:")
    lines.append(evidence or "No evidence extracted.")
    lines.append("")
    lines.append("Summary:")
    lines.append(summary or "No summary available.")
    lines.append("")
    if links:
        lines.append("Links:")
        lines.append(_format_links(links))
    lines.append("According to the above content, if there may be links containing useful information, please use the "
                 "visit tool to access them. If not, please use the search tool to continue searching.")
    return "\n".join(lines).strip()

def _format_links(links: List[Dict[str, str]]) -> str:
        return "\n".join(
            [
                f"- [{link.get('text', '') or link.get('title', '')}]({link.get('url', '')})"
                for link in links
            ]
        )

def _inject_inline_links(content: str, links: Iterable[Dict[str, str]], limit: int, page_url: str) -> str:
    if not content or not links:
        return content
    if limit == 0:
        return content

    actionable: List[Tuple[str, str]] = []
    prefixes_to_skip = ("File:", "Category:")
    normalised_limit = max(0, limit)
    for link in links:
        target = (link.get("target") or "").strip()
        if not target or target.startswith(prefixes_to_skip):
            continue
        text = (link.get("text") or "").strip()
        if not text:
            text = target
        actionable.append((text, target))
        if normalised_limit and len(actionable) >= normalised_limit:
            break

    base_url = _derive_link_base(page_url)
    updated = content
    for text, target in actionable:
        link_url = _build_link_url(target, base_url)
        marker = f"[{text}]({link_url})"
        start = 0
        pieces: List[str] = []
        replaced = False
        while True:
            idx = updated.find(text, start)
            if idx == -1:
                break
            pieces.append(updated[start:idx])
            pieces.append(marker)
            start = idx + len(text)
            replaced = True
        if replaced:
            pieces.append(updated[start:])
            updated = "".join(pieces)
    return updated


def _derive_link_base(page_url: str) -> Optional[str]:
    parsed = urlparse(page_url or "")
    if not parsed.scheme or not parsed.netloc:
        return None
    if "/wiki/" in (parsed.path or ""):
        return f"{parsed.scheme}://{parsed.netloc}/wiki/"

    path = parsed.path or ""
    slash_idx = path.rfind("/")
    if slash_idx == -1:
        base_path = "/"
    else:
        base_path = path[: slash_idx + 1] or "/"
    if not base_path.startswith("/"):
        base_path = f"/{base_path}"
    if not base_path.endswith("/"):
        base_path = f"{base_path}/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def _build_link_url(target: str, base_url: Optional[str]) -> str:
    cleaned = (target or "").strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if parsed.scheme:
        return cleaned

    if "#" in cleaned:
        title_part, fragment = cleaned.split("#", 1)
    else:
        title_part, fragment = cleaned, ""
    slug = quote(title_part.replace(" ", "_"))
    fragment_suffix = f"#{quote(fragment.replace(' ', '_'))}" if fragment else ""

    if base_url:
        base = base_url if base_url.endswith("/") else f"{base_url}/"
        return f"{base}{slug}{fragment_suffix}"
    return f"localwiki://{slug}{fragment_suffix}"
