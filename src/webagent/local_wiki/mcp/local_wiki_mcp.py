from typing import Optional, List, Union, Dict, Tuple, Iterable
from urllib.parse import urlparse, unquote, quote
import os
import json
import asyncio
import numpy as np
from openai import AsyncOpenAI, OpenAIError

from fastmcp import FastMCP
from fastmcp.tools import tool
from elasticsearch import Elasticsearch
import json_repair

from src.local_wiki.retrievers.retrievers import build_retriever
from src.local_wiki.retrievers.encoders import build_encoder, load_model

from elasticsearch import Elasticsearch, BadRequestError, SerializationError, ConflictError, NotFoundError, TransportError
ELASTICSEARCH_RUMTIME_ERRORS = (
    BadRequestError,
    SerializationError,
    ConflictError,
    NotFoundError,
    TransportError
)

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY")
SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL")
SUMMARY_MAX_RETRIES = max(1, int(os.getenv("SUMMARY_MAX_RETRIES", "3")))
SUMMARY_MAX_LENGTH = 32000 # ~ 32768 

VISIT_SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

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

## User Goal
{goal}

## Webpage Content
{webpage_content}
"""

_SUMMARY_CLIENT: Optional[AsyncOpenAI] = None
_SUMMARY_CLIENT_FAILED = False


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


async def _summarise_visit_page(
    title: str,
    url: str,
    goal: str,
    content: str,
    client: AsyncOpenAI,
) -> Optional[str]:
    if len(content) > SUMMARY_MAX_LENGTH:
        print(f"Content too long, truncated to {SUMMARY_MAX_LENGTH} characters.")
    prompt = VISIT_SUMMARY_PROMPT.format(
        goal=goal or "N/A",
        webpage_content=content[:SUMMARY_MAX_LENGTH],
    )
    for _ in range(SUMMARY_MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=SUMMARY_MODEL,
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
        except OpenAIError:
            continue

        raw = response.choices[0].message.content if response.choices else None
        if not raw:
            continue
        data = json_repair.loads(raw)
        if not data:
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

mcp = FastMCP()

class VLLMEncoder:
    def __init__(self, vllm_endpoint: str, vllm_model_name: str) -> None:
        self._endpoint = vllm_endpoint
        self._model_name = vllm_model_name
        self._client = AsyncOpenAI(base_url=self._endpoint, api_key="EMPTY")

    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [f"Query:{texts}"]
        else:
            texts = [f"Query:{t}" for t in texts]

        response = await self._client.embeddings.create(
            model=self._model_name,
            input=texts
        )
        embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        return embeddings


class LocalWikiSearch:

    def __init__(self,
                 es_host: str,
                 index: str,
                 vllm_endpoint: str,
                 model_name: str,
                 max_candidates: int = 5,
                 retriever_type: str = 'api') -> None:
        self.es_host = es_host
        self.es_client = Elasticsearch(self.es_host)
        self.max_candidates = max_candidates
        self.index = index
        self.retriever_type = retriever_type
        if retriever_type == 'api':
            encoder = VLLMEncoder(vllm_endpoint, model_name)
            self.retriever = build_retriever('api', self.es_client, index, encoder)
        elif retriever_type == 'dense':
            model = load_model(model_name)
            encoder = build_encoder(model_name, model)
            self.retriever = build_retriever('dense', self.es_client, index, encoder)
        else:
            raise ValueError(f"Invalid type \"{retriever_type}\"")
        
    def parse_search_results(self, results: List[Tuple[str,List[Dict[str, object]]]]) -> str:
        blocks = []
        for q, res in results:
            if len(res) == 0:
                blocks.append(f"No relevant pages found for '{q}'.")
                continue
            entries = []
            for i, r in enumerate(res):
                title = r.get("title","[No Title]")
                score = r.get("score", None)
                url = r.get("url", f"localwiki://{title.replace(' ', '_')}")
                entry_lines = [f"{i}. [{title}]({url})"]
                if score is not None:
                    entry_lines.append(f"Score: {score:.3f}")
                # snippet = self._build_snippet_block(result, query, snippet_client)
                snippet = r.get("text")[:512]
                if snippet:
                    entry_lines.append(f"Snippet: {snippet}")
                entries.append("\n".join(entry_lines).strip())
            blocks.append("\n".join([
                f"A Local wiki search for '{q}' found {len(res)} results:",
                "",
                "## Web Results",
                "\n\n".join(entries),
                "",
                "Call the visit tool (if provided) to inspect full content.",
            ]).strip())
            
        return "\n=======\n".join(blocks)
        
    @tool()
    async def search(self, query: Union[str, List[str]]) -> str:
        """
        Searches the local Wikipedia index for relevant pages.

        :param query: A str of search query or a list of queries
        :type query: Union[str, List[str]]
        :return: Search results. Each result has "title", "score", "text" and "url".
        :rtype: str
        """
        if isinstance(query, str):
            if self.retriever_type == 'api':
                results = await self.retriever.search(query=query, top_k=self.max_candidates)
            else:
                results = self.retriever.search(query=query, top_k=self.max_candidates)
            results = [(query, results)]
        else:
            if self.retriever_type == 'api':
                batch_results = await self.retriever.batch_search(queries=query, top_k=self.max_candidates)
            else:
                batch_results = self.retriever.batch_search(queries=query, top_k=self.max_candidates)
            results = list(zip(query, batch_results))

        return self.parse_search_results(results)
    
class LocalWikiVisit:
    def __init__(self, es_host: str, index: str, summary: bool = False) -> None:
        self.es_host = es_host
        self.index = index
        self.es_client = Elasticsearch(self.es_host)
        self.summary = summary
        self.summary_client = _get_summary_client() if summary else None

    @tool()
    async def visit(self, url: Union[str, List[str]], goal: Optional[str] = None) -> str:
        """
        Retrieves full content for one or more local wiki pages by url.

        :param url: A str of url or a list of urls
        :type url: Union[str, List[str]]
        :param goal: optional context for what you are looking for
        :return: Title and content of the web page.
        :rtype: str
        """
        if isinstance(url, str):
            return await self._visit_single(url, goal)
        results = await asyncio.gather(*[self._visit_single(u, goal) for u in url])
        return "\n".join(results)
        
    async def _visit_single(self, url: str, goal: Optional[str] = None) -> str:
        query = {"query": {"term" : {"url": {"value": url}}}}
        try:
            response = self.es_client.search(index=self.index, body=query)
        except ELASTICSEARCH_RUMTIME_ERRORS as exc:
            print(f"Failed to fetch '{url}': {exc} from {self.es_host} in index '{self.index}'")
            raise

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            print(f"Page with url '{url}' was not found from {self.es_host} in index '{self.index}'.")
            return f"Page with url '{url}' was not found."

        title = hits[0].get("_source", {}).get("title", "[No Title]")
        text = hits[0].get("_source", {}).get("text", "")
        links = hits[0].get("_source", {}).get("links", [])

        annotated = self._inject_inline_links(text, links, url)

        if self.summary and self.summary_client:
            summary = await _summarise_visit_page(
                title=title,
                url=url,
                goal=goal or "",
                content=annotated,
                client=self.summary_client,
            )
            if summary:
                return summary

        return f"# [{title}]({url})\n\n" + annotated
        
    def _inject_inline_links(self, content: str, links: Iterable[Dict[str, str]], page_url: str, limit: Optional[int] = None) -> str:
        if not content or not links:
            return content
        if limit == 0:
            return content

        actionable: List[Tuple[str, str]] = []
        prefixes_to_skip = ("File:", "Category:")
        for link in links:
            target = (link.get("target") or "").strip()
            if not target or target.startswith(prefixes_to_skip):
                continue
            text = (link.get("text") or "").strip()
            if not text:
                text = target
            actionable.append((text, target))
            if limit and len(actionable) >= limit:
                break

        base_url = self._derive_link_base(page_url)
        updated = content
        for text, target in actionable:
            link_url = self._build_link_url(target, base_url) # convert wiki title to url
            marker = f"[{text}]({link_url})"
            # Only replace the first occurrence
            updated = updated.replace(text, marker, 1)
        return updated
    
    def _derive_link_base(self, page_url: str) -> Optional[str]:
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


    def _build_link_url(self, target: str, base_url: Optional[str]) -> str:
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

es_host = ELASTICSEARCH_HOST
index = ELASTICSEARCH_INDEX

# vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://192.168.77.15:8200/v1")
vllm_model_name = os.getenv("VLLM_MODEL", "/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B")


searcher = LocalWikiSearch(es_host, index, "", vllm_model_name, retriever_type='dense')
visitor = LocalWikiVisit(es_host, index, summary=True)
mcp = FastMCP()
mcp.add_tool(searcher.search)
mcp.add_tool(visitor.visit)

app = mcp.http_app(path="/mcp/", stateless_http=True)
