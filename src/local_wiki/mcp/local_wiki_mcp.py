from typing import Optional, List, Union, Dict, Tuple, Iterable
from urllib.parse import urlparse, unquote, quote

from fastmcp import FastMCP
from fastmcp.tools import tool
from elasticsearch import Elasticsearch
import uvicorn

from src.local_wiki.retrievers.retrievers import build_retriever

from elasticsearch import Elasticsearch, BadRequestError, SerializationError, ConflictError, NotFoundError, TransportError
ELASTICSEARCH_RUMTIME_ERRORS = (
    BadRequestError,
    SerializationError,
    ConflictError,
    NotFoundError,
    TransportError
)

mcp = FastMCP()

class LocalWikiSearch:
    
    def __init__(self, 
                 es_host: str, 
                 index: str, 
                 max_candidates=5, 
                 type = 'dense', 
                 emb_model: Optional[str] = None) -> None:
        self.es_host = es_host
        self.es_client = Elasticsearch(self.es_host)
        self.max_candidates = max_candidates
        self.index = index
        if type == 'dense':
            if not emb_model:
                raise ValueError("Need a encoder for dense search")
            from src.local_wiki.retrievers.encoders import build_encoder, load_model
            model = load_model(emb_model)
            encoder = build_encoder(emb_model, model)
            self.retriever = build_retriever(type, self.es_client, index, encoder)
        else:
            raise ValueError(f"Invalid type \"{type}\"")
        
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
    def search(self, query: Union[str, List[str]]) -> str:
        """
        Searches the local Wikipedia index for relevant pages.
        
        :param query: A str of search query or a list of queries
        :type query: Union[str, List[str]]
        :return: Search results. Each result has "title", "score", "text" and "url".
        :rtype: str
        """
        if isinstance(query, str):
            results = self.retriever.search(query=query, top_k=self.max_candidates) # TODO: error handling
            results = [(query, results)]
        else:
            results = self.retriever.batch_search(queries=query, top_k=self.max_candidates)
            results = list(zip(query, results))
            
        return self.parse_search_results(results)
    
class LocalWikiVisit:
    def __init__(self, es_host: str, index: str, summary=False) -> None:
        self.es_host = es_host
        self.index = index
        self.es_client = Elasticsearch(self.es_host)
        self.summary = summary
        if self.summary:
            pass # TODO
    @tool()
    def visit(self, url: Union[str, List[str]], goal: Optional[str]=None) -> str:
        """
        Retrieves full content for one or more local wiki pages by url.
        
        :param url: A str of url or a list of urls
        :type url: Union[str, List[str]]
        :return: Title and content of the web page.
        :rtype: str
        """
        if isinstance(url, str):
            return self._visit_single(url)
        return "\n".join([self._visit_single(u) for u in url])
        
    def _visit_single(self, url: str) -> str:
        query = {"query": {"term" : {"url": {"value": url}}}}
        try:
            response = self.es_client.search(index=self.index, body=query)
        except ELASTICSEARCH_RUMTIME_ERRORS as exc:
            print(f"Failed to fetch '{url}': {exc} from {self.es_host} in index '{self.index}'")
            raise # this should be passed to the mcp client
        
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            print(f"Page with url '{url}' was not found from {self.es_host} in index '{self.index}'.")
            return f"Page with url '{url}' was not found."
        
        title = hits[0].get("_source", {}).get("title", "[No Title]")
        text = hits[0].get("_source", {}).get("text", "")
        links = hits[0].get("_source", {}).get("links", [])
        
        annotated = self._inject_inline_links(text, links, url)
        
        if self.summary:
            pass # TODO
        
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

es_host = "http://192.168.77.12:9200"
index = "wiki20251001_qwen3-embedding-0.6b"
emb_model = "/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B"
    
searcher = LocalWikiSearch(es_host,index,emb_model=emb_model)
visitor = LocalWikiVisit(es_host,index)
mcp = FastMCP()
mcp.add_tool(searcher.search)
mcp.add_tool(visitor.visit)

app = mcp.http_app(path="/mcp/", stateless_http=True)
    
# if __name__ == "__main__":
#     uvicorn.run("src.local_wiki.mcp:app::app", host="0.0.0.0", port=8100, workers=4)
# TODO: this will load 4 duplicate sentence transformers