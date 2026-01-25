from typing import Optional, List, Union, Dict, Tuple

from fastmcp import FastMCP
from fastmcp.tools import tool
from elasticsearch import Elasticsearch
import uvicorn

from src.local_wiki.retrievers.retrievers import build_retriever
from src.local_wiki.retrievers.encoders import build_encoder, load_model

mcp = FastMCP()

class LocalWikiSearch:
    name = "search"
    description = (
        "Searches the local Wikipedia index for relevant pages. "
        "Returns candidate titles with optional relevance scores and snippets."
    )
    arguments_schema = {
        "query": ["search query", "..."],
    }
    
    def __init__(self, es_host: str, index: str, max_candidates=5, type = 'dense', emb_model: Optional[str] = None) -> None:
        self.es_client = Elasticsearch(es_host)
        self.max_candidates = max_candidates
        if type == 'dense':
            if not emb_model:
                raise ValueError("Need a encoder for dense search")
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
            results = self.retriever.search(query=query, top_k=self.max_candidates)
            results = [(query, results)]
        else:
            results = self.retriever.batch_search(queries=query, top_k=self.max_candidates)
            results = list(zip(query, results))
            
        return self.parse_search_results(results)
    
searcher = LocalWikiSearch(
    "http://192.168.77.12:9200",
    "wiki20251001_qwen3-embedding-0.6b",
    emb_model="/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B")
mcp = FastMCP()
mcp.add_tool(searcher.search)

app = mcp.http_app(path="/mcp/", stateless_http=True)
    
# if __name__ == "__main__":
#     uvicorn.run("src.local_wiki.mcp:app::app", host="0.0.0.0", port=8100, workers=4)
# TODO: this will load 4 duplicate sentence transformers