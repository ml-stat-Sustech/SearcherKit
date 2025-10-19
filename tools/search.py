# tools.py

from elasticsearch import Elasticsearch, ConnectionError
from typing import List, Dict, Any, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from basetool import BaseTool
from retrievers.retrievers import BaseRetriever, build_retriever
from retrievers.encoders import BaseEncoder, build_encoder, load_model

class SearchTool(BaseTool):
    """
    A tool to search for relevant pages in the local knowledge base using a query.
    This tool acts as a wrapper around a configured retriever instance.
    """
    def __init__(self, retriever: BaseRetriever):
        """
        Initializes the SearchTool with a specific retriever instance.

        Args:
            retriever (BaseRetriever): A configured instance of BM25Retriever, 
                                       DenseRetriever, or HybridRetriever.
        """
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "SearchLocalWiki"

    @property
    def description(self) -> str:
        return (
            "Searches the local Wikipedia knowledge base to find pages relevant to a query. "
            "It returns a ranked list of page titles. "
            "To read the content of a page, you MUST use the page's full title from the list as the `title` parameter for the `VisitForLocalWiki` tool."
        )
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "query",
                "type": "string",
                "description": "The search query or topic to look for.",
                "required": True,
            },
            {
                "name": "top_k",
                "type": "integer",
                "description": "Optional. The number of top matching page titles to return. Defaults to 10.",
                "required": False,
            }
        ]

    def call(self, query: str, top_k: int = 5) -> str:
        """
        使用配置好的 retriever 执行搜索，并将结果格式化后返回给 LLM Agent。
        """
        if not query:
            return "Error: A query must be provided."

        try:
            results = self.retriever.search(query=query, top_k=top_k)

            if not results:
                return f"No relevant pages found for the query: '{query}'"
            
            # 为 Agent 格式化输出 (新版)
            instruction = (
                f"Found {len(results)} relevant pages for '{query}'. "
                f"To read the content of a page, use its full title as the `title` parameter for the `VisitForLocalWiki` tool."
            )
            
            output_parts = [instruction, ""] # 添加一个空行以作分隔
            
            for i, res in enumerate(results):
                title = res.get('title', '[No Title Found]')
                output_parts.append(f"{i+1}. {title}")
            
            return "\n".join(output_parts)

        except Exception as e:
            return f"An error occurred during search for query '{query}': {e}"

if __name__ == '__main__':
    # --- Configuration ---
    ES_HOST = 'http://192.168.77.12:9200'
    # INDEX_NAME = 'wiki20251001_qwen3-embedding-0.6b'
    # MODEL_NAME = 'Qwen3-Embedding-0.6B'

    INDEX_NAME = 'wiki20251001_e5-base-v2'
    MODEL_NAME = 'e5-base-v2' # 

    RETRIEVER_TYPE = 'dense' # Use 'bm25', 'dense'

    try:
        # --- Step 1: Build the necessary components ---
        print("Connecting to Elasticsearch and initializing model...")
        es_client = Elasticsearch(ES_HOST, request_timeout=60)
        
        encoder_instance = None
        if RETRIEVER_TYPE in ['dense', 'hybrid']:
            model = load_model(f'/mnt/sharedata/ssd_large/common/LLMs/{MODEL_NAME}')
            encoder_instance = build_encoder(MODEL_NAME, model)
        
        retriever_instance = build_retriever(
            retriever_type=RETRIEVER_TYPE,
            es_client=es_client,
            index_name=INDEX_NAME,
            encoder=encoder_instance
        )
        print(f"Components initialized successfully. Using '{RETRIEVER_TYPE}' retriever.")

        # --- Step 2: Instantiate the tools ---
        search_tool = SearchTool(retriever=retriever_instance)
        print("Search and Visit tools are ready.")

        # --- Step 3: Simulate the Agent's workflow ---
        user_query = "What college did the author of 'The Da Vinci Code' attend?"
        print(f"\n[AGENT ACTION] Using Search Tool for query: '{user_query}'")
        
        # Agent calls the search tool
        search_result_text = search_tool.call(query=user_query, top_k=10)
        
        print("\n--- Search Tool Output ---")
        print(search_result_text)
        print("--------------------------\n")


    except Exception as e:
        print(f"An unexpected error occurred during the test workflow: {e}")