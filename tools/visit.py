from elasticsearch import Elasticsearch, ConnectionError
from typing import List, Dict, Any, Optional

from basetool import BaseTool

class VisitTool(BaseTool):
    """
    A tool to retrieve the full content of a document from Elasticsearch by its exact title.
    The returned format is optimized for an LLM Agent to easily parse and execute subsequent actions.
    """
    def __init__(self, es_client: Elasticsearch, index_name: str):
        if not es_client.ping():
            raise ConnectionError("Cannot connect to the Elasticsearch client.")
        self.es = es_client
        self.index_name = index_name

    @property
    def name(self) -> str:
        return "VisitForLocalWiki"

    @property
    def description(self) -> str:
        return (
            "Retrieves the full content of a page from the knowledge base given its exact title. "
            "The output includes the page's text and a list of 'Actionable Links'. "
            "To navigate to a linked page, you MUST use the value from 'visitable_title' as the 'title' parameter in your next call to this tool."
        )

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "title",
                "type": "string",
                "description": "The exact title of the page to retrieve.",
                "required": True,
            },
            {
                "name": "max_links",
                "type": "integer",
                "description": "Optional. The maximum number of actionable links to return.",
                "required": False,
            },
            {
                "name": "body_max_tokens",
                "type": "integer",
                "description": "Optional. The maximum number of characters for the page's body content. If omitted, the full text is returned.",
                "required": False,
            }
        ]

    def call(self, title: str, max_links: int = 100, body_max_tokens: Optional[int] = None) -> str:
            """
            Retrieves a document by its title and formats it into a string that is friendly for an LLM Agent.

            Args:
                title (str): The exact title of the page to retrieve.
                max_links (int): The maximum number of actionable links to display in the output. Defaults to 100.
                body_max_tokens (Optional[int]): The maximum number of characters for the page's body content.
                                                If None, the entire content is returned. Defaults to None.
            """
            if not title:
                return "Error: A title must be provided."

            es_query = {
                "size": 1,
                "query": {"match_phrase": {"title": title}}
            }

            try:
                response = self.es.search(index=self.index_name, body=es_query)
                hits = response['hits']['hits']

                if not hits:
                    return f"Error: Page with title '{title}' not found."

                source = hits[0].get('_source', {})
                page_title = source.get('title', 'N/A')
                page_text = source.get('text', 'No content available.')
                page_url = source.get('url', 'No URL available.')
                page_links = source.get('links', [])

                # --- Truncate page_text based on body_max_tokens ---
                if body_max_tokens is not None and len(page_text) > body_max_tokens:
                    page_text = page_text[:body_max_tokens] + "..."

                # 1. Assemble the basic information
                output_parts = [
                    f"Title: {page_title}",
                    f"URL: {page_url}",
                    "------------------",
                    "Page Content:",
                    page_text,
                    "------------------"
                ]

                if page_links:
                    prefixes_to_exclude = ('File:', 'Category:')
                    filtered_links = [
                        link for link in page_links
                        if not link.get('target', '').strip().startswith(prefixes_to_exclude)
                    ]

                    if filtered_links:
                        instruction = (
                            "Actionable Links: To explore the related topics for further information, "
                            f"call the `{self.name}` tool using the corresponding `visitable_title`."
                        )
                        output_parts.append(instruction)

                        links_to_show = filtered_links[:max_links]

                        for i, link_obj in enumerate(links_to_show):
                            link_text = link_obj.get('text', '[No Text]')
                            link_target = link_obj.get('target', '[No Target]')
                            output_parts.append(f"{i+1}. {link_text} (visitable_title: '{link_target}')")
                        
                        if len(filtered_links) > max_links:
                            output_parts.append("...")

                return "\n".join(output_parts)

            except Exception as e:
                return f"An error occurred while trying to visit page '{title}': {e}"



if __name__ == '__main__':
    ES_HOST = 'http://192.168.77.12:9200'
    INDEX_NAME = 'wiki20251001_e5-base-v2'

    try:
        print("Connecting to Elasticsearch...")
        es_client = Elasticsearch(ES_HOST, request_timeout=30)
        es_client.ping()
        print("Connection successful.")

        visit_tool = VisitTool(es_client=es_client, index_name=INDEX_NAME)
        print("VisitTool initialized.")

        title = "Martin Lee"
        
        print(f"\nAttempting to visit '{title}'...")
        result = visit_tool.call(title=title, max_links=1000, body_max_tokens=95000)
        print("\n--- Tool Output ---")
        print(result)
        print("---------------------------------\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")