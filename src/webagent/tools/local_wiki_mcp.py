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
        pass