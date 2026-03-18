import json
from typing import Any

from fastmcp.client.client import CallToolResult

from webagent.tools import BaseMCPTool

class BrowseCompPlusSearch(BaseMCPTool):
    """
    BrowseComp Plus search tool that presents results in tongyi-deepresearch format.
    https://github.com/texttron/BrowseComp-Plus/blob/main/search_agent/tongyi_utils/tool_search.py
    """
    
    async def _run(self, arguments: dict[str, Any]) -> str:
        result = await self._run_mcp_tool(arguments)
        return self._format_results(result, arguments["query"])
    
    # TODO
    def _format_results(self, result: CallToolResult, query: str):
        results = json.loads(result.content[0].text)
        formatted = []
        for idx, r in enumerate(results, 1):
            passage_text = r["snippet"]
            title = ""

            if passage_text.startswith("---\ntitle:"):
                lines = passage_text.split("\n")
                if len(lines) > 1:
                    title = lines[1].replace("title:", "").strip().strip("\"")

            if not title:
                first_line = passage_text.split('\n')[0].strip()
                title = first_line[:50] + "..." if len(first_line) > 50 else first_line

            # Follows original tongyi search tool format
            formatted_result = f"{idx}. [{title}]\n{passage_text}"
            formatted.append(formatted_result)

        return f"A search for '{query}' found {len(formatted)} results:\n\n## Web Results\n" + "\n\n".join(formatted)