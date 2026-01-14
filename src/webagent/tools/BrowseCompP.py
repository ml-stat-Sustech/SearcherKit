from __future__ import annotations
import json
from typing import Dict, TYPE_CHECKING, Optional

import backoff
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from openai import AsyncOpenAI, RateLimitError

from .base import BaseTool
if TYPE_CHECKING:
    from agents.base import AgentState
    from tools.base import ToolCall


SNIPPET_LENGTH = 200
class SearchTool(BaseTool):
    name = 'search'
    arguments_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Array of search queries to issue.",
                }
            },
            "required": ["query"],
        }
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 5 results for each query in one call."
    
    def __init__(self) -> None:
        super().__init__()
        
    async def init(self, endpoint="http://127.0.0.1:8100/mcp"):
        transport = SSETransport(
            url=endpoint,
            headers={"Authorization": "Bearer token"}
        )

        self.client = Client(transport)
        async with self.client:
            tools = await self.client.list_tools()
            info = None
            for tool in tools:
                if tool.name == "search":
                    info = tool
            if info is None:
                raise ValueError(f"No tool search found from {endpoint}")
        
    async def run(self, call: ToolCall, state: AgentState) -> str:
        assert call.name == self.name, f"{call.name} != {self.name}"
        async with self.client:
            queries = call.arguments['query']
            results = []
            for query in queries:
                result = await self.client.call_tool("search", {"query":query})
                result = json.loads(result.content[0].text)
                results.append(f"Result for {query}:\n\n"+"".join(self._format_search_result(s) for s in result))
            return "\n\n".join(results)
            
    @staticmethod
    def _format_search_result(result: dict):
        doc_id = result["docid"]
        snippet = result["snippet"][:SNIPPET_LENGTH]
        return f"\n----------ID: {doc_id}------------\n\n{snippet}\n\n-------------------------------\n"
    
SUMMARY_MODEL="/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B"
SUMMARY_KEY=""
SUMMARY_BASE_URL="http://192.168.77.15:8200/v1"
SUMMARY_MAX_LENGTH=120000
EXTRACT_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## User Goal
{goal}

## Task Guidelines
1. Locate the portions that directly support the goal.
2. Extract the most relevant evidence.
3. Provide a concise summary and judge usefulness.

Respond strictly in JSON:
{{
  "rational": "...",
  "evidence": "...",
  "summary": "..."
}}

## Webpage Content
{webpage_content}
"""
    
class VisitTool(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    arguments_schema = {
        "type": "object",
        "properties": {
            "docid": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The DocID(s) of the webpage(s) to visit. Can be a single ID or an array of IDs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["docid", "goal"]
  }
    
    def __init__(self) -> None:
        super().__init__()
        
    async def init(self, endpoint="http://127.0.0.1:8100/mcp"):
        transport = SSETransport(
            url=endpoint,
            headers={"Authorization": "Bearer token"}
        )

        self.client = Client(transport)
        async with self.client:
            tools = await self.client.list_tools()
            info = None
            for tool in tools:
                if tool.name == "get_document":
                    info = tool
                    break
            if info is None:
                raise ValueError(f"No tool named 'get_document' from {endpoint}")  
        
    async def _summarize(self, content: str, goal: Optional[str]) -> Optional[Dict[str, str]]:
        prompt = EXTRACT_PROMPT.format(webpage_content=content[:SUMMARY_MAX_LENGTH], goal=goal)
        client = AsyncOpenAI(api_key=SUMMARY_KEY, base_url=SUMMARY_BASE_URL)
        query = backoff.on_exception(backoff.expo, RateLimitError)(client.chat.completions.create)
        response = await query(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20, 
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        payload = response.choices[0].message.content
        assert payload
        payload = payload.split("</think")[-1]
        payload = payload.strip().lstrip("```json").rstrip("```").strip()
        
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} for {payload}")
            return None
        
        # check json
        if not payload["evidence"] or not isinstance(payload["evidence"], str):
            print(f"Invalid evidence: {payload['evidence']}")
            return None
        if not payload["summary"] or not isinstance(payload["summary"], str):
            print(f"Invalid summary: {payload['summary']}")
            return None
        if payload["rational"] and not isinstance(payload["rational"], str):
            print(f"Invalid rational: {payload['rational']}")
            return None
        return payload
    
    def _format_summary(self, docid: str, goal: str, data: Dict[str, str]) -> str:
        rational = data.get("rational", "")
        evidence = data.get("evidence", "")
        summary = data.get("summary", "")

        lines = [
            f"Useful information in {docid} for goal '{goal}':",
            "",
        ]
        if rational:
            lines.append(f"Reasoning: {rational}")
            lines.append("")
        lines.append("Evidence:")
        lines.append(evidence or "No evidence extracted.")
        lines.append("")
        lines.append("Summary:")
        lines.append(summary or "No summary available.")
        return "\n".join(lines).strip()

    def _fallback_summary(self, docid: str, goal: str) -> str:
        return (
            f"Useful information in {docid} for goal '{goal}':\n\n"
            "Evidence:\nUnable to access the page.\n\n"
            "Summary:\nNo information available."
        )
                        
            
    async def run(self, call: ToolCall, state: AgentState) -> str:
        assert call.name == self.name, f"{call.name} != {self.name}"
        async with self.client:
            docids = call.arguments['docid']
            goal = call.arguments['goal']
            results = []
            for docid in docids:
                resp = await self.client.call_tool("get_document", {"docid": docid})
            # result = await self.client.call_tool("get_document", {"docid": call.arguments['docid']})
                if not resp.content:
                    results.append(self._fallback_summary(
                        docid=docid,
                        goal=goal,
                    ))
                    continue
                resp = json.loads(resp.content[0].text)
                content = resp['text']
                summary = await self._summarize(content, goal)
                if not summary:
                    results.append(self._fallback_summary(
                        docid=docid,
                        goal=goal,
                    ))
                    continue
                results.append(self._format_summary(docid=docid, goal=goal, data=summary))
        return "\n\n".join(results)
    
FAKEURL = "https://www.browsecomp.com/"
    
class FakeVisitTool(VisitTool):
    arguments_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }
    
    async def run(self, call: ToolCall, state: AgentState) -> str:
        urls = call.arguments["url"]
        docids = []
        for url in urls:
            docids.append(url.lstrip(FAKEURL))
        del call.arguments["url"]
        call.arguments["docid"] = docids
        return await super().run(call, state)
    
class FakeSearchTool(SearchTool):
    
    async def run(self, call: ToolCall, state: AgentState) -> str:
        result = await super().run(call, state)
        return result.replace("--ID: ", "--"+FAKEURL)