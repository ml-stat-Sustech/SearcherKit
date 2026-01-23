from __future__ import annotations

import json
import os
import asyncio
from typing import Dict, List, Union, TYPE_CHECKING

import requests
from openai import AsyncOpenAI

from .base import BaseTool, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentState

DEFAULT_MAX_QUERIES = int(os.getenv("MAX_MULTIQUERY_NUM", 3))
JINA_API_KEY = os.getenv("JINA_API_KEY")
DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE = os.getenv("DASHSCOPE_MODEL_SERVER", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SUMMARY_MODEL = os.getenv("WEBDANCER_VISIT_MODEL", "qwen2.5-72b-instruct")
SUMMERY_MAX_TOKENS = int(os.getenv("WEBDANCER_VISIT_MAX_TOKENS", 128024))

EXTRACT_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

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


class VisitTool(BaseTool):
    name = "visit"
    description = "Visit one or more webpages and summarise content relevant to the stated goal."
    arguments_schema = {
        "url": ["https://example.com/page", "..."],
        "goal": "string; description of the information you want to extract from the pages",
    }

    async def run(self, call: ToolCall, state: "AgentState") -> str:
        # if not JINA_API_KEY:
        #     return "[Visit] Missing JINA_API_KEY environment variable."
        if not DASHSCOPE_KEY:
            return "[Visit] Missing DASHSCOPE_API_KEY environment variable."

        raw_urls: Union[str, List[str]] = call.arguments.get("url")
        goal = call.arguments.get("goal", "")
        if isinstance(raw_urls, str):
            urls = [raw_urls]
        elif isinstance(raw_urls, list) and raw_urls:
            urls = [str(u) for u in raw_urls]
        else:
            return "[Visit] Expected {'url': ..., 'goal': ...}."

        limit = min(len(urls), DEFAULT_MAX_QUERIES)
        urls = urls[:limit]
        
        futures= [self._process_single(url, goal) for url in urls]
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        summaries = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                summaries.append(f"[Visit] Error while processing sample {i}: {res}")

        # with ThreadPoolExecutor(max_workers=limit) as pool:
        #     futures = {pool.submit(self._process_single, url, goal): url for url in urls}
        #     summaries: List[str] = []
        #     for future in as_completed(futures):
        #         try:
        #             summaries.append(future.result())
        #         except Exception as exc:  # noqa: BLE001
        #             summaries.append(f"[Visit] Error fetching {futures[future]}: {exc}")
        return "\n=======\n".join(summaries)

    async def _process_single(self, url: str, goal: str) -> str:
        content = self._fetch_via_jina(url)
        if not content:
            return self._fallback_summary(url, goal)

        prompt = EXTRACT_PROMPT.format(webpage_content=content, goal=goal)
        summary = await self._summarize(prompt)
        if not summary:
            return self._fallback_summary(url, goal)

        return self._format_summary(url, goal, summary)

    def _fetch_via_jina(self, url: str) -> str:
        if JINA_API_KEY:
            headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
        else:
            print("[Visit] Missing JINA_API_KEY, using Jina in no-key mode. Fetch may be rate-limited.")
            headers = None
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers)
            if response.status_code == 200:
                return response.text
            print(f"[Visit] Error fetching {url}: {response.status_code} {response.reason}")
        except Exception as e:
            print(f"[Visit] Error fetching {url}: {e}")
            return ""
        return ""

    async def _summarize(self, prompt: str) -> Dict[str, str]:
        client = AsyncOpenAI(api_key=DASHSCOPE_KEY, base_url=DASHSCOPE_BASE)
        try:
            response = await client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt[:SUMMERY_MAX_TOKENS]}],
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content
            assert payload
            payload = payload.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(payload)
        except Exception as e:
            print(f"[Visit] Error summarizing: {e}")
            return {}

    def _format_summary(self, url: str, goal: str, data: Dict[str, str]) -> str:
        rational = data.get("rational", "")
        evidence = data.get("evidence", "")
        summary = data.get("summary", "")
        links = data.get("urls", [])

        lines = [
            f"Useful information in {url} for goal '{goal}':",
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
        lines.append("Related Links:")
        lines.append(self._format_links(links) if links else "No related links found.")
        
        return "\n".join(lines).strip()
    
    def _format_links(self, links: List[Dict[str, str]]) -> str:
        return "\n".join(
            [
                f"- [{link.get('text', '') or link.get('title', '')}]({link.get('url', '')})"
                for link in links
            ]
        )

    def _fallback_summary(self, url: str, goal: str) -> str:
        return (
            f"Useful information in {url} for goal '{goal}':\n\n"
            "Evidence:\nUnable to access the page.\n\n"
            "Summary:\nNo information available."
        )

def main():
    tool = VisitTool()
    print(tool.run(ToolCall(name='visit', arguments={"url": "https://www.baidu.com", "goal": "Summarize the content of the page."}), None))

if __name__ == "__main__":
    main()