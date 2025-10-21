from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union, TYPE_CHECKING

import requests
from openai import OpenAI

from .base import BaseTool, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.base import AgentState

DEFAULT_MAX_QUERIES = int(os.getenv("MAX_MULTIQUERY_NUM", 3))
JINA_API_KEY = os.getenv("JINA_API_KEY")
DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE = os.getenv("DASHSCOPE_MODEL_SERVER", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SUMMARY_MODEL = os.getenv("WEBDANCER_VISIT_MODEL", "qwen2.5-72b-instruct")

EXTRACT_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## Webpage Content
{webpage_content}

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
"""


class VisitTool(BaseTool):
    name = "visit"
    description = "Visit one or more webpages and summarise content relevant to the stated goal."

    def run(self, call: ToolCall, state: "AgentState") -> str:
        if not JINA_API_KEY:
            return "[Visit] Missing JINA_API_KEY environment variable."
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

        with ThreadPoolExecutor(max_workers=limit) as pool:
            futures = {pool.submit(self._process_single, url, goal): url for url in urls}
            summaries: List[str] = []
            for future in as_completed(futures):
                try:
                    summaries.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    summaries.append(f"[Visit] Error fetching {futures[future]}: {exc}")
        return "\n=======\n".join(summaries)

    def _process_single(self, url: str, goal: str) -> str:
        content = self._fetch_via_jina(url)
        if not content:
            return self._fallback_summary(url, goal)

        prompt = EXTRACT_PROMPT.format(webpage_content=content, goal=goal)
        summary = self._summarize(prompt)
        if not summary:
            return self._fallback_summary(url, goal)

        return self._format_summary(url, goal, summary)

    def _fetch_via_jina(self, url: str) -> str:
        headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=15)
            if response.status_code == 200:
                return response.text
        except Exception:
            return ""
        return ""

    def _summarize(self, prompt: str) -> Dict[str, str]:
        client = OpenAI(api_key=DASHSCOPE_KEY, base_url=DASHSCOPE_BASE)
        try:
            response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content
            return json.loads(payload)
        except Exception:
            return {}

    def _format_summary(self, url: str, goal: str, data: Dict[str, str]) -> str:
        rational = data.get("rational", "")
        evidence = data.get("evidence", "")
        summary = data.get("summary", "")

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
        return "\n".join(lines).strip()

    def _fallback_summary(self, url: str, goal: str) -> str:
        return (
            f"Useful information in {url} for goal '{goal}':\n\n"
            "Evidence:\nUnable to access the page.\n\n"
            "Summary:\nNo information available."
        )
