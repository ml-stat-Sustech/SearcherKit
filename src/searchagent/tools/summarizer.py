"""Summary generation component for source-backed tools."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

import json_repair
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, OpenAIError, RateLimitError

from searchagent.common.retry import RetryConfig, RetryPolicy, retry_async
from searchagent.errors import RecoverableError
from searchagent.sources.base import Document

if TYPE_CHECKING:
    from searchagent.tools.base import SummarizerConfig

SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## Task Guidelines
1. Locate the portions that directly support the goal.
2. Extract the most relevant evidence.
3. Provide a concise summary.

Respond strictly in JSON:
{{
  "evidence": "...",
  "summary": "..."
}}

## User Goal
{goal}

## Webpage Content
{webpage_content}
"""

_OPENAI_ERRORS = (
    OpenAIError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

class SummaryError(RecoverableError):
    """Raised when the summary model response is missing or malformed."""

def _limit_request(semaphore: asyncio.Semaphore | None) -> Any:
    if semaphore is None:
        return nullcontext()
    return semaphore

def format_document_content(document: Document) -> str:
    title = document.title or "[No Title]"
    url = document.url or document.id
    return f"[{title}]({url})\n{document.text}".strip()


class Summarizer:
    """OpenAI-compatible summary generation component."""

    @overload
    def __init__(
        self,
        *,
        config: "SummarizerConfig",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_chars: int = 400000,
        timeout: float = 3600,
        max_concurrency: int | None = None,
        default_kwargs: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        config: "SummarizerConfig | None" = None,
        model: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        max_chars: int = 400000,
        timeout: float = 3600,
        max_concurrency: int | None = None,
        default_kwargs: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if config is not None:
            model = config.model
            api_key = api_key or config.api_key
            base_url = base_url or config.base_url
            max_chars = config.max_chars
            timeout = config.timeout
            if max_concurrency is None:
                max_concurrency = config.max_concurrency
            default_kwargs = default_kwargs or config.default_kwargs
            retry_policy = retry_policy or (RetryPolicy(config=config.retry_config) if config.retry_config is not None else None)

        if not model:
            raise ValueError("summary tool requires summarizer.model")
        
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        summary_client = AsyncOpenAI(**client_kwargs)

        self.model = model
        self.timeout = timeout
        self.max_chars = max_chars
        self.default_kwargs = dict(default_kwargs or {})
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self.retry_policy = retry_policy or RetryPolicy(exceptions=(*_OPENAI_ERRORS, SummaryError, ValueError))
        self.client = summary_client

    async def summarize(self, *, goal: str, content: str) -> tuple[str, str]:
        try:
            return await retry_async(
                self._request_summary,
                goal,
                content[: self.max_chars],
                policy=self.retry_policy,
                op_name="summarizer.summary",
            )
        except (*_OPENAI_ERRORS, SummaryError, ValueError) as exc:
            raise RecoverableError("failed to generate summary") from exc

    async def _request_summary(self, goal: str, content: str) -> tuple[str, str]:
        prompt = SUMMARY_PROMPT.format(goal=goal or "N/A", webpage_content=content)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "timeout": self.timeout,
            **self.default_kwargs,
        }
        async with _limit_request(self._semaphore):
            response = await self.client.chat.completions.create(**payload)
        raw = response.choices[0].message.content if response.choices else None
        if not raw:
            raise SummaryError("summary model returned empty content")
        data = json_repair.loads(raw)
        if not isinstance(data, Mapping):
            raise SummaryError("summary model returned non-object JSON")
        evidence = str(data.get("evidence", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not evidence and not summary:
            raise SummaryError("summary model returned no evidence or summary")
        return evidence, summary
