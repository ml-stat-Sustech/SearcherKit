from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

from webagent.agent.agent import Agent
from webagent.utils.config import instantiate
from webagent.utils.retry import retry_async, RetryPolicy

if TYPE_CHECKING:
    from webagent.llm.chat_types import ChatMessage

class AgentRunner:
    """
    Agent Runner:
    - Manage runtime resources
    - Create agent (from code or config)
    - Accept and run agent tasks (with concurrency limit if provided)
    - Manage retry and logging
    """
    def __init__(self,
                 build_agent: Callable[[], Agent] | None = None,
                 agent_config: Any = None,
                 max_concurrency: int | None = None):
        if not build_agent and not agent_config:
            raise ValueError("Either build_agent or agent_config must be provided")
        if build_agent and agent_config:
            # log warning, use build_agent
            pass
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")

        self.build_agent = build_agent or (lambda: instantiate(agent_config))
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    @asynccontextmanager
    async def _limit(self):
        if self._semaphore is None:
            yield
            return
        async with self._semaphore:
            yield

    def submit(self, 
               query: str, 
               extra: dict[str, Any] | None = None, 
               retry_policy: RetryPolicy | None = None) -> asyncio.Task[list[ChatMessage]]:
        async def _run():
            async with self._limit():
                agent = self.build_agent()
                return await agent.run(query, extra=extra)

        if retry_policy:
            return asyncio.create_task(retry_async(_run, policy=retry_policy))
        return asyncio.create_task(_run())

    async def submit_batch(
        self,
        queries: Iterable[str],
        *,
        extras: Iterable[dict[str, Any] | None] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> list[asyncio.Task[list[ChatMessage]]]:
        query_list = list(queries)
        if extras is None:
            extra_list: Sequence[dict[str, Any] | None] = [None] * len(query_list)
        else:
            extra_list = list(extras)
            if len(extra_list) != len(query_list):
                raise ValueError("extras must have the same length as queries")

        tasks = [self.submit(query, extra=extra, retry_policy=retry_policy) for query, extra in zip(query_list, extra_list)]
        return tasks
        
