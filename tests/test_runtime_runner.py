from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from searchagent.common.messages import assistant
from searchagent.runtime.runner import AgentRunner


class WindowGuardLoader:
    def __init__(self, *, total: int, window: int, completed: dict[str, int]) -> None:
        self.total = total
        self.window = window
        self.completed = completed
        self.index = 0
        self.produced = 0

    def __iter__(self) -> "WindowGuardLoader":
        return self

    def __next__(self) -> tuple[str, dict[str, Any] | None, Any | None]:
        if self.index >= self.total:
            raise StopIteration
        if self.produced - self.completed["count"] >= self.window:
            raise AssertionError("runner consumed beyond scheduler window")
        index = self.index
        self.index += 1
        self.produced += 1
        return f"query-{index}", None, None


class SlowAgent:
    def __init__(self, completed: dict[str, int]) -> None:
        self.completed = completed

    async def run(
        self,
        query: str,
        session_id: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> list[Any]:
        await asyncio.sleep(0.01)
        self.completed["count"] += 1
        return [assistant("done")]


def test_runner_does_not_consume_entire_dataloader_before_tasks_complete(tmp_path) -> None:
    async def run_case() -> None:
        completed = {"count": 0}
        runner = AgentRunner(config=SimpleNamespace(agent=None, max_concurrency=2))
        runner.build_agent = lambda: SlowAgent(completed)
        loader = WindowGuardLoader(total=5, window=2, completed=completed)

        summary = await runner.run(
            dataloader=loader,
            output_path=tmp_path,
            checkpoint={"enabled": False},
        )

        assert summary["total"] == 5
        assert summary["completed"] == 5
        assert summary["failed"] == 0
        assert completed["count"] == 5

    asyncio.run(run_case())
