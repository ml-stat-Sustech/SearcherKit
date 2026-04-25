from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BatchItem:
    index: int
    sample_id: str
    prompt: str
    extra: dict[str, Any] | None
    answer: Any
    record_path: Path
    trace_id: str


@dataclass(slots=True)
class BatchSummary:
    run_id: str
    output_dir: Path
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    resumed: int = 0
    total_turns: int = 0
    total_tool_calls: int = 0
    avg_turns: float = 0.0
    avg_tool_calls: float = 0.0
    time_elapsed: float = 0.0
    checkpoint_path: str | None = None
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def mark_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def add_completed(self, stats: dict[str, Any]) -> None:
        self.completed += 1
        self.total_turns += int(stats.get("turns", 0))
        self.total_tool_calls += int(stats.get("tool_calls", 0))

    def finalize(self, elapsed: float) -> dict[str, Any]:
        self.time_elapsed = elapsed
        if self.completed:
            self.avg_turns = self.total_turns / self.completed
            self.avg_tool_calls = self.total_tool_calls / self.completed
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "resumed": self.resumed,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "avg_turns": self.avg_turns,
            "avg_tool_calls": self.avg_tool_calls,
            "time_elapsed": self.time_elapsed,
            "checkpoint_path": self.checkpoint_path,
            "skip_reasons": dict(self.skip_reasons),
        }

