from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from searchagent.runtime.interactive_selection import active_model_event_data, infer_active_source
from searchagent.agent.search_agent import SearchAgent, SearchAgentConfig
from searchagent.common.errors import SearchAgentError
from searchagent.common.log import (
    configure_run_logging,
    get_logger,
    log_context,
    update_trace_metadata,
)
from searchagent.common.live_events import LiveEvent, LiveEventSink, MultiSink, RunEventRecorder
from searchagent.runtime.trace import (
    history_stats,
    make_run_id,
    make_trace_id,
    serialize_message,
    tool_summary,
)

logger = get_logger(__name__)

InteractiveRunStatus = Literal["completed", "failed", "cancelled"]


@dataclass
class InteractiveQueryConfig:
    """Configuration used by the interactive query runtime.

    This is a projection of run configuration for ad hoc user-entered queries.
    Batch-only fields such as dataloaders, checkpoints, batch concurrency, and
    outer runner retry do not belong here.
    """

    agent: SearchAgentConfig = field(default_factory=SearchAgentConfig)
    output_path: str | None = None
    logging: dict[str, Any] | None = None
    record_dir: str | None = None


@dataclass(slots=True)
class InteractiveRunResult:
    status: InteractiveRunStatus
    record_path: Path
    payload: dict[str, Any]


def resolve_record_dir(config: InteractiveQueryConfig) -> Path:
    if config.record_dir:
        return Path(config.record_dir)
    if config.output_path:
        return Path(config.output_path) / "interactive"
    return Path("outputs/interactive")


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _record_path(record_dir: Path, *, trace_id: str) -> Path:
    return record_dir / f"{_timestamp_for_filename()}_{trace_id}.json"


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "repr": repr(exc),
    }


def _interactive_logging_cfg(base_cfg: dict[str, Any] | None, *, record_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg) if base_cfg is not None else {}
    cfg["global_file"] = str(record_dir / "run.log")
    trace_cfg = cfg.get("trace")
    if not isinstance(trace_cfg, dict):
        trace_cfg = {}
    trace_cfg["dir"] = str(record_dir / "traces")
    trace_format = str(trace_cfg.get("format", "text")).strip().lower()
    trace_cfg["filename_template"] = "{trace_id}.json" if trace_format == "json" else "{trace_id}.log"
    cfg["trace"] = trace_cfg
    return cfg


class InteractiveQueryRunner:
    """Run one ad hoc query and persist an Interactive Run Record."""

    def __init__(
        self,
        *,
        config: InteractiveQueryConfig,
        build_agent: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.record_dir = resolve_record_dir(config)
        self._build_agent = build_agent or self._build_default_agent
        self.run_id = make_run_id()

    def _build_default_agent(self) -> SearchAgent:
        return SearchAgent(config=copy.deepcopy(self.config.agent))

    async def run_query(
        self,
        query: str,
        *,
        live_event_sink: LiveEventSink | None = None,
    ) -> InteractiveRunResult:
        if not query.strip():
            raise ValueError("query must be non-empty")

        self.record_dir.mkdir(parents=True, exist_ok=True)
        configure_run_logging(
            output_dir=self.record_dir,
            cfg=_interactive_logging_cfg(self.config.logging, record_dir=self.record_dir),
        )

        trace_id = make_trace_id()
        record_path = _record_path(self.record_dir, trace_id=trace_id)
        started_at = datetime.now().isoformat(timespec="milliseconds")
        started_monotonic = time.monotonic()
        recorder = RunEventRecorder()
        sink = MultiSink(recorder, live_event_sink)

        async def emit(kind: str, message: str, data: dict[str, Any] | None = None) -> None:
            await sink(LiveEvent(kind=kind, message=message, data=data or {}))  # type: ignore[arg-type]

        def build_payload(
            *,
            status: InteractiveRunStatus,
            ended_at: str,
            elapsed: float,
            history: list[Any] | None = None,
            stats: dict[str, Any] | None = None,
            error: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "kind": "interactive_query_run",
                "status": status,
                "run_id": self.run_id,
                "trace_id": trace_id,
                "input": query,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed": elapsed,
                "events": recorder.events,
                "history": history,
                "stats": stats,
            }
            if error is not None:
                payload["error"] = error
            return payload

        def write_payload(payload: dict[str, Any]) -> None:
            record_path.write_text(
                json.dumps(payload, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )

        with log_context(scope="trace", run_id=self.run_id, trace_id=trace_id, turn="-"):
            active_source = infer_active_source(self.config.agent)
            await emit(
                "run_started",
                "Interactive query run started",
                {
                    "run_id": self.run_id,
                    "trace_id": trace_id,
                    "record_path": str(record_path),
                    "input": query,
                    "active_source": active_source.as_event_data(),
                    "active_model": active_model_event_data(self.config.agent.llm_client),
                },
            )
            update_trace_metadata(
                run_id=self.run_id,
                sample_id="-",
                run={"run_id": self.run_id},
                sample={
                    "trace_id": trace_id,
                    "query": query,
                    "record_path": str(record_path),
                    "started_at": started_at,
                    "kind": "interactive_query_run",
                },
                execution={"status": "running", "success": None, "error": None},
            )
            logger.info("Starting interactive query run query=%r", query[:120])
            try:
                agent = self._build_agent()
                history = await agent.run(query, session_id=0, live_event_sink=sink)
                stats = history_stats(history)
                summary = tool_summary(history)
                ended_at = datetime.now().isoformat(timespec="milliseconds")
                elapsed = round(time.monotonic() - started_monotonic, 3)
                serialized_history = [serialize_message(message) for message in history]
                await emit(
                    "run_completed",
                    "Interactive query run completed",
                    {"run_id": self.run_id, "trace_id": trace_id, "record_path": str(record_path), "stats": stats},
                )
                payload = build_payload(
                    status="completed",
                    ended_at=ended_at,
                    elapsed=elapsed,
                    history=serialized_history,
                    stats=stats,
                )
                write_payload(payload)
                update_trace_metadata(
                    execution={"status": "completed", "success": True, "error": None, "ended_at": ended_at, "elapsed": elapsed},
                    stats={**stats, "tool_summary": summary},
                )
                logger.info("Completed interactive query run path=%s", record_path)
                return InteractiveRunResult(status="completed", record_path=record_path, payload=payload)
            except asyncio.CancelledError as exc:
                ended_at = datetime.now().isoformat(timespec="milliseconds")
                elapsed = round(time.monotonic() - started_monotonic, 3)
                await emit(
                    "run_cancelled",
                    "Interactive query run cancelled",
                    {"run_id": self.run_id, "trace_id": trace_id, "record_path": str(record_path), "elapsed": elapsed},
                )
                payload = build_payload(
                    status="cancelled",
                    ended_at=ended_at,
                    elapsed=elapsed,
                    error=_error_payload(exc),
                )
                write_payload(payload)
                update_trace_metadata(
                    execution={"status": "cancelled", "success": False, "error": "cancelled", "ended_at": ended_at, "elapsed": elapsed},
                )
                logger.info("Cancelled interactive query run path=%s", record_path)
                return InteractiveRunResult(status="cancelled", record_path=record_path, payload=payload)
            except (
                SearchAgentError,
                OSError,
                TimeoutError,
                ValueError,
            ) as exc:
                ended_at = datetime.now().isoformat(timespec="milliseconds")
                elapsed = round(time.monotonic() - started_monotonic, 3)
                error = _error_payload(exc)
                await emit(
                    "run_failed",
                    f"Interactive query run failed: {exc}",
                    {"run_id": self.run_id, "trace_id": trace_id, "record_path": str(record_path), "error": error, "elapsed": elapsed},
                )
                payload = build_payload(
                    status="failed",
                    ended_at=ended_at,
                    elapsed=elapsed,
                    error=error,
                )
                write_payload(payload)
                update_trace_metadata(
                    execution={"status": "failed", "success": False, "error": str(exc), "ended_at": ended_at, "elapsed": elapsed},
                )
                logger.exception("Interactive query run failed query=%r", query[:120])
                return InteractiveRunResult(status="failed", record_path=record_path, payload=payload)
