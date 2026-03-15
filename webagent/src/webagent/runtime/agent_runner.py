from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING
from uuid import uuid4

from webagent.agent.agent import Agent
from webagent.log import configure_run_logging, get_logger, log_context, update_trace_metadata
from webagent.utils.config import instantiate
from webagent.utils.retry import retry_async, RetryPolicy

if TYPE_CHECKING:
    from webagent.llm.chat_types import ChatMessage

logger = get_logger(__name__)


def _preview_query(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _serialize_message(message: Any) -> Any:
    if is_dataclass(message):
        return asdict(message)
    return message


def _history_stats(history: list[ChatMessage]) -> dict[str, int]:
    turns = 0
    tool_calls = 0
    tool_messages = 0

    for message in history:
        if message.role == "assistant":
            turns += 1
            message_tool_calls = message.tool_calls or []
            tool_calls += len(message_tool_calls)
        elif message.role == "tool":
            tool_messages += 1

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
    }


def _tool_summary(history: list[ChatMessage]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for message in history:
        if message.role != "assistant":
            continue
        for tool_call in message.tool_calls or []:
            summary[tool_call.name] = summary.get(tool_call.name, 0) + 1
    return summary


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:6]}"


def _make_trace_id() -> str:
    return uuid4().hex[:12]


class AgentRunner:
    """
    Agent Runner:
    - Manage runtime resources
    - Create agent (from code or config)
    - Accept and run agent tasks (with concurrency limit if provided)
    - Manage retry and logging
    """

    def __init__(
        self,
        build_agent: Callable[[], Agent] | None = None,
        agent_config: Any = None,
        max_concurrency: int | None = None,
    ):
        if not build_agent and not agent_config:
            raise ValueError("Either build_agent or agent_config must be provided")
        if build_agent and agent_config:
            logger.warning("Both build_agent and agent_config were provided; using build_agent")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")

        self.build_agent = build_agent or (
            lambda: instantiate(
                cfg=agent_config,
                recursive=True,
                resolve_imports=True,
            )
        )
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        logger.info("AgentRunner initialized max_concurrency=%s", self.max_concurrency)

    @asynccontextmanager
    async def _limit(self):
        if self._semaphore is None:
            yield
            return
        async with self._semaphore:
            yield

    def submit(
        self,
        query: str,
        extra: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> asyncio.Task[list[ChatMessage]]:
        logger.info("Received request query=%r has_extra=%s", _preview_query(query), bool(extra))

        async def _run():
            async with self._limit():
                logger.info("Starting agent execution query=%r", _preview_query(query))
                agent = self.build_agent()
                try:
                    history = await agent.run(query, extra=extra)
                except Exception:
                    logger.exception("Agent execution failed query=%r", _preview_query(query))
                    raise
                logger.info(
                    "Agent execution completed query=%r messages=%s",
                    _preview_query(query),
                    len(history),
                )
                return history

        if retry_policy:
            return asyncio.create_task(
                retry_async(
                    _run,
                    policy=retry_policy,
                    op_name="agent_runner.submit",
                    log=logger,
                )
            )
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

        logger.info("Submitting batch requests count=%s", len(query_list))
        tasks = [
            self.submit(query, extra=extra, retry_policy=retry_policy)
            for query, extra in zip(query_list, extra_list)
        ]
        return tasks

    async def run(
        self,
        *,
        cfg: Any = None,
        data_source: Iterable[tuple[str, dict[str, Any] | None, Any | None]] | None = None,
        output_path: str | Path | None = None,
        retry_policy: RetryPolicy | None = None,
        overwrite_output: bool = False,
    ) -> dict[str, Any]:
        """
        Run a batch of agent tasks and persist per-sample trajectories plus summary stats.

        Inputs can be provided either through `cfg` or through explicit parameters.
        The effective runtime values are resolved from the union of both sources:
        `data_source` and `output_path` are required after merging, while
        `retry_policy` and `overwrite_output` are optional. When both `cfg` and
        explicit parameters provide the same field, `cfg` takes precedence. This
        matches the intended usage where `cfg` defines the active experiment setup
        and explicit parameters are mainly for secondary integrations or custom
        wrappers.

        Args:
            cfg: Optional config object. If present, `cfg.data_source`,
                `cfg.output_path`, `cfg.retry_policy`, and
                `cfg.overwrite_output` are used to fill runtime values, with
                `data_source` and `output_path` treated as required.
            data_source: Optional iterable yielding `(prompt, extra, answer)` tuples.
                Used directly unless `cfg.data_source` is provided.
            output_path: Optional output directory for trajectory files and
                `summary.json`. Used unless `cfg.output_path` is provided.
            retry_policy: Optional retry policy for each agent execution. Used
                unless `cfg.retry_policy` is provided, in which case the policy is
                instantiated from config.
            overwrite_output: Whether to overwrite existing per-sample output
                files.

        Returns:
            A summary dictionary containing counts and aggregate statistics such as
            completed samples, failed samples, total turns, total tool calls, and
            their averages.

        Raises:
            ValueError: If merged inputs still do not provide required values.
        """

        # Process cfg / params

        cfg_values: dict[str, Any] = {}
        if cfg is not None:
            cfg_data_source = cfg.get("data_source")
            if cfg_data_source is not None:
                cfg_values["data_source"] = instantiate(
                    cfg=cfg_data_source,
                    recursive=True,
                    resolve_imports=True,
                )

            cfg_output_path = cfg.get("output_path")
            if cfg_output_path is not None:
                cfg_values["output_path"] = cfg_output_path

            cfg_retry_policy = cfg.get("retry_policy")
            if cfg_retry_policy is not None:
                cfg_values["retry_policy"] = instantiate(
                    cfg=cfg_retry_policy,
                    recursive=True,
                    resolve_imports=True,
                )

            cfg_overwrite_output = cfg.get("overwrite_output")
            if cfg_overwrite_output is not None:
                cfg_values["overwrite_output"] = bool(cfg_overwrite_output)
            cfg_logging = cfg.get("logging")
            if cfg_logging is not None:
                cfg_values["logging"] = cfg_logging

        explicit_values = {
            "data_source": data_source,
            "output_path": output_path,
            "retry_policy": retry_policy,
            "overwrite_output": overwrite_output,
            "logging": None,
        }

        overlap_fields = [
            field
            for field, explicit_value in explicit_values.items()
            if field in cfg_values and explicit_value is not None and explicit_value != cfg_values[field]
        ]
        if overlap_fields:
            # cfg usually represents the intended experiment setup, while explicit
            # parameters are secondary overrides used by downstream integrations.
            logger.warning(
                "Both cfg and explicit parameters were provided for %s; cfg values take precedence",
                overlap_fields,
            )

        resolved_values = dict(explicit_values)
        resolved_values.update(cfg_values)

        missing_fields: list[str] = []
        if resolved_values["data_source"] is None:
            missing_fields.append("data_source")
        if resolved_values["output_path"] is None:
            missing_fields.append("output_path")
        if missing_fields:
            raise ValueError(
                f"run missing required values after merging cfg and explicit parameters: {', '.join(missing_fields)}"
            )

        data_source = resolved_values["data_source"]
        resolved_retry_policy = resolved_values["retry_policy"]
        resolved_overwrite_output = bool(resolved_values["overwrite_output"])
        output_dir = Path(resolved_values["output_path"])
        logging_cfg = resolved_values["logging"]

        # Begin agent run

        output_dir.mkdir(parents=True, exist_ok=True)
        configure_run_logging(output_dir=output_dir, cfg=logging_cfg)
        run_id = _make_run_id()

        with log_context(scope="global", run_id=run_id):
            logger.info("Starting agent batch run output_dir=%s", output_dir)

            summary: dict[str, Any] = {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "total": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "total_turns": 0,
                "total_tool_calls": 0,
                "avg_turns": 0.0,
                "avg_tool_calls": 0.0,
            }

            async def _run_one(
                index: int,
                prompt: str,
                extra: dict[str, Any] | None,
                answer: Any,
                record_path: Path,
                trace_id: str,
            ) -> dict[str, Any]:
                sample_id = f"{index:06d}"
                started_at = datetime.now().isoformat(timespec="milliseconds")
                started_monotonic = time.monotonic()
                with log_context(
                    scope="trace",
                    run_id=run_id,
                    sample_id=sample_id,
                    trace_id=trace_id,
                    turn="-",
                ):
                    update_trace_metadata(
                        run_id=run_id,
                        sample_id=sample_id,
                        run={"run_id": run_id},
                        sample={
                            "index": index,
                            "sample_id": sample_id,
                            "trace_id": trace_id,
                            "query": prompt,
                            "record_path": str(record_path),
                            "started_at": started_at,
                        },
                        execution={
                            "status": "running",
                            "success": None,
                            "error": None,
                        },
                    )
                    logger.info(
                        "Starting sample execution index=%s output=%s query=%r",
                        index,
                        record_path,
                        _preview_query(prompt),
                    )
                    try:
                        history = await self.submit(
                            prompt,
                            extra=extra,
                            retry_policy=resolved_retry_policy,
                        )
                        stats = _history_stats(history)
                        tool_summary = _tool_summary(history)
                        ended_at = datetime.now().isoformat(timespec="milliseconds")
                        elapsed = round(time.monotonic() - started_monotonic, 3)
                        payload = {
                            "index": index,
                            "trace_id": trace_id,
                            "input": prompt,
                            "extra": extra,
                            "answer": answer,
                            "history": [_serialize_message(message) for message in history],
                            "stats": stats,
                        }
                        record_path.write_text(
                            json.dumps(payload, ensure_ascii=False, default=str),
                            encoding="utf-8",
                        )
                        update_trace_metadata(
                            execution={
                                "status": "completed",
                                "success": True,
                                "error": None,
                                "ended_at": ended_at,
                                "elapsed": elapsed,
                            },
                            stats={
                                **stats,
                                "tool_summary": tool_summary,
                            },
                        )
                    except Exception as exc:
                        ended_at = datetime.now().isoformat(timespec="milliseconds")
                        elapsed = round(time.monotonic() - started_monotonic, 3)
                        update_trace_metadata(
                            execution={
                                "status": "failed",
                                "success": False,
                                "error": str(exc),
                                "ended_at": ended_at,
                                "elapsed": elapsed,
                            },
                        )
                        logger.exception(
                            "Sample execution failed index=%s query=%r",
                            index,
                            _preview_query(prompt),
                        )
                        raise
                    logger.info(
                        "Completed sample execution index=%s path=%s turns=%s tool_calls=%s",
                        index,
                        record_path,
                        stats["turns"],
                        stats["tool_calls"],
                    )
                    return stats

            scheduled_tasks: list[tuple[asyncio.Task[dict[str, Any]], int, str]] = []
            for index, (prompt, extra, answer) in enumerate(data_source):
                summary["total"] += 1
                record_path = output_dir / f"{index:06d}.json"
                if record_path.exists() and not resolved_overwrite_output:
                    logger.info("Skipping existing output index=%s path=%s", index, record_path)
                    summary["skipped"] += 1
                    continue

                trace_id = _make_trace_id()
                task = asyncio.create_task(
                    _run_one(index, prompt, extra, answer, record_path, trace_id),
                    name=f"sample-{index:06d}",
                )
                scheduled_tasks.append((task, index, trace_id))

            logger.info("Scheduled batch requests count=%s", len(scheduled_tasks))

            if scheduled_tasks:
                task_list = [task for task, _, _ in scheduled_tasks]
                results = await asyncio.gather(*task_list, return_exceptions=True)
                for (task, index, trace_id), result in zip(scheduled_tasks, results):
                    if isinstance(result, Exception):
                        summary["failed"] += 1
                        logger.error(
                            "Agent batch item failed index=%s trace_id=%s task=%s error=%r",
                            index,
                            trace_id,
                            task.get_name(),
                            result,
                        )
                        continue
                    summary["completed"] += 1
                    summary["total_turns"] += result["turns"]
                    summary["total_tool_calls"] += result["tool_calls"]

            completed = summary["completed"]
            if completed:
                summary["avg_turns"] = summary["total_turns"] / completed
                summary["avg_tool_calls"] = summary["total_tool_calls"] / completed

            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(
                "Completed agent batch run total=%s completed=%s failed=%s skipped=%s avg_turns=%.3f avg_tool_calls=%.3f",
                summary["total"],
                summary["completed"],
                summary["failed"],
                summary["skipped"],
                summary["avg_turns"],
                summary["avg_tool_calls"],
            )
            return summary
