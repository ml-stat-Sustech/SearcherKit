from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from searchagent.agent.search_agent import SearchAgent, SearchAgentConfig
from searchagent.log import configure_run_logging, get_logger, log_context, update_trace_metadata
from searchagent.common.dataloader import DataConfig, DataItem, GenericDataLoader
from searchagent.errors import SearchAgentError
from searchagent.runtime import startup
from searchagent.common.retry import retry_async, RetryConfig, RetryPolicy
from searchagent.runtime.batch import BatchItem, BatchSummary
from searchagent.runtime.checkpoint import CheckpointConfig, CheckpointStore
from searchagent.runtime.errors import CheckpointError
from searchagent.runtime.trace import (
    history_stats as _history_stats,
    make_run_id as _make_run_id,
    make_trace_id as _make_trace_id,
    preview_query as _preview_query,
    serialize_message as _serialize_message,
    tool_summary as _tool_summary,
)

if TYPE_CHECKING:
    from searchagent.common.messages import ChatMessage

logger = get_logger(__name__)


@dataclass
class RunConfig:
    agent: SearchAgentConfig = field(default_factory=SearchAgentConfig)
    max_concurrency: int | None = None
    dataloader: DataConfig | None = None
    output_path: str | None = None
    retry_policy: RetryConfig | None = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    overwrite_output: bool = False
    logging: dict[str, Any] | None = None
    auto_startup: Any | None = None

class AgentRunner:
    """
    Agent Runner:
    - Manage runtime resources
    - Create agent (from config)
    - Accept and run agent tasks (with concurrency limit if provided)
    - Manage retry and logging
    """
    def __init__(
        self,
        *,
        config: RunConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("config must be provided")
        if config.max_concurrency is not None and config.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")

        self.build_agent = lambda: SearchAgent(config=config.agent)
        self.cfg = config
        self.max_concurrency = config.max_concurrency
        self._semaphore = asyncio.Semaphore(config.max_concurrency) if config.max_concurrency else None
        self.run_id = 0

    async def init(self):
        await startup.check_and_start(self.cfg)
        logger.info("AgentRunner initialized max_concurrency=%s", self.max_concurrency)

    async def close(self):
        logger.info("Closing AgentRunner")
        await startup.shutdown()
    
    async def __aenter__(self) -> "AgentRunner":
        await self.init()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

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
                self.run_id += 1
                history = await agent.run(query, extra=extra, session_id=self.run_id)
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
                    op_name="runner.submit",
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
        cfg: RunConfig | None = None,
        dataloader: Iterable[tuple[str, dict[str, Any] | None, Any | None]] | None = None,
        output_path: str | Path | None = None,
        retry_policy: RetryPolicy | None = None,
        checkpoint: CheckpointConfig | dict[str, Any] | None = None,
        overwrite_output: bool | None = None,
    ) -> dict[str, Any]:
        """
        Run a batch of agent tasks and persist per-sample trajectories plus summary stats.

        Inputs can be provided either through `cfg` or through explicit parameters.
        The effective runtime values are resolved from the union of both sources:
        `dataloader` and `output_path` are required after merging, while
        `retry_policy` and `overwrite_output` are optional. When both `cfg` and
        explicit parameters provide the same field, `cfg` takes precedence. This
        matches the intended usage where `cfg` defines the active experiment setup
        and explicit parameters are mainly for secondary plugins or custom
        wrappers.

        Args:
            cfg: Optional config object. If present, `cfg.dataloader`,
                `cfg.output_path`, `cfg.retry_policy`, and
                `cfg.overwrite_output` are used to fill runtime values, with
                `dataloader` and `output_path` treated as required.
            dataloader: Optional iterable yielding `(prompt, extra, answer)` tuples.
                Used directly unless `cfg.dataloader` is provided.
            output_path: Optional output directory for trajectory files and
                `summary.json`. Used unless `cfg.output_path` is provided.
            retry_policy: Optional retry policy for each agent execution. Used
                unless `cfg.retry_policy` is provided, in which case the policy is
                instantiated from config.
            checkpoint: Optional checkpoint config. Used unless `cfg.checkpoint`
                is provided.
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
            cfg_dataloader = cfg.dataloader
            if cfg_dataloader is not None:
                cfg_values["dataloader"] = GenericDataLoader(config=cfg_dataloader)

            cfg_output_path = cfg.output_path
            if cfg_output_path is not None:
                cfg_values["output_path"] = cfg_output_path

            cfg_retry_policy = cfg.retry_policy
            if cfg_retry_policy is not None:
                cfg_values["retry_policy"] = RetryPolicy(config=cfg_retry_policy)
                

            cfg_overwrite_output = cfg.overwrite_output
            if cfg_overwrite_output is not None:
                cfg_values["overwrite_output"] = cfg_overwrite_output
            cfg_checkpoint = cfg.checkpoint
            if cfg_checkpoint is not None:
                cfg_values["checkpoint"] = cfg_checkpoint
            cfg_logging = cfg.logging
            if cfg_logging is not None:
                cfg_values["logging"] = cfg_logging

        base_checkpoint = (
            checkpoint
            if checkpoint is not None
            else (getattr(self.cfg, "checkpoint", None) if cfg is None else None)
        )
        explicit_values = {
            "dataloader": dataloader,
            "output_path": output_path,
            "retry_policy": retry_policy,
            "checkpoint": base_checkpoint,
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
            # parameters are secondary overrides used by downstream plugins.
            logger.warning(
                "Both cfg and explicit parameters were provided for %s; cfg values take precedence",
                overlap_fields,
            )

        resolved_values = dict(explicit_values)
        resolved_values.update(cfg_values)

        missing_fields: list[str] = []
        if resolved_values["dataloader"] is None:
            missing_fields.append("dataloader")
        if resolved_values["output_path"] is None:
            missing_fields.append("output_path")
        if missing_fields:
            raise ValueError(
                f"run missing required values after merging cfg and explicit parameters: {', '.join(missing_fields)}"
            )

        dataloader = resolved_values["dataloader"]
        resolved_retry_policy = resolved_values["retry_policy"]
        resolved_overwrite_output = bool(resolved_values["overwrite_output"])
        output_dir = Path(resolved_values["output_path"])
        checkpoint_cfg = resolved_values["checkpoint"]
        logging_cfg = resolved_values["logging"]

        # Begin agent run

        output_dir.mkdir(parents=True, exist_ok=True)
        configure_run_logging(output_dir=output_dir, cfg=logging_cfg)
        run_id = _make_run_id()
        checkpoint_store = CheckpointStore.from_output_dir(output_dir, config=checkpoint_cfg)
        if checkpoint_store is not None:
            await checkpoint_store.start_run(run_id=run_id, output_dir=output_dir)

        with log_context(scope="global", run_id=run_id):
            logger.info("Starting agent batch run output_dir=%s", output_dir)

            summary = BatchSummary(
                run_id=run_id,
                output_dir=output_dir,
                checkpoint_path=str(checkpoint_store.path) if checkpoint_store is not None else None,
            )

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
                    if checkpoint_store is not None:
                        await checkpoint_store.mark_started(
                            sample_id=sample_id,
                            run_id=run_id,
                            trace_id=trace_id,
                            started_at=started_at,
                        )
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
                            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
                            encoding="utf-8"
                        )
                        if checkpoint_store is not None:
                            await checkpoint_store.mark_completed(
                                sample_id=sample_id,
                                stats=stats,
                                record_path=record_path,
                                ended_at=ended_at,
                                elapsed=elapsed,
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
                    except (SearchAgentError, OSError, TimeoutError, ValueError, RuntimeError) as exc:
                        ended_at = datetime.now().isoformat(timespec="milliseconds")
                        elapsed = round(time.monotonic() - started_monotonic, 3)
                        if checkpoint_store is not None:
                            try:
                                await checkpoint_store.mark_failed(
                                    sample_id=sample_id,
                                    error=str(exc),
                                    ended_at=ended_at,
                                    elapsed=elapsed,
                                )
                            except CheckpointError as checkpoint_exc:
                                logger.error(
                                    "Failed to persist checkpoint failure state sample_id=%s error=%s",
                                    sample_id,
                                    checkpoint_exc,
                                )
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

            start_time = time.time()

            dataloader_iter = enumerate(dataloader)
            dataloader_exhausted = False
            max_in_flight = self.max_concurrency or 32
            scheduled_count = 0
            in_flight: dict[asyncio.Task[dict[str, Any]], BatchItem] = {}
            failure_types = (
                SearchAgentError,
                OSError,
                TimeoutError,
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
            )

            async def _schedule_next() -> bool:
                nonlocal dataloader_exhausted, scheduled_count
                while not dataloader_exhausted:
                    try:
                        index, (prompt, extra, answer) = next(dataloader_iter)
                    except StopIteration:
                        dataloader_exhausted = True
                        return False

                    summary.total += 1
                    sample_id = f"{index:06d}"
                    record_path = output_dir / f"{index:06d}.json"
                    if (
                        checkpoint_store is not None
                        and not resolved_overwrite_output
                        and checkpoint_store.is_completed(sample_id, record_path)
                    ):
                        logger.info(
                            "Skipping checkpoint-completed output index=%s path=%s",
                            index,
                            record_path,
                        )
                        summary.mark_skip("checkpoint_completed")
                        continue
                    if record_path.exists() and not resolved_overwrite_output:
                        logger.info("Skipping existing output index=%s path=%s", index, record_path)
                        summary.mark_skip("existing_output")
                        continue

                    checkpoint_sample = (
                        checkpoint_store.sample_state(sample_id)
                        if checkpoint_store is not None
                        else None
                    )
                    trace_id = (
                        str(checkpoint_sample.get("trace_id"))
                        if checkpoint_sample and checkpoint_sample.get("trace_id")
                        else _make_trace_id()
                    )
                    if checkpoint_store is not None:
                        if checkpoint_store.is_resume_candidate(sample_id):
                            summary.resumed += 1
                        await checkpoint_store.mark_pending(
                            sample_id=sample_id,
                            index=index,
                            trace_id=trace_id,
                            record_path=record_path,
                            prompt=prompt,
                        )
                    item = BatchItem(
                        index=index,
                        sample_id=sample_id,
                        prompt=prompt,
                        extra=extra,
                        answer=answer,
                        record_path=record_path,
                        trace_id=trace_id,
                    )
                    task = asyncio.create_task(
                        _run_one(index, prompt, extra, answer, record_path, trace_id),
                        name=f"sample-{index:06d}",
                    )
                    in_flight[task] = item
                    scheduled_count += 1
                    return True
                return False

            while len(in_flight) < max_in_flight and await _schedule_next():
                pass

            logger.info(
                "Scheduled initial batch requests count=%s max_in_flight=%s",
                len(in_flight),
                max_in_flight,
            )

            while in_flight:
                done, _ = await asyncio.wait(
                    in_flight.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    item = in_flight.pop(task)
                    try:
                        result = task.result()
                    except failure_types as exc:
                        summary.failed += 1
                        logger.error(
                            "Agent batch item failed index=%s trace_id=%s task=%s error=%r",
                            item.index,
                            item.trace_id,
                            task.get_name(),
                            exc,
                        )
                        continue
                    summary.add_completed(result)

                while len(in_flight) < max_in_flight and await _schedule_next():
                    pass

            logger.info("Scheduled batch requests count=%s", scheduled_count)

            summary_payload = summary.finalize(time.time() - start_time)

            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info(
                "Completed agent batch run total=%s in %.3f sec. completed=%s failed=%s skipped=%s avg_turns=%.3f avg_tool_calls=%.3f",
                summary_payload["total"],
                summary_payload["time_elapsed"],
                summary_payload["completed"],
                summary_payload["failed"],
                summary_payload["skipped"],
                summary_payload["avg_turns"],
                summary_payload["avg_tool_calls"],
            )
            return summary_payload
