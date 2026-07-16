from __future__ import annotations

import asyncio
import time
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.fully_async_rollout import _get_global_worker
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

from searcherkit.common.log import get_logger

logger = get_logger(__name__)


def _flatten_samples(node: Any) -> list[Sample]:
    if isinstance(node, Sample):
        return [node]
    if isinstance(node, list):
        out: list[Sample] = []
        for item in node:
            out.extend(_flatten_samples(item))
        return out
    return []


def _rollout_id_count(groups: list[Any]) -> int:
    rollout_ids = set()
    for group in groups:
        for sample in _flatten_samples(group):
            rollout_ids.add(sample.rollout_id if sample.rollout_id is not None else sample.index)
    return len(rollout_ids)


async def _generate_rollout_async(args: Any, rollout_id: int, data_buffer: Any) -> RolloutFnTrainOutput:
    """Fully-async rollout with the same dynamic filtering as sync rollout."""

    if not args.rollout_global_dataset:
        raise ValueError("fully-async rollout requires rollout_global_dataset")

    worker = _get_global_worker(args, data_buffer)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )
    metric_gatherer = MetricGatherer()
    target = args.rollout_batch_size
    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started

    logger.info(
        "searcherkit fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )
    while len(collected) < target:
        drained = 0
        for gid, group in worker.get_completed_groups():
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue
            collected[gid] = group
            drained += 1
            if len(collected) >= target:
                break

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > 30:
            logger.info(
                "searcherkit fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    def _key(group: list[Sample]) -> int:
        for sample in group:
            idx = getattr(sample, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    data = sorted(collected.values(), key=_key)[:target]
    rollout_count = _rollout_id_count(data)
    min_rollouts = args.rollout_batch_size * args.n_samples_per_prompt
    if rollout_count < min_rollouts:
        logger.warning(
            "searcherkit fully-async rollout %d returned %d rollout ids for %d groups; expected at least %d",
            rollout_id,
            rollout_count,
            len(data),
            min_rollouts,
        )
    logger.info(
        "searcherkit fully-async rollout %d: done in %.1fs, queue_left=%d, metrics=%s",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
        metric_gatherer.collect(),
    )
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())


def generate_rollout_fully_async(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    if evaluation:
        raise ValueError("fully-async rollout does not support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
