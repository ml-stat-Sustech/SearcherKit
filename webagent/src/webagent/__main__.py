from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from webagent.log import get_logger, setup_logger
from webagent.runtime.agent_runner import AgentRunner
from webagent.utils.config import instantiate

logger = get_logger(__name__)


def _serialize_message(message: Any) -> Any:
    # TODO: normalize Tool objects and tool_call arguments for JSON-safe, round-trippable history.
    if is_dataclass(message):
        return asdict(message)
    return message

async def _run(cfg: DictConfig) -> None:
    agent_cfg = cfg.get("agent")
    data_source_cfg = cfg.get("data_source")
    output_dir = Path(cfg.get("output_path") or "outputs/agent_history")

    logger.info("Starting webagent batch run output_dir=%s", output_dir)
    data_source = instantiate(cfg=data_source_cfg, recursive=True, resolve_imports=True)
    runner = AgentRunner(agent_config=agent_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[asyncio.Future[list[Any]]] = []
    meta: dict[int, dict[str, Any]] = {}
    for index, (prompt, extra, answer) in enumerate(data_source):
        submitted = runner.submit(prompt, extra=extra)
        task = submitted if isinstance(submitted, asyncio.Task) else asyncio.create_task(submitted)
        tasks.append(task)
        meta[index] = {"index": index, "input": prompt, "answer": answer}
    logger.info("Scheduled requests count=%s", len(tasks))

    for task in asyncio.as_completed(tasks):
        history = await task
        info = meta[index]
        row = {
            "input": info["input"],
            "answer": info["answer"],
            "history": [_serialize_message(msg) for msg in history],
        }
        output_path = output_dir / f"{info['index']:06d}.json"
        output_path.write_text(json.dumps(row, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info("Wrote result index=%s path=%s messages=%s", info["index"], output_path, len(history))

    logger.info("Completed webagent batch run count=%s", len(tasks))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logger()
    asyncio.run(_run(cfg))

if __name__ == "__main__":
    main()
