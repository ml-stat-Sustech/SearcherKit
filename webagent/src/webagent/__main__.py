from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from webagent.log import get_logger, setup_logger
from webagent.runtime.agent_runner import AgentRunner
from webagent.runtime.evaluate import evaluate_main
from webagent.utils.config import instantiate

from webagent.llm.chat_types import ChatMessage

logger = get_logger(__name__)

def _serialize_message(message: ChatMessage) -> Any:
    if is_dataclass(message):
        return asdict(message)
    return message

def _save_to_path(path: Path, index: int, input: str, answer: str, history: list[ChatMessage]) -> None:
    path.write_text(json.dumps({
        "input": input,
        "answer": answer,
        "history": [_serialize_message(msg) for msg in history],
    }, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Wrote result index=%d path=%s messages=%s", index, path, len(history))

async def _run(cfg: DictConfig) -> None:
    data_source_cfg = cfg.get("data_source")
    data_source = instantiate(cfg=data_source_cfg, recursive=True, resolve_imports=True)

    output_dir = Path(cfg.get("output_path") or "outputs/agent_history")
    logger.info("Starting webagent batch run output_dir=%s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = cfg.get("agent")
    runner = AgentRunner(agent_config=agent_cfg)

    tasks: list[asyncio.Future[list[Any]]] = []
    pbar = tqdm(total=0)

    for index, (prompt, extra, answer) in enumerate(data_source):
        output_path = output_dir / f"{index:06d}.json"
        if output_path.exists():
            logger.info("Skipping existing output index=%s path=%s", index, output_path)
            continue
        task = runner.submit(prompt, extra=extra)
        def _on_done(
            done_task: asyncio.Future[list[Any]],
            *,
            output_path: Path = output_path,
            index: int = index,
            prompt: str = prompt,
            answer: str = answer,
        ) -> None:
            _save_to_path(output_path, index, prompt, answer, done_task.result())
            pbar.update(1)
        
        task.add_done_callback(_on_done)
        tasks.append(task)
    logger.info("Scheduled requests count=%s", len(tasks))
    pbar.reset(total=len(tasks))

    if tasks:
        await asyncio.gather(*tasks)
    logger.info("Completed webagent batch run count=%s", len(tasks))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logger()
    asyncio.run(_run(cfg))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv.pop(1)
        evaluate_main()
    else:
        main()
