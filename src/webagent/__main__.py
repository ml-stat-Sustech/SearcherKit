from __future__ import annotations

import asyncio
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from webagent.log import get_logger, setup_logger
from webagent.runtime.agent_runner import AgentRunner, RunConfig
from webagent.runtime.evaluate import evaluate_main
from webagent.agent.webagent import WebAgentConfig
from webagent.llm.client import ClientConfig, OpenAIConfig
from webagent.llm.parser import ParserConfig, QwenParserConfig
from webagent.tools.base import ToolConfig
from webagent.commons.retry import RetryConfig
from webagent.commons.dataloader import DataConfig

logger = get_logger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)
cs.store(group="agent", name="WebAgent", node=WebAgentConfig)
cs.store(group="llm", name="OpenAIClient", node=ClientConfig)
cs.store(group="llm", name="QwenParser", node=ParserConfig)
cs.store(group="commons", name="RetryPolicy", node=RetryConfig)
cs.store(group="commons", name="GenericDataLoader", node=DataConfig)

# Register nested configs so OmegaConf.structured can resolve them
# when building the ConfigStore schemas
cs.store(name="__openai_config__", node=OpenAIConfig)
cs.store(name="__qwen_parser_config__", node=QwenParserConfig)

# Register a concrete ToolConfig instance (not the class) so Hydra can merge
# tool list items against a concrete schema.
cs.store(name="__tool_config__", node=ToolConfig())

async def _run(cfg: DictConfig) -> None:
    config: RunConfig = OmegaConf.to_object(cfg)
    async with AgentRunner(config=config) as runner:
        await runner.run(cfg=cfg)


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
