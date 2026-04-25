from __future__ import annotations

import asyncio
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

try:
    import uvloop
except ImportError:
    uvloop = None

if uvloop is not None:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from searchagent.log import get_logger, setup_logger
from searchagent.runtime.runner import AgentRunner, RunConfig
from searchagent.runtime.evaluate import evaluate_main
from searchagent.agent.search_agent import SearchAgentConfig
from searchagent.llm.base import (
    AnthropicConfig,
    ClientConfig,
    DashScopeConfig,
    OllamaConfig,
    OpenAIConfig,
    TransformersConfig,
    VllmConfig,
)
from searchagent.llm.parsers import ParserConfig, QwenParserConfig
from searchagent.tools.base import ToolConfig
from searchagent.sources import SourceConfig
from searchagent.common.retry import RetryConfig
from searchagent.common.dataloader import DataConfig

logger = get_logger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)
cs.store(group="agent", name="SearchAgent", node=SearchAgentConfig)
cs.store(group="llm", name="OpenAIClient", node=ClientConfig)
cs.store(group="llm", name="AnthropicClient", node=ClientConfig)
cs.store(group="llm", name="DashScopeClient", node=ClientConfig)
cs.store(group="llm", name="VllmClient", node=ClientConfig)
cs.store(group="llm", name="OllamaClient", node=ClientConfig)
cs.store(group="llm", name="TransformersClient", node=ClientConfig)
cs.store(group="llm", name="QwenParser", node=ParserConfig)
cs.store(group="common", name="RetryPolicy", node=RetryConfig)
cs.store(group="common", name="GenericDataLoader", node=DataConfig)

# Register nested configs so OmegaConf.structured can resolve them
# when building the ConfigStore schemas
cs.store(name="__openai_config__", node=OpenAIConfig)
cs.store(name="__anthropic_config__", node=AnthropicConfig)
cs.store(name="__dashscope_config__", node=DashScopeConfig)
cs.store(name="__vllm_config__", node=VllmConfig)
cs.store(name="__ollama_config__", node=OllamaConfig)
cs.store(name="__transformers_config__", node=TransformersConfig)
cs.store(name="__qwen_parser_config__", node=QwenParserConfig)

# Register a concrete ToolConfig instance (not the class) so Hydra can merge
# tool list items against a concrete schema.
cs.store(name="__tool_config__", node=ToolConfig())
cs.store(name="__source_config__", node=SourceConfig())

async def _run(cfg: DictConfig) -> None:
    config: RunConfig = OmegaConf.to_object(cfg)
    async with AgentRunner(config=config) as runner:
        await runner.run(cfg=cfg)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logger()
    asyncio.run(_run(cfg))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        sys.argv.pop(1)
        evaluate_main()
    else:
        main()
