from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from searchagent.agent.search_agent import SearchAgentConfig
from searchagent.common.dataloader import DataConfig
from searchagent.common.errors import FatalError
from searchagent.common.retry import RetryConfig
from searchagent.llm.base import (
    AnthropicConfig,
    ClientConfig,
    OpenAIConfig,
    VllmConfig,
)
from searchagent.llm.parsers import ParserConfig
from searchagent.runtime.runner import RunConfig
from searchagent.sources import SourceConfig
from searchagent.tools.base import SummarizerConfig, ToolConfig


_CONFIG_STORE_REGISTERED = False
T = TypeVar("T")


class ConfigError(FatalError):
    """Configuration is missing, invalid, or points to an unknown component."""


def register_config_store() -> None:
    global _CONFIG_STORE_REGISTERED
    if _CONFIG_STORE_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="config", node=RunConfig)
    cs.store(group="agent", name="SearchAgent", node=SearchAgentConfig)
    cs.store(group="llm", name="OpenAIClient", node=ClientConfig)
    cs.store(group="llm", name="AnthropicClient", node=ClientConfig)
    cs.store(group="llm", name="VllmClient", node=ClientConfig)
    cs.store(group="llm", name="TongyiDeepResearchParser", node=ParserConfig)
    cs.store(group="llm", name="UpstreamParser", node=ParserConfig)
    cs.store(group="common", name="RetryPolicy", node=RetryConfig)
    cs.store(group="common", name="GenericDataLoader", node=DataConfig)
    cs.store(name="__openai_config__", node=OpenAIConfig)
    cs.store(name="__anthropic_config__", node=AnthropicConfig)
    cs.store(name="__vllm_config__", node=VllmConfig)
    cs.store(name="__tool_config__", node=ToolConfig())
    cs.store(name="__summarizer_config__", node=SummarizerConfig())
    cs.store(name="__source_config__", node=SourceConfig())
    _CONFIG_STORE_REGISTERED = True


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def resolve_config_dir(
    config_path: str | Path | None,
    *,
    default: Path | None = None,
) -> Path:
    if config_path is None:
        return default or default_config_dir()
    return Path(config_path).expanduser().resolve()


def clean_overrides(overrides: Sequence[str] | None) -> list[str]:
    if not overrides:
        return []
    cleaned = list(overrides)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned


def compose_config(
    *,
    config_path: str | Path | None = None,
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    default_config_dir: Path | None = None,
) -> DictConfig:
    register_config_store()
    config_dir = resolve_config_dir(config_path, default=default_config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"config path does not exist or is not a directory: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=clean_overrides(overrides))


def compose_dataclass_config(
    config_type: type[T],
    *,
    config_path: str | Path | None = None,
    config_name: str = "config",
    overrides: Sequence[str] | None = None,
    default_config_dir: Path | None = None,
) -> T:
    if not is_dataclass(config_type):
        raise TypeError(f"config_type must be a dataclass type, got {config_type!r}")
    cfg = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
        default_config_dir=default_config_dir,
    )
    try:
        structured = OmegaConf.structured(config_type)
        merged = OmegaConf.merge(structured, cfg)
        obj = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        raise ConfigError(f"invalid {config_type.__name__} config: {exc}") from exc
    if not isinstance(obj, config_type):
        raise ConfigError(f"resolved config is not a {config_type.__name__}: {config_name}")
    return obj
