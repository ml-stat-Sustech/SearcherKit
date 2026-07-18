from __future__ import annotations

from dataclasses import is_dataclass
from importlib.resources import files
from pathlib import Path
from typing import Sequence, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from searcherkit.agent import SearchAgentConfig
from searcherkit.common.dataloader import DataConfig
from searcherkit.common.errors import FatalError
from searcherkit.common.retry import RetryConfig
from searcherkit.llm.base import ClientConfig
from searcherkit.llm.parsers import ParserConfig
from searcherkit.runtime.runner import RunConfig
from searcherkit.sources import SourceConfig
from searcherkit.tools.base import SummarizerConfig, ToolConfig


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
    cs.store(name="__tool_config__", node=ToolConfig())
    cs.store(name="__summarizer_config__", node=SummarizerConfig())
    cs.store(name="__source_config__", node=SourceConfig())
    _CONFIG_STORE_REGISTERED = True


def default_config_dir() -> Path:
    return Path(str(files("searcherkit").joinpath("config")))


def resolve_config_file(
    config_path: str | Path | None,
    *,
    default: Path | None = None,
) -> Path:
    packaged_dir = default or default_config_dir()
    if config_path is None:
        candidates = [packaged_dir / "config.yaml"]
    else:
        requested = Path(config_path).expanduser()
        local_candidates = [requested]
        if requested.suffix == "":
            local_candidates.append(requested.with_suffix(".yaml"))
        candidates = local_candidates
        if not requested.is_absolute():
            packaged = packaged_dir / requested
            candidates.append(packaged)
            if packaged.suffix == "":
                candidates.append(packaged.with_suffix(".yaml"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"config file does not exist; searched: {searched}")


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
    overrides: Sequence[str] | None = None,
    default_config_dir: Path | None = None,
) -> DictConfig:
    register_config_store()
    config_file = resolve_config_file(config_path, default=default_config_dir)
    with initialize_config_dir(config_dir=str(config_file.parent), version_base=None):
        return compose(config_name=config_file.stem, overrides=clean_overrides(overrides))


def compose_dataclass_config(
    config_type: type[T],
    *,
    config_path: str | Path | None = None,
    overrides: Sequence[str] | None = None,
    default_config_dir: Path | None = None,
) -> T:
    if not is_dataclass(config_type):
        raise TypeError(f"config_type must be a dataclass type, got {config_type!r}")
    cfg = compose_config(
        config_path=config_path,
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
        raise ConfigError(f"resolved config is not a {config_type.__name__}: {config_path or 'config.yaml'}")
    return obj
