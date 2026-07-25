"""OmegaConf-friendly instantiation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from importlib.resources import files
from pathlib import Path
import re
from typing import Any, Callable, TypeVar
import warnings

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from searcherkit.common.errors import FatalError


T = TypeVar("T")
_CONFIG_STORE_REGISTERED = False

_PKG_PREFIX = "pkg://"
_FILE_PREFIX = "file://"
_PKG_IMPORT_RE = re.compile(
    r"^pkg://"
    r"(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
    r"(?::(?P<attr>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))?"
    r"$"
)
_FILE_IMPORT_RE = re.compile(
    r"^file://"
    r"(?P<path>.+?\.py)"
    r"(?::(?P<attr>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))?"
    r"$"
)


def _to_container(cfg: Any) -> Any:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if isinstance(cfg, list):
        return list(cfg)
    raise TypeError(f"cfg must be a mapping or DictConfig, got {type(cfg)!r}")


def import_from_path(path: str) -> Any:
    """Import a path like 'pkg://pkg.mod[:attr]' or 'file:///abs/mod.py[:attr]'."""
    if not isinstance(path, str):
        raise ValueError(
            "target must be an import path like "
            f"'{_PKG_PREFIX}pkg.mod[:attr]' or '{_FILE_PREFIX}/abs/path.py[:attr]', got {path!r}"
        )
    pkg_match = _PKG_IMPORT_RE.fullmatch(path)
    file_match = _FILE_IMPORT_RE.fullmatch(path)
    if not pkg_match and not file_match:
        raise ValueError(
            "target must be an import path like "
            f"'{_PKG_PREFIX}pkg.mod[:attr]' or '{_FILE_PREFIX}/abs/path.py[:attr]', got {path!r}"
        )

    if pkg_match:
        module_name = pkg_match.group("module")
        attr_path = pkg_match.group("attr")
        module = import_module(module_name)
    else:
        assert file_match is not None
        file_path = Path(file_match.group("path")).expanduser().resolve()
        attr_path = file_match.group("attr")
        if not file_path.is_file():
            raise FileNotFoundError(f"target file '{file_path}' not found")
        module_name = f"_searcherkit_config_file_{abs(hash(str(file_path)))}"
        spec = spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load module from '{file_path}'")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

    try:
        if not attr_path:
            return module
        value: Any = module
        for attr in attr_path.split("."):
            value = getattr(value, attr)
        return value
    except AttributeError as exc:
        raise AttributeError(f"target '{path}' not found") from exc


def _resolve_imports(value: Any, *, target_key: str) -> Any:
    """Recursively resolve pkg:// and file:// import strings (excluding target key)."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == target_key:
                out[key] = item
                continue
            out[key] = _resolve_imports(item, target_key=target_key)
        return out
    if isinstance(value, list):
        return [_resolve_imports(item, target_key=target_key) for item in value]
    if isinstance(value, str) and (
        value.startswith(_PKG_PREFIX) or value.startswith(_FILE_PREFIX)
    ):
        try:
            return import_from_path(value)
        except (
            ModuleNotFoundError,
            ImportError,
            AttributeError,
            ValueError,
            FileNotFoundError,
        ):
            raise
    return value


def instantiate(
    factory: Optional[Callable[..., T]] = None,
    cfg: Any = None,
    *,
    target_key: str = "target",
    recursive: bool = False,
    resolve_imports: bool = False,
    **kwargs: Any,
) -> T:
    """Deprecated: instantiate an object from a factory and/or OmegaConf config.

    Prefer explicit config dataclass constructors, e.g. `Cls(config=cfg)`, in new
    code. This helper remains for older plugin/source factory paths.

    Args:
        factory: Callable/class to instantiate. If None, config must provide `target`.
        cfg: DictConfig/dict with params and optional `target` import path.
        target_key: Key name for the import path in cfg.
        recursive: If True, recursively instantiate nested configs containing `target`.
        resolve_imports: If True, attempt to import pkg:// strings in config.
        **kwargs: Overrides for cfg values.
    """
    warnings.warn(
        "searcherkit.common.config.instantiate is deprecated; use explicit "
        "config dataclass constructors instead",
        DeprecationWarning,
        stacklevel=2,
    )

    data = _to_container(cfg)
    if resolve_imports:
        data = _resolve_imports(data, target_key=target_key)

    if isinstance(data, list):
        raise TypeError("cfg must be a mapping or DictConfig when instantiating a target")

    target = data.pop(target_key, None)
    if factory is None:
        if target is None:
            raise ValueError("factory is None and cfg has no target")
        factory = import_from_path(target)

    if recursive:
        for key, value in list(data.items()):
            if isinstance(value, Mapping) and target_key in value:
                data[key] = instantiate(
                    cfg=value,
                    target_key=target_key,
                    recursive=True,
                    resolve_imports=resolve_imports,
                )
            elif isinstance(value, list):
                new_list: list[Any] = []
                for item in value:
                    if isinstance(item, Mapping) and target_key in item:
                        new_list.append(
                            instantiate(
                                cfg=item,
                                target_key=target_key,
                                recursive=True,
                                resolve_imports=resolve_imports,
                            )
                        )
                    elif isinstance(item, Mapping):
                        # Recurse into nested mappings without targets.
                        new_list.append(
                            _resolve_imports(item, target_key=target_key)
                            if resolve_imports
                            else dict(item)
                        )
                    else:
                        new_list.append(item)
                data[key] = new_list

    data.update(kwargs)
    return factory(**data)  # type: ignore[misc]


class FromOmegaConfigMixin:
    """Mixin to construct classes from OmegaConf/dict configs."""
    @classmethod
    def from_omegaconf(cls, cfg: Any = None, **kwargs: Any) -> Any:
        return instantiate(cls, cfg=cfg, **kwargs)


class ConfigError(FatalError):
    """Configuration is missing, invalid, or points to an unknown component."""


def register_config_store() -> None:
    global _CONFIG_STORE_REGISTERED
    if _CONFIG_STORE_REGISTERED:
        return

    # Keep these imports local: several config dataclasses depend on this
    # module's import helpers, so importing them at module load would cycle.
    from searcherkit.agent import SearchAgentConfig
    from searcherkit.common.dataloader import DataConfig
    from searcherkit.common.retry import RetryConfig
    from searcherkit.llm.base import ClientConfig
    from searcherkit.llm.parsers import ParserConfig
    from searcherkit.runtime.runner import RunConfig
    from searcherkit.sources import SourceConfig
    from searcherkit.tools.base import SummarizerConfig, ToolConfig

    cs = ConfigStore.instance()
    cs.store(name="config", node=RunConfig)
    cs.store(group="agent", name="SearchAgent", node=SearchAgentConfig)
    cs.store(group="llm", name="OpenAIClient", node=ClientConfig)
    cs.store(group="llm", name="AnthropicClient", node=ClientConfig)
    cs.store(group="llm", name="VllmClient", node=ClientConfig)
    cs.store(group="llm", name="TongyiDeepResearchParser", node=ParserConfig)
    cs.store(group="llm", name="UpstreamParser", node=ParserConfig)
    cs.store(group="common", name="RetryConfig", node=RetryConfig)
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
    cfg = compose_config(config_path=config_path, overrides=overrides, default_config_dir=default_config_dir)
    try:
        structured = OmegaConf.structured(config_type)
        merged = OmegaConf.merge(structured, cfg)
        obj = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        raise ConfigError(f"invalid {config_type.__name__} config: {exc}") from exc
    if not isinstance(obj, config_type):
        raise ConfigError(f"resolved config is not a {config_type.__name__}: {config_path or 'config.yaml'}")
    return obj
