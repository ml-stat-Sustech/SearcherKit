"""OmegaConf-friendly instantiation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable, Optional, TypeVar

from omegaconf import DictConfig, OmegaConf

from webagent.log import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


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


def _import_from_path(path: str) -> Callable[..., Any]:
    """Import a dotted path like 'pkg.mod.Class' and return the attribute."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"target must be an import path like 'pkg.mod.Class', got {path!r}")
    module_name, _, attr = path.rpartition(".")
    module = import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"target '{path}' not found") from exc


def _resolve_imports(value: Any, *, target_key: str) -> Any:
    """Recursively resolve importable dotted-path strings (excluding target key)."""
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
    if isinstance(value, str) and "." in value:
        try:
            return _import_from_path(value)
        except (ModuleNotFoundError, ImportError, AttributeError):
            logger.warning("Config import failed for '%s'; keeping raw string", value)
            return value
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
    """Instantiate an object from a factory and/or OmegaConf config.

    Args:
        factory: Callable/class to instantiate. If None, config must provide `target`.
        cfg: DictConfig/dict with params and optional `target` import path.
        target_key: Key name for the import path in cfg.
        recursive: If True, recursively instantiate nested configs containing `target`.
        resolve_imports: If True, attempt to import dotted-path strings in config.
        **kwargs: Overrides for cfg values.
    """

    data = _to_container(cfg)
    if resolve_imports:
        data = _resolve_imports(data, target_key=target_key)

    if isinstance(data, list):
        raise TypeError("cfg must be a mapping or DictConfig when instantiating a target")

    target = data.pop(target_key, None)
    if factory is None:
        if target is None:
            raise ValueError("factory is None and cfg has no target")
        factory = _import_from_path(target)

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
