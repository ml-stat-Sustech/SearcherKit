"""Factories for named data source adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from searchagent.common.config import instantiate

from .base import DataSource


@dataclass
class SourceConfig:
    """Configuration for one named data source.

    Source-specific constructor arguments live in ``params`` so Hydra can keep
    this schema stable while concrete adapters define their own fields.
    """

    name: str = ""
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _as_source_config_dict(config: SourceConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, SourceConfig):
        data: dict[str, Any] = {"name": config.name}
        if config.target is not None:
            data["target"] = config.target
        data.update(config.params)
        return data
    data = dict(config)
    params = data.pop("params", None)
    if isinstance(params, Mapping):
        data.update(params)
    elif params is not None:
        raise TypeError("source params must be a mapping when provided")
    return data


def _require_source_contract(source: Any, *, name: str) -> DataSource:
    search = getattr(source, "search", None)
    fetch = getattr(source, "fetch", None)
    if not callable(search) or not callable(fetch):
        raise TypeError(
            f"source {name!r} must provide callable async search() and fetch() methods"
        )
    return source


def build_source(config: SourceConfig | Mapping[str, Any]) -> DataSource:
    """Instantiate one data source from a config mapping."""

    data = _as_source_config_dict(config)
    name = str(data.pop("name", "") or data.get("target", "<unnamed>"))
    source = instantiate(
        cfg=data,
        recursive=True,
        resolve_imports=True,
    )
    return _require_source_contract(source, name=name)


def build_sources(
    configs: Iterable[SourceConfig | Mapping[str, Any]] | Mapping[str, DataSource] | None,
) -> dict[str, DataSource]:
    """Build a name -> source mapping used by source-backed tools."""

    if configs is None:
        return {}
    if isinstance(configs, Mapping):
        return {
            str(name): _require_source_contract(source, name=str(name))
            for name, source in configs.items()
        }

    sources: dict[str, DataSource] = {}
    for config in configs:
        data = _as_source_config_dict(config)
        name = str(data.get("name", ""))
        if not name:
            raise ValueError("each configured source must include a non-empty name")
        if name in sources:
            raise ValueError(f"duplicate source name: {name!r}")
        sources[name] = build_source(data)
    return sources
