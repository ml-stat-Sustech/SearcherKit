"""Configuration for plugin data conversion commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from searchagent.errors import ConfigError


@dataclass(slots=True)
class OpenSeekerMSSwiftConversionConfig:
    input_path: str
    output_path: str
    max_records: int = 0


def _from_dict_config(cfg: DictConfig) -> OpenSeekerMSSwiftConversionConfig:
    allowed = {
        "input_path",
        "output_path",
        "max_records",
    }
    keys = set(cfg.keys())
    unexpected = sorted(keys - allowed)
    if unexpected:
        raise ConfigError(f"unexpected OpenSeekerMSSwiftConversionConfig fields: {', '.join(unexpected)}")

    missing = sorted(key for key in ("input_path", "output_path") if key not in keys)
    if missing:
        raise ConfigError(f"missing SFT conversion config field: {missing[0]}")

    return OpenSeekerMSSwiftConversionConfig(
        input_path=str(cfg.input_path),
        output_path=str(cfg.output_path),
        max_records=int(cfg.get("max_records", 0)),
    )


def load_config(
    *,
    config_path: str | Path | None = None,
    config_name: str = "openseeker_ms_swift",
    overrides: Sequence[str] | None = None,
) -> OpenSeekerMSSwiftConversionConfig:
    config_dir = _resolve_config_dir(config_path)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"config path does not exist or is not a directory: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=_clean_overrides(overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ConfigError(f"SFT conversion config must be a mapping: {config_name}")
    return _from_dict_config(OmegaConf.create(resolved))


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "plugins" / "conversion"


def _resolve_config_dir(config_path: str | Path | None) -> Path:
    if config_path is None:
        return _default_config_dir()
    return Path(config_path).expanduser().resolve()


def _clean_overrides(overrides: Sequence[str] | None) -> list[str]:
    if not overrides:
        return []
    cleaned = list(overrides)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned
