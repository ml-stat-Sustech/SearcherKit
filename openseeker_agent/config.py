"""Configuration structures for SFT runtime/evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from searchagent.common.errors import ConfigError


@dataclass
class OpenSeekerBCPGenerateConfig:
    dataset_path: Path
    out_dir: Path
    normalized_dataset: Path | None = None
    max_tokens: int = 32768
    tool_count_max: int = 200
    max_worker: int = 60
    pool_no_progress_timeout: int = 18000
    pool_restart_rounds: int = 2
    max_retry_rounds: int = 10
    limit: int = -1
    print_stream: bool = False
    sequential: bool = False
    run_log_path: Path | None = None
    no_run_log: bool = False


@dataclass
class OpenSeekerBCPJudgeConfig:
    data_path: Path | None = None
    max_workers: int = 20
    limit: int = -1
    top_k: int = -1
    save_path: Path | None = None
    run_after_generate: bool = False


@dataclass
class OpenSeekerBCPRunConfig:
    generate: OpenSeekerBCPGenerateConfig
    judge: OpenSeekerBCPJudgeConfig


def _optional_path(value: Any) -> Path | None:
    if value in (None, "null", ""):
        return None
    return Path(str(value))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_openseeker_bcp_run_config(path: str | Path) -> OpenSeekerBCPRunConfig:
    config_path = Path(path).expanduser()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ConfigError(f"OpenSeeker BCP run config must be a mapping: {config_path}")
    allowed = {"generate", "judge"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ConfigError(f"unexpected OpenSeekerBCPRunConfig fields: {', '.join(unexpected)}")
    generate_raw = raw.get("generate") or {}
    judge_raw = raw.get("judge") or {}
    if not isinstance(generate_raw, dict):
        raise ConfigError("generate config must be a mapping")
    if not isinstance(judge_raw, dict):
        raise ConfigError("judge config must be a mapping")

    generate = OpenSeekerBCPGenerateConfig(
        dataset_path=Path(str(generate_raw.get("dataset_path") or "")),
        out_dir=Path(str(generate_raw.get("out_dir") or "")),
        normalized_dataset=_optional_path(generate_raw.get("normalized_dataset")),
        max_tokens=int(generate_raw.get("max_tokens", 32768)),
        tool_count_max=int(generate_raw.get("tool_count_max", 200)),
        max_worker=int(generate_raw.get("max_worker", 60)),
        pool_no_progress_timeout=int(generate_raw.get("pool_no_progress_timeout", 18000)),
        pool_restart_rounds=int(generate_raw.get("pool_restart_rounds", 2)),
        max_retry_rounds=int(generate_raw.get("max_retry_rounds", 10)),
        limit=int(generate_raw.get("limit", -1)),
        print_stream=_bool_value(generate_raw.get("print_stream")),
        sequential=_bool_value(generate_raw.get("sequential")),
        run_log_path=_optional_path(generate_raw.get("run_log_path")),
        no_run_log=_bool_value(generate_raw.get("no_run_log")),
    )
    judge = OpenSeekerBCPJudgeConfig(
        data_path=_optional_path(judge_raw.get("data_path")),
        max_workers=int(judge_raw.get("max_workers", 20)),
        limit=int(judge_raw.get("limit", -1)),
        top_k=int(judge_raw.get("top_k", -1)),
        save_path=_optional_path(judge_raw.get("save_path")),
        run_after_generate=_bool_value(judge_raw.get("run_after_generate")),
    )
    return OpenSeekerBCPRunConfig(generate=generate, judge=judge)
