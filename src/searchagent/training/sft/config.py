"""Configuration for agentic SFT training launch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from searchagent.errors import ConfigError


@dataclass
class SFTTrainingDataConfig:
    train_path: str = "outputs/sft/ms_swift.jsonl"


@dataclass
class SFTBackendConfig:
    swift_bin: str = "swift"
    command: str = "sft"
    model: str = "/home/jovyan/Qwen3-8B"
    template: str = "qwen3"
    agent_template: str = "hermes"
    loss_scale: str = "hermes"
    output_dir: str = "outputs/sft/model"
    tuner_type: str | None = "full"
    train_type: str | None = None
    torch_dtype: str | None = "bfloat16"
    rope_scaling: dict[str, Any] | str | None = None
    max_model_len: int | None = 262144
    max_length: int | None = 262144
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    num_train_epochs: float = 1
    save_steps: int = 100
    eval_steps: int | None = None
    deepspeed: str | None = "zero3_offload"
    dataloader_num_workers: int = 16
    gradient_checkpointing: bool = True
    dataset_num_proc: int = 16
    load_from_cache_file: bool = True
    packing: bool = False
    sequence_parallel_size: int | None = 8
    attn_impl: str | None = "flash_attn"
    save_total_limit: int | None = 2
    logging_steps: int = 1
    report_to: str = "swanlab"
    split_dataset_ratio: float | None = 0.0
    save_only_model: bool | None = None
    use_liger_kernel: bool | None = None
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTTrainingConfig:
    data: SFTTrainingDataConfig = field(default_factory=SFTTrainingDataConfig)
    backend: SFTBackendConfig = field(default_factory=SFTBackendConfig)


def _dataclass_from_mapping(cls: type[Any], value: Mapping[str, Any] | None) -> Any:
    if value is None:
        return cls()
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    allowed = {field.name for field in fields(cls)}
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ConfigError(f"unexpected {cls.__name__} fields: {', '.join(unexpected)}")
    return cls(**dict(value))


def _training_data_from_mapping(value: Mapping[str, Any] | None) -> SFTTrainingDataConfig:
    if value is None:
        return SFTTrainingDataConfig()
    train_path = value.get("train_path", SFTTrainingDataConfig.train_path)
    return SFTTrainingDataConfig(train_path=str(train_path))


def load_training_config(path: str | Path) -> SFTTrainingConfig:
    config_path = Path(path).expanduser()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"SFT training config must be a mapping: {config_path}")
    allowed = {"data", "agent", "backend"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ConfigError(f"unexpected SFTTrainingConfig fields: {', '.join(unexpected)}")
    return SFTTrainingConfig(
        data=_training_data_from_mapping(raw.get("data")),
        backend=_dataclass_from_mapping(SFTBackendConfig, raw.get("backend")),
    )
