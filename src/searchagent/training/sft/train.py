"""Config-driven ms-swift training launcher for SearchAgent agentic SFT."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from searchagent.errors import ConfigError
from searchagent.training.sft.config import SFTTrainingConfig, load_training_config


def _json_arg(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _append_arg(argv: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    key = f"--{name}"
    if isinstance(value, bool):
        argv.extend([key, str(value).lower()])
        return
    if isinstance(value, Mapping):
        argv.extend([key, _json_arg(value)])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        argv.append(key)
        argv.extend(str(item) for item in value)
        return
    argv.extend([key, str(value)])


def build_swift_command(config: SFTTrainingConfig, dataset_path: str | Path) -> list[str]:
    backend = config.backend
    argv = shlex.split(backend.swift_bin)
    if backend.command:
        argv.append(backend.command)

    base_args: dict[str, Any] = {
        "model": backend.model,
        "template": backend.template,
        "agent_template": backend.agent_template,
        "loss_scale": backend.loss_scale,
        "dataset": str(Path(dataset_path).expanduser()),
        "output_dir": backend.output_dir,
        "tuner_type": backend.tuner_type,
        "train_type": backend.train_type,
        "torch_dtype": backend.torch_dtype,
        "rope_scaling": backend.rope_scaling,
        "max_model_len": backend.max_model_len,
        "max_length": backend.max_length,
        "per_device_train_batch_size": backend.per_device_train_batch_size,
        "per_device_eval_batch_size": backend.per_device_eval_batch_size,
        "gradient_accumulation_steps": backend.gradient_accumulation_steps,
        "learning_rate": backend.learning_rate,
        "num_train_epochs": backend.num_train_epochs,
        "save_steps": backend.save_steps,
        "eval_steps": backend.eval_steps,
        "deepspeed": backend.deepspeed,
        "dataloader_num_workers": backend.dataloader_num_workers,
        "gradient_checkpointing": backend.gradient_checkpointing,
        "dataset_num_proc": backend.dataset_num_proc,
        "load_from_cache_file": backend.load_from_cache_file,
        "packing": backend.packing,
        "sequence_parallel_size": backend.sequence_parallel_size,
        "attn_impl": backend.attn_impl,
        "save_total_limit": backend.save_total_limit,
        "logging_steps": backend.logging_steps,
        "report_to": backend.report_to,
        "split_dataset_ratio": backend.split_dataset_ratio,
        "save_only_model": backend.save_only_model,
        "use_liger_kernel": backend.use_liger_kernel,
    }
    for key, value in base_args.items():
        _append_arg(argv, key, value)
    for key, value in backend.extra_args.items():
        _append_arg(argv, key, value)
    return argv


def run_training(config: SFTTrainingConfig, *, dry_run: bool) -> int:
    command = build_swift_command(config, config.data.train_path)
    print("resolved swift command:")
    print(shlex.join(command))
    if dry_run:
        return 0

    env = os.environ.copy()
    env.update({key: str(value) for key, value in config.backend.env.items()})
    try:
        completed = subprocess.run(command, env=env, check=False)
    except FileNotFoundError as exc:
        raise ConfigError(f"swift executable not found: {config.backend.swift_bin!r}") from exc
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SearchAgent agentic SFT data with ms-swift.")
    parser.add_argument("--config", default="scripts/sft/sft_training.yaml", help="Path to SFT training YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print swift command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config)
    return run_training(config, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
