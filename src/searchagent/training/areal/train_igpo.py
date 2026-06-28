from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _print_fallback_help(missing_name: str) -> None:
    print(
        "Full AReaL IGPO CLI dependencies are not importable in this environment "
        f"({missing_name}). Showing SearchAgent-specific fallback help.\n",
        file=sys.stderr,
    )
    print(
        "usage: python -m searchagent.training.areal.train_igpo --config CONFIG "
        "[overrides ...]\n\n"
        "SearchAgent AReaL-IGPO training entry. Full AReaL dependencies are "
        "required to parse and run the complete training CLI.\n\n"
        "options:\n"
        "  -h, --help       show this help message and exit\n"
        "  --config CONFIG  AReaL YAML config path.\n",
    )


def _build_config_type(grpo_config_type, workflow_config_type):
    namespace = {
        "__module__": __name__,
        "__annotations__": {
            "workflow": workflow_config_type,
            "eval_workflow": workflow_config_type,
            "dynamic_filter_fn": str | None,
        },
        "workflow": field(default_factory=workflow_config_type),
        "eval_workflow": field(default_factory=workflow_config_type),
        "dynamic_filter_fn": field(
            default="searchagent.training.rewards.should_accept"
        ),
    }
    config_type = type(
        "SearchAgentARealIGPOTrainingConfig",
        (grpo_config_type,),
        namespace,
    )
    return dataclass(config_type)


def _install_step_level_clipping() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    patch_dir = repo_root / "step_level_clipping"
    if patch_dir.is_dir():
        sys.path.insert(0, str(patch_dir))

    try:
        import patch_areal_functional
    except ModuleNotFoundError as exc:
        if exc.name == "patch_areal_functional":
            raise RuntimeError(
                "AREAL_ENABLE_STEP_LEVEL_CLIPPING=1 but "
                "step_level_clipping/patch_areal_functional.py was not found."
            ) from exc
        raise

    patch_areal_functional.install()


def _load_dependencies(argv: list[str]):
    try:
        from datasets import load_dataset

        from areal.api.cli_args import GRPOConfig, load_expr_config

        from searchagent.training.areal.igpo import IGPOTrainer
        from searchagent.training.config import WorkFlowConfig
    except ModuleNotFoundError as exc:
        if any(arg in {"-h", "--help"} for arg in argv):
            _print_fallback_help(exc.name)
            raise SystemExit(0) from exc
        raise RuntimeError(
            "AReaL IGPO training dependencies are not installed or not on "
            "PYTHONPATH. Install AReaL and datasets before running AReaL-IGPO "
            "training."
        ) from exc

    if os.environ.get("AREAL_ENABLE_STEP_LEVEL_CLIPPING") == "1":
        _install_step_level_clipping()

    return (
        load_dataset,
        IGPOTrainer,
        load_expr_config,
        _build_config_type(GRPOConfig, WorkFlowConfig),
    )


def main(args: list[str] | None = None) -> None:
    argv = sys.argv[1:] if args is None else args
    load_dataset, IGPOTrainer, load_expr_config, config_type = _load_dependencies(argv)
    config, _ = load_expr_config(argv, config_type)

    train_dataset = load_dataset(
        "json",
        data_files=config.train_dataset.path,
        split="train",
    )
    valid_dataset = load_dataset(
        "json",
        data_files=config.valid_dataset.path,
        split="train",
    )

    with IGPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="searchagent.training.areal.workflow_igpo.ARealIGPOSearchAgentWorkflow",
            eval_workflow="searchagent.training.areal.workflow_igpo.ARealIGPOSearchAgentWorkflow",
            workflow_kwargs={"config": config.workflow},
            eval_workflow_kwargs={"config": config.eval_workflow},
            dynamic_filter_fn=config.dynamic_filter_fn,
        )


if __name__ == "__main__":
    main()
