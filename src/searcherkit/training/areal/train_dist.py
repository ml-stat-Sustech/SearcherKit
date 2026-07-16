from __future__ import annotations

import sys
from dataclasses import dataclass, field


def _print_fallback_help(missing_name: str) -> None:
    print(
        "Full AReaL CLI dependencies are not importable in this environment "
        f"({missing_name}). Showing SearcherKit-specific fallback help.\n",
        file=sys.stderr,
    )
    print(
        "usage: python -m searcherkit.training.areal.train_dist --config CONFIG "
        "[overrides ...]\n\n"
        "SearcherKit AReaL training entry. Full AReaL dependencies are required "
        "to parse and run the complete training CLI.\n\n"
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
            default="searcherkit.training.rewards.should_accept"
        ),
    }
    config_type = type(
        "SearchAgentARealTrainingConfig",
        (grpo_config_type,),
        namespace,
    )
    return dataclass(config_type)


def _load_dependencies(argv: list[str]):
    try:
        from datasets import load_dataset

        from areal import PPOTrainer
        from areal.api.cli_args import GRPOConfig, load_expr_config

        from searcherkit.training.config import WorkFlowConfig
    except ModuleNotFoundError as exc:
        if any(arg in {"-h", "--help"} for arg in argv):
            _print_fallback_help(exc.name)
            raise SystemExit(0) from exc
        raise RuntimeError(
            "AReaL training dependencies are not installed or not on PYTHONPATH. "
            "Install AReaL and datasets before running AReaL training."
        ) from exc
    return load_dataset, PPOTrainer, load_expr_config, _build_config_type(GRPOConfig, WorkFlowConfig)


def main(args: list[str] | None = None) -> None:
    argv = sys.argv[1:] if args is None else args
    load_dataset, PPOTrainer, load_expr_config, config_type = _load_dependencies(argv)
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

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="searcherkit.training.areal.workflow.ARealSearchAgentWorkflow",
            eval_workflow="searcherkit.training.areal.workflow.ARealSearchAgentWorkflow",
            workflow_kwargs={"config": config.workflow},
            eval_workflow_kwargs={"config": config.eval_workflow},
            dynamic_filter_fn=config.dynamic_filter_fn,
        )


if __name__ == "__main__":
    main()
