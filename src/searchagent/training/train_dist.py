from __future__ import annotations

import sys

from datasets import load_dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config

from searchagent.training.config import SearchAgentTrainingConfig


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SearchAgentTrainingConfig)

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
            workflow="searchagent.training.workflow.ARealSearchAgentWorkflow",
            eval_workflow="searchagent.training.workflow.ARealSearchAgentWorkflow",
            workflow_kwargs={"config": config.workflow},
            eval_workflow_kwargs={"config": config.eval_workflow},
            dynamic_filter_fn=config.dynamic_filter_fn,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
