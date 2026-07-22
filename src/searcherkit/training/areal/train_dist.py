from __future__ import annotations

import sys
from datasets import load_dataset

from areal.api.cli_args import load_expr_config

from searcherkit.training.areal.config import SearchAgentTrainingConfig
from searcherkit.training.areal.trainer import SearchAgentPPOTrainer



def main(args: list[str] | None = None) -> None:
    argv = sys.argv[1:] if args is None else args
    config, _ = load_expr_config(argv, SearchAgentTrainingConfig)

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

    with SearchAgentPPOTrainer(
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
