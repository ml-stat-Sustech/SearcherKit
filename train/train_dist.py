import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from igpo.igpo_trainer import IGPOTrainer
from areal.api.cli_args import load_expr_config

from config_type import SearchAgentTrainingConfig

def main(args):
    config, _ = load_expr_config(args, SearchAgentTrainingConfig)

    train_dataset = load_dataset('json', 
                                 data_files=config.train_dataset.path,
                                 split="train")
    valid_dataset = load_dataset('json', 
                                 data_files=config.valid_dataset.path,
                                 split="train")
    
    workflow_config = config.workflow
    eval_workflow_config = config.eval_workflow

    with IGPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset
    ) as trainer:
        trainer.train(
            workflow="rollout_webagent.ARealSearchAgentWorkflow",
            eval_workflow="rollout_webagent.ARealSearchAgentWorkflow",
            workflow_kwargs={
                "config": workflow_config
            },
            eval_workflow_kwargs={
                "config": eval_workflow_config
            },
            dynamic_filter_fn=config.dynamic_filter_fn  # DAPO filtering
        )

if __name__ == "__main__":
    main(sys.argv[1:])