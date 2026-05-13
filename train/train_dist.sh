export SearchAgent_LOG_LEVEL=WARN
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python3 train_dist.py \
    --config train_dist.yaml \
    scheduler.type=ray \
    cluster.n_nodes=2 \
    actor.backend=fsdp:d4c2 \
    actor.weight_update_mode=xccl \
    actor.optimizer.lr=5e-6 \
    actor.optimizer.lr_scheduler_type=constant \
    actor.optimizer.warmup_steps_proportion=0 \
    actor.path=/home/jovyan1/Qwen3-8B \
    rollout.backend=sglang:d8 \
    rollout.max_concurrent_rollouts=128 \
    gconfig.n_samples=8 \
    trial_name=qwen3_emb_8b_$(date +%Y%m%d_%H%M%S) \
    train_dataset.batch_size=128 \
    train_dataset.path=/home/jovyan1/Infoseek_ASearcher_filtered_merged.jsonl
    # actor.path=/home/jovyan/agentic_sft/output/Qwen3-8B_sft_Openseeker_qwen3_yarn_262k/v4-20260327-045741/checkpoint-1459 
    # train_dataset.path=/home/jovyan1/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean/ASearcher_en_no-math_Qwen3-8B-reject-sample-clean.jsonl

# qwen3_train_$(date +%Y%m%d_%H%M%S)
