from __future__ import annotations

import copy
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import torch


_ANSWER_BEGIN_TOKENS = 3
_ANSWER_END_TOKENS = 2


@dataclass(frozen=True)
class ActorIGPOScoreRequest:
    sample_index: int
    reward_index: int
    response_pos: int
    token_ids: list[int]
    answer_mask: list[bool]
    score_response_length: int


def _answer_tokens(tokenizer: Any, answer: str) -> tuple[list[int], int, int]:
    content = f"\\boxed{{{answer}}}"
    encoded = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": content}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, dict):
        token_ids = list(encoded["input_ids"])
    else:
        token_ids = list(encoded)
    answer_start = min(_ANSWER_BEGIN_TOKENS, len(token_ids))
    answer_end = max(answer_start, len(token_ids) - _ANSWER_END_TOKENS)
    return token_ids, answer_start, answer_end


def _metadata_spans(metadata: Any) -> list[tuple[int, int]]:
    if not isinstance(metadata, dict):
        return []
    spans = metadata.get("ig_response_spans")
    if spans is None:
        spans = metadata.get("train_model_output_spans")
    if spans is None:
        spans = metadata.get("model_output_spans")
    if not isinstance(spans, list):
        return []
    out = []
    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if end > start:
            out.append((start, end))
    return out


def build_actor_igpo_score_requests(
    *,
    tokenizer: Any,
    tokens: list[int],
    response_length: int,
    metadata: Any,
    sample_index: int = 0,
) -> list[ActorIGPOScoreRequest]:
    if response_length <= 0:
        return []
    if not isinstance(metadata, dict):
        return []
    ground_truth = metadata.get("ground_truth")
    if ground_truth is None or str(ground_truth) == "":
        return []

    spans = _metadata_spans(metadata)
    if not spans:
        return []
    spans = [(max(0, start), min(response_length, end)) for start, end in spans if start < response_length]
    spans = [(start, end) for start, end in spans if end > start]
    if not spans:
        return []

    prompt_length = len(tokens) - response_length
    if prompt_length < 0:
        return []

    answer_ids, answer_start, answer_end = _answer_tokens(tokenizer, str(ground_truth))
    if answer_end <= answer_start:
        return []
    requests: list[ActorIGPOScoreRequest] = []

    # Baseline p(y | p_0), before observing any assistant action in this segment.
    prefix_cut_positions = [0]
    response_positions = [-1]
    for idx in range(len(spans) - 1):
        # AReal scores the next prompt state p_{k+1}; for SearchAgent merged
        # segments, the next output span starts after tool/context tokens have
        # been stitched into the response segment.
        prefix_cut_positions.append(spans[idx + 1][0])
        response_positions.append(spans[idx][1] - 1)

    for reward_index, (prefix_response_pos, response_pos) in enumerate(
        zip(prefix_cut_positions, response_positions, strict=True)
    ):
        prefix_len = prompt_length + prefix_response_pos
        prefix = list(tokens[:prefix_len])
        score_answer_ids = answer_ids[:answer_end]
        answer_mask = [False] * len(score_answer_ids)
        for pos in range(answer_start, answer_end):
            if 0 <= pos < len(answer_mask):
                answer_mask[pos] = True
        requests.append(
            ActorIGPOScoreRequest(
                sample_index=sample_index,
                reward_index=reward_index - 1,
                response_pos=response_pos,
                token_ids=prefix + score_answer_ids,
                answer_mask=answer_mask,
                score_response_length=max(0, answer_end - answer_start),
            )
        )
    return requests


def scatter_actor_igpo_rewards(
    *,
    response_length: int,
    response_positions: list[int],
    answer_logprobs: list[float],
) -> tuple[list[float], list[int], list[float]]:
    raw_rewards = [
        float(answer_logprobs[idx + 1] - answer_logprobs[idx])
        for idx in range(max(0, len(answer_logprobs) - 1))
    ]
    token_rewards = [0.0] * response_length
    reward_mask = [0] * response_length
    for reward_idx, pos in enumerate(response_positions):
        if reward_idx >= len(raw_rewards):
            break
        if 0 <= pos < response_length:
            token_rewards[pos] = raw_rewards[reward_idx]
            reward_mask[pos] = 1
    return token_rewards, reward_mask, raw_rewards


def actor_igpo_reward_metrics(
    *,
    token_rewards: list[float],
    reward_mask: list[int],
    expected_reward_count: int | None = None,
    score_request_count: int | None = None,
) -> dict[str, float]:
    if len(token_rewards) != len(reward_mask):
        raise ValueError(
            f"IGPO metric length mismatch: got {len(token_rewards)} rewards and {len(reward_mask)} mask values"
        )

    masked_rewards = [
        abs(float(reward))
        for reward, keep in zip(token_rewards, reward_mask, strict=True)
        if int(keep) != 0
    ]
    mask_count = float(len(masked_rewards))
    if not masked_rewards:
        metrics = {
            "actor_ig_reward_mask_count": 0.0,
            "actor_ig_reward_abs_mean": 0.0,
            "actor_ig_reward_nonzero_ratio": 0.0,
        }
        if expected_reward_count is not None:
            metrics["actor_ig_expected_reward_count"] = float(expected_reward_count)
            metrics["actor_ig_reward_mask_coverage_ratio"] = 0.0
        if score_request_count is not None:
            metrics["actor_ig_score_request_count"] = float(score_request_count)
        return metrics

    nonzero_count = sum(1 for reward in masked_rewards if reward > 1e-12)
    metrics = {
        "actor_ig_reward_mask_count": mask_count,
        "actor_ig_reward_abs_mean": float(sum(masked_rewards) / len(masked_rewards)),
        "actor_ig_reward_nonzero_ratio": float(nonzero_count / len(masked_rewards)),
    }
    if expected_reward_count is not None:
        metrics["actor_ig_expected_reward_count"] = float(expected_reward_count)
        metrics["actor_ig_reward_mask_coverage_ratio"] = (
            mask_count / float(expected_reward_count)
            if expected_reward_count > 0
            else 0.0
        )
    if score_request_count is not None:
        metrics["actor_ig_score_request_count"] = float(score_request_count)
    return metrics


def _context_parallel_world_size() -> int:
    from megatron.core import mpu

    return int(mpu.get_context_parallel_world_size())


def _slice_log_prob_with_cp(
    values: list[float] | list[int],
    total_length: int,
    response_length: int,
    qkv_format: str,
    max_seq_len: int | None,
) -> list[float] | list[int]:
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    return slice_log_prob_with_cp(values, total_length, response_length, qkv_format, max_seq_len)


def prepare_actor_igpo_reward_tensor(
    *,
    values: list[float] | list[int],
    total_length: int,
    response_length: int,
    qkv_format: str,
    max_seq_len: int | None,
    device: torch.device | str | int,
) -> torch.Tensor:
    if len(values) != response_length:
        raise ValueError(
            f"IGPO reward length mismatch: got {len(values)}, expected response_length={response_length}"
        )
    local_values = values
    if _context_parallel_world_size() > 1:
        local_values = _slice_log_prob_with_cp(
            values,
            total_length,
            response_length,
            qkv_format,
            max_seq_len,
        )
    return torch.tensor(local_values, dtype=torch.float32, device=device)


def _tensor_to_float_mean(log_probs: torch.Tensor) -> float:
    if log_probs.numel() == 0:
        return 0.0
    return float(log_probs.float().mean().item())


def get_igpo_answer_logprobs(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    with_entropy: bool = False,
) -> dict[str, list[torch.Tensor]]:
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    _, output = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=max_seq_lens,
    )
    values = []
    for idx, log_prob in enumerate(output["log_probs"]):
        if mpu.get_context_parallel_world_size() > 1 and not getattr(args, "allgather_cp", False):
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            log_prob = all_gather_with_cp(
                log_prob,
                total_lengths[idx],
                response_lengths[idx],
                getattr(args, "qkv_format", "thd"),
                max_seq_len,
            )
        values.append(torch.tensor(_tensor_to_float_mean(log_prob), device=logits.device))
    return torch.empty((0,), device=logits.device), {"igpo_answer_logprobs": values}


def _build_score_rollout_data(
    *,
    args: Namespace,
    tokenizer: Any,
    rollout_data: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[ActorIGPOScoreRequest]]:
    requests: list[ActorIGPOScoreRequest] = []
    tokens = rollout_data["tokens"]
    response_lengths = rollout_data["response_lengths"]
    metadata = rollout_data.get("metadata") or [None] * len(tokens)
    for sample_index, (sample_tokens, response_length, sample_metadata) in enumerate(
        zip(tokens, response_lengths, metadata, strict=False)
    ):
        token_ids = sample_tokens.detach().cpu().tolist() if isinstance(sample_tokens, torch.Tensor) else list(sample_tokens)
        requests.extend(
            build_actor_igpo_score_requests(
                tokenizer=tokenizer,
                tokens=token_ids,
                response_length=int(response_length),
                metadata=sample_metadata,
                sample_index=sample_index,
            )
        )

    if not requests:
        return None, requests

    score_tokens = [
        torch.tensor(request.token_ids, dtype=torch.long, device=torch.cuda.current_device())
        for request in requests
    ]
    loss_masks = [
        torch.ones(request.score_response_length, dtype=torch.float32, device=torch.cuda.current_device())
        for request in requests
    ]
    score_data: dict[str, Any] = {
        "tokens": score_tokens,
        "response_lengths": [request.score_response_length for request in requests],
        "total_lengths": [len(request.token_ids) for request in requests],
        "loss_masks": loss_masks,
        "multimodal_train_inputs": [None] * len(requests),
    }

    if getattr(args, "qkv_format", "thd") == "bshd":
        from megatron.core import mpu

        pad_size = mpu.get_tensor_model_parallel_world_size() * int(getattr(args, "data_pad_size_multiplier", 128))
        max_seq_len = max(score_data["total_lengths"])
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
        score_data["max_seq_lens"] = [max_seq_len] * len(requests)

    micro_batch_size = max(1, int(getattr(args, "searchagent_igpo_actor_score_micro_batch_size", 8)))
    score_data["micro_batch_indices"] = [
        list(range(start, min(start + micro_batch_size, len(requests))))
        for start in range(0, len(requests), micro_batch_size)
    ]
    microbatch_group_size = max(1, int(getattr(args, "searchagent_igpo_actor_score_microbatch_group_size", 1)))
    if microbatch_group_size > 1:
        pad_count = (-len(score_data["micro_batch_indices"])) % microbatch_group_size
        if pad_count and requests:
            last_index = len(requests) - 1
            score_data["micro_batch_indices"].extend([[last_index] for _ in range(pad_count)])
    score_data["num_microbatches"] = [len(score_data["micro_batch_indices"])]
    score_data["global_batch_sizes"] = [len(requests)]
    return score_data, requests


def attach_actor_side_igpo_rewards(
    *,
    args: Namespace,
    model: Any,
    tokenizer: Any,
    rollout_data: dict[str, Any],
) -> None:
    score_data, requests = _build_score_rollout_data(args=args, tokenizer=tokenizer, rollout_data=rollout_data)
    if score_data is None:
        if any(int(length) > 0 for length in rollout_data["response_lengths"]):
            raise ValueError(
                "Actor-side IGPO could not build any gold-answer score requests. "
                "Ensure rollout metadata contains ground_truth and ig_response_spans."
            )
        _attach_empty_rewards(args, rollout_data)
        return

    from megatron.core import mpu
    from megatron.core.utils import get_model_config
    from slime.backends.megatron_utils.data import DataIterator
    from slime.backends.megatron_utils.model import forward_only

    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    scoring_args = copy.copy(args)
    scoring_args.use_dynamic_batch_size = False
    scoring_args.searchagent_igpo_actor_score_microbatch_group_size = 1
    if vpp_size > 1:
        scoring_args.searchagent_igpo_actor_score_microbatch_group_size = int(
            get_model_config(model[0]).microbatch_group_size_per_vp_stage
        )
        score_data, requests = _build_score_rollout_data(
            args=scoring_args,
            tokenizer=tokenizer,
            rollout_data=rollout_data,
        )
        if score_data is None:
            if any(int(length) > 0 for length in rollout_data["response_lengths"]):
                raise ValueError(
                    "Actor-side IGPO could not build any gold-answer score requests. "
                    "Ensure rollout metadata contains ground_truth and ig_response_spans."
                )
            _attach_empty_rewards(args, rollout_data)
            return
    iterator = [DataIterator(score_data, score_data["micro_batch_indices"]) for _ in range(vpp_size)]
    score_output = forward_only(
        get_igpo_answer_logprobs,
        scoring_args,
        model,
        iterator,
        score_data["num_microbatches"],
    )
    answer_logprobs = [
        float(item.detach().cpu().item())
        for item in score_output.get("igpo_answer_logprobs", [])[: len(requests)]
    ]
    if not answer_logprobs:
        from megatron.core import mpu

        if not mpu.is_pipeline_last_stage():
            return
        if len(requests) > 0:
            raise RuntimeError("Actor-side IGPO scoring produced no answer logprobs on the pipeline last stage.")
        _attach_empty_rewards(args, rollout_data)
        return

    grouped_logprobs: list[list[float]] = [[] for _ in rollout_data["tokens"]]
    grouped_positions: list[list[int]] = [[] for _ in rollout_data["tokens"]]
    grouped_request_counts = [0 for _ in rollout_data["tokens"]]
    for request, logprob in zip(requests, answer_logprobs, strict=False):
        grouped_logprobs[request.sample_index].append(logprob)
        grouped_request_counts[request.sample_index] += 1
        if request.reward_index >= 0:
            grouped_positions[request.sample_index].append(request.response_pos)

    ig_token_rewards = []
    ig_reward_mask = []
    ig_turn_rewards = []
    ig_metric_values: dict[str, list[float]] = {
        "actor_ig_reward_mask_count": [],
        "actor_ig_reward_abs_mean": [],
        "actor_ig_reward_nonzero_ratio": [],
        "actor_ig_score_request_count": [],
        "actor_ig_expected_reward_count": [],
        "actor_ig_reward_mask_coverage_ratio": [],
    }
    qkv_format = getattr(args, "qkv_format", "thd")
    max_seq_lens = rollout_data.get("max_seq_lens")
    device = torch.cuda.current_device()
    for idx, (response_length, logprobs, positions, request_count) in enumerate(
        zip(rollout_data["response_lengths"], grouped_logprobs, grouped_positions, grouped_request_counts, strict=False)
    ):
        token_rewards, reward_mask, raw_rewards = scatter_actor_igpo_rewards(
            response_length=int(response_length),
            response_positions=positions,
            answer_logprobs=logprobs,
        )
        for key, value in actor_igpo_reward_metrics(
            token_rewards=token_rewards,
            reward_mask=reward_mask,
            expected_reward_count=len(positions),
            score_request_count=request_count,
        ).items():
            ig_metric_values[key].append(value)
        max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
        ig_token_rewards.append(
            prepare_actor_igpo_reward_tensor(
                values=token_rewards,
                total_length=int(rollout_data["total_lengths"][idx]),
                response_length=int(response_length),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
                device=device,
            )
        )
        ig_reward_mask.append(
            prepare_actor_igpo_reward_tensor(
                values=reward_mask,
                total_length=int(rollout_data["total_lengths"][idx]),
                response_length=int(response_length),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
                device=device,
            )
        )
        ig_turn_rewards.append(raw_rewards)

    rollout_data["ig_token_rewards"] = ig_token_rewards
    rollout_data["ig_reward_mask"] = ig_reward_mask
    rollout_data["ig_turn_rewards"] = ig_turn_rewards
    rollout_data.update(ig_metric_values)


def _attach_empty_rewards(args: Namespace, rollout_data: dict[str, Any]) -> None:
    ig_token_rewards = []
    ig_reward_mask = []
    qkv_format = getattr(args, "qkv_format", "thd")
    max_seq_lens = rollout_data.get("max_seq_lens")
    device = torch.cuda.current_device()
    for idx, response_length in enumerate(rollout_data["response_lengths"]):
        max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
        ig_token_rewards.append(
            prepare_actor_igpo_reward_tensor(
                values=[0.0] * int(response_length),
                total_length=int(rollout_data["total_lengths"][idx]),
                response_length=int(response_length),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
                device=device,
            )
        )
        ig_reward_mask.append(
            prepare_actor_igpo_reward_tensor(
                values=[0] * int(response_length),
                total_length=int(rollout_data["total_lengths"][idx]),
                response_length=int(response_length),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
                device=device,
            )
        )
    rollout_data["ig_token_rewards"] = ig_token_rewards
    rollout_data["ig_reward_mask"] = ig_reward_mask
    rollout_data["ig_turn_rewards"] = [[] for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_reward_mask_count"] = [0.0 for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_reward_abs_mean"] = [0.0 for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_reward_nonzero_ratio"] = [0.0 for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_score_request_count"] = [0.0 for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_expected_reward_count"] = [0.0 for _ in rollout_data["response_lengths"]]
    rollout_data["actor_ig_reward_mask_coverage_ratio"] = [0.0 for _ in rollout_data["response_lengths"]]
