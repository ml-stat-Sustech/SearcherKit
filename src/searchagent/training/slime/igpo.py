from __future__ import annotations

from typing import Any

import torch


def _advantage_estimator(args: Any) -> str:
    value = str(getattr(args, "advantage_estimator", "grpo"))
    if value not in {"grpo", "igpo"}:
        raise ValueError(f"Unsupported SearchAgent advantage estimator: {value}")
    return value


def _context_parallel_size() -> int:
    from megatron.core import mpu

    return int(mpu.get_context_parallel_world_size())


def _full_tensor(
    args: Any,
    local: torch.Tensor,
    total_length: int,
    response_length: int,
    max_seq_len: int | None,
) -> torch.Tensor:
    if _context_parallel_size() > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        return all_gather_with_cp(
            local,
            total_length,
            response_length,
            getattr(args, "qkv_format", "thd"),
            max_seq_len,
        )
    return local


def _slice_tensor(
    args: Any,
    full: torch.Tensor,
    total_length: int,
    response_length: int,
    max_seq_len: int | None,
) -> torch.Tensor:
    if _context_parallel_size() > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        return slice_log_prob_with_cp(
            full,
            total_length,
            response_length,
            getattr(args, "qkv_format", "thd"),
            max_seq_len,
        )
    return full


def _normalize_ig_rewards(
    ig_rewards: list[torch.Tensor],
    reward_masks: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    valid_values = []
    for rewards, reward_mask, loss_mask in zip(ig_rewards, reward_masks, loss_masks, strict=False):
        keep = reward_mask.bool() & loss_mask.bool()
        if keep.any():
            valid_values.append(rewards[keep])
    if not valid_values:
        return ig_rewards

    values = torch.cat(valid_values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if torch.isnan(std) or std <= 0:
        return [torch.zeros_like(rewards) for rewards in ig_rewards]
    return [
        torch.where(
            reward_mask.bool() & loss_mask.bool(),
            (rewards - mean) / (std + 1e-8),
            torch.zeros_like(rewards),
        )
        for rewards, reward_mask, loss_mask in zip(ig_rewards, reward_masks, loss_masks, strict=False)
    ]


def _discounted_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for idx in range(rewards.numel() - 1, -1, -1):
        if mask[idx].bool():
            running = rewards[idx] + gamma * running
            returns[idx] = running
        else:
            returns[idx] = 0
    return returns


def _grpo_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    kl = rollout_data["kl"]
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=kl[0].device)
    returns = [torch.ones_like(item) * rewards[idx] for idx, item in enumerate(kl)]
    rollout_data["advantages"] = [item for item in returns]
    rollout_data["returns"] = returns


def _igpo_advantages(args: Any, rollout_data: dict[str, Any]) -> None:
    if "ig_token_rewards" not in rollout_data:
        raise ValueError(
            "IGPO requires ig_token_rewards in rollout_data. Ensure --advantage-estimator igpo is used for rollout."
        )
    if "ig_reward_mask" not in rollout_data:
        raise ValueError(
            "IGPO requires ig_reward_mask in rollout_data. Ensure --advantage-estimator igpo is used for rollout."
        )

    kl = rollout_data["kl"]
    rewards = rollout_data["rewards"]
    loss_masks = rollout_data["loss_masks"]
    ig_token_rewards = rollout_data["ig_token_rewards"]
    ig_reward_mask = rollout_data["ig_reward_mask"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    max_seq_lens = rollout_data.get("max_seq_lens")
    gamma = float(getattr(args, "gamma", 1.0))
    kl_coef = float(getattr(args, "kl_coef", 0.0))
    ig_reward_coef = float(getattr(args, "searchagent_igpo_reward_coef", 1.0))
    outcome_reward_coef = float(getattr(args, "searchagent_igpo_outcome_reward_coef", 1.0))

    full_ig = []
    full_masks = []
    full_reward_masks = []
    full_kl = []
    for idx, (ig_reward, reward_mask, mask, kl_item) in enumerate(
        zip(ig_token_rewards, ig_reward_mask, loss_masks, kl, strict=False)
    ):
        total_len = total_lengths[idx]
        response_len = response_lengths[idx]
        max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
        full_ig.append(_full_tensor(args, ig_reward.float(), total_len, response_len, max_seq_len))
        full_reward_masks.append(_full_tensor(args, reward_mask.float(), total_len, response_len, max_seq_len))
        full_masks.append(mask.float())
        full_kl.append(_full_tensor(args, kl_item.float(), total_len, response_len, max_seq_len))

    full_ig = _normalize_ig_rewards(full_ig, full_reward_masks, full_masks)

    advantages = []
    returns = []
    for idx, (mask, ig_reward, kl_item) in enumerate(zip(full_masks, full_ig, full_kl, strict=False)):
        token_rewards = ig_reward_coef * ig_reward - kl_coef * kl_item
        valid = mask.bool().nonzero(as_tuple=True)[0]
        if valid.numel() > 0:
            token_rewards[valid[-1]] += outcome_reward_coef * float(rewards[idx])
        full_return = _discounted_returns(token_rewards, mask, gamma)
        total_len = total_lengths[idx]
        response_len = response_lengths[idx]
        max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
        local_return = _slice_tensor(args, full_return, total_len, response_len, max_seq_len).detach()
        returns.append(local_return)
        advantages.append(local_return)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    rollout_data["ig_token_rewards_full"] = full_ig
    rollout_data["ig_reward_mask_full"] = full_reward_masks
    rollout_data["ig_token_rewards_scaled"] = [item * ig_reward_coef for item in full_ig]


def compute_advantages_and_returns(args: Any, rollout_data: dict[str, Any]) -> None:
    if _advantage_estimator(args) == "grpo":
        _grpo_advantages(args, rollout_data)
        return
    _igpo_advantages(args, rollout_data)
