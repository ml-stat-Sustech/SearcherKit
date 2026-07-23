from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _step_level_logprobs_and_advantages(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    step_boundary_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        logprobs.shape == proximal_logprobs.shape == advantages.shape == loss_mask.shape
    ):
        raise ValueError("step-level PPO inputs must have matching shapes")

    if logprobs.ndim == 2:
        ranges = [(row, 0, logprobs.shape[1]) for row in range(logprobs.shape[0])]
    elif logprobs.ndim == 1:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for packed step-level PPO inputs")
        boundaries = cu_seqlens.tolist()
        ranges = [
            (None, int(start), int(end))
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    else:
        raise ValueError(
            f"step-level PPO expects 1D or 2D tensors, got {logprobs.ndim}D"
        )

    step_logprobs = logprobs.clone()
    step_advantages = advantages.clone()
    log_ratio = logprobs - proximal_logprobs
    mask = loss_mask.bool()
    boundary_mask = mask if step_boundary_mask is None else step_boundary_mask.bool()
    if boundary_mask.shape != mask.shape:
        raise ValueError("step boundary mask must match loss_mask shape")

    for row, start, end in ranges:
        item_boundary_mask = (
            boundary_mask[start:end] if row is None else boundary_mask[row, start:end]
        )
        item_mask = mask[start:end] if row is None else mask[row, start:end]
        valid_positions = torch.where(item_boundary_mask)[0]
        if valid_positions.numel() == 0:
            continue
        split_points = (
            torch.where(valid_positions[1:] != valid_positions[:-1] + 1)[0] + 1
        )
        for positions in torch.tensor_split(valid_positions, split_points.tolist()):
            positions = positions[item_mask[positions]] + start
            if positions.numel() == 0:
                continue
            index = positions if row is None else (row, positions)
            mean_log_ratio = log_ratio[index].mean()
            step_logprobs[index] = proximal_logprobs[index] + mean_log_ratio
            step_advantages[index] = advantages[index].mean()

    return step_logprobs, step_advantages


def make_step_level_ppo_loss(
    ppo_actor_loss_fn: Callable[..., tuple[torch.Tensor, dict[str, Any]]],
) -> Callable[..., tuple[torch.Tensor, dict[str, Any]]]:
    if getattr(ppo_actor_loss_fn, "_searcherkit_step_level", False):
        return ppo_actor_loss_fn

    def step_level_ppo_loss_fn(
        *,
        logprobs: torch.Tensor,
        proximal_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        rejection_sampling: Any | None = None,
        importance_sampling_level: str = "token",
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        common_kwargs = dict(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        if importance_sampling_level != "step":
            return ppo_actor_loss_fn(**common_kwargs)

        aggregation_mask = loss_mask
        if rejection_sampling is not None:
            from areal.utils.functional import apply_rejection_sampling

            aggregation_mask = apply_rejection_sampling(
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                loss_mask=loss_mask,
                cu_seqlens=cu_seqlens,
                config=rejection_sampling,
            ).loss_mask

        step_logprobs, step_advantages = _step_level_logprobs_and_advantages(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            advantages=advantages,
            loss_mask=aggregation_mask,
            cu_seqlens=cu_seqlens,
            step_boundary_mask=loss_mask,
        )
        common_kwargs.update(
            logprobs=step_logprobs,
            advantages=step_advantages,
            importance_sampling_level="token",
        )
        return ppo_actor_loss_fn(**common_kwargs)

    step_level_ppo_loss_fn._searcherkit_step_level = True  # type: ignore[attr-defined]
    return step_level_ppo_loss_fn
