from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _step_level_ppo_inputs(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    step_boundary_mask: torch.Tensor | None = None,
    behavior_log_ratio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # All PPO inputs describe the same token positions, whether padded or packed.
    if not (
        logprobs.shape
        == proximal_logprobs.shape
        == old_logprobs.shape
        == advantages.shape
        == loss_mask.shape
    ):
        raise ValueError("step-level PPO inputs must have matching shapes")

    # Packed batches need explicit sample ranges so adjacent samples never form one step.
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

    step_proximal_log_ratio = (logprobs - proximal_logprobs).clone()
    step_advantages = advantages.clone()
    if behavior_log_ratio is None:
        behavior_log_ratio = (proximal_logprobs - old_logprobs).detach()
    elif behavior_log_ratio.shape != logprobs.shape:
        raise ValueError("behavior log ratio must match step-level PPO input shapes")
    step_behavior_log_ratio = behavior_log_ratio.clone()
    mask = loss_mask.bool()

    # step_boundary_mask is the pre-rejection loss mask: its contiguous spans define
    # the original assistant-step boundaries. loss_mask is the post-rejection mask:
    # it selects which tokens within each original step contribute to aggregation.
    # Keeping them separate prevents a rejected middle token from splitting one step.
    boundary_mask = mask if step_boundary_mask is None else step_boundary_mask.bool()
    if boundary_mask.shape != mask.shape:
        raise ValueError("step boundary mask must match loss_mask shape")

    for row, start, end in ranges:
        item_boundary_mask = (
            boundary_mask[start:end] if row is None else boundary_mask[row, start:end]
        )
        item_mask = mask[start:end] if row is None else mask[row, start:end]

        # Each contiguous run in the boundary mask is one assistant step.
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
            mean_proximal_log_ratio = step_proximal_log_ratio[index].mean()
            mean_behavior_log_ratio = behavior_log_ratio[index].mean()

            # Broadcast each geometric-mean ratio across its step so the copied PPO
            # calculation below can retain AReaL's token-shaped loss and statistics.
            step_proximal_log_ratio[index] = mean_proximal_log_ratio
            step_behavior_log_ratio[index] = mean_behavior_log_ratio
            step_advantages[index] = advantages[index].mean()

    return step_proximal_log_ratio, step_behavior_log_ratio, step_advantages


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
        eps_clip: float = 0.2,
        eps_clip_higher: float | None = None,
        c_clip: float | None = None,
        rejection_sampling: Any | None = None,
        importance_sampling_level: str = "token",
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        common_kwargs = dict(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            c_clip=c_clip,
            loss_mask=loss_mask,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        if importance_sampling_level != "step":
            return ppo_actor_loss_fn(**common_kwargs)

        # Keep the original token count as the loss denominator. Rejection may
        # narrow the mask, but it must not amplify the gradients of retained tokens.
        loss_mask_count = loss_mask.count_nonzero() or 1

        # Rejection sampling remains an unchanged preprocessing stage. Its mask
        # chooses the positions to aggregate, and its emitted weights carry clamps.
        aggregation_mask = loss_mask
        behavior_log_ratio = None
        rejection_result = None
        if rejection_sampling is not None:
            from areal.utils.functional import apply_rejection_sampling

            rejection_result = apply_rejection_sampling(
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                loss_mask=loss_mask,
                cu_seqlens=cu_seqlens,
                config=rejection_sampling,
            )
            aggregation_mask = rejection_result.loss_mask

            # Preserve rejection sampling decisions, including token-level clamping,
            # before reducing behavior weights at the PPO step level.
            behavior_weight = rejection_result.behave_imp_weight.detach()
            behavior_log_ratio = behavior_weight.log()

        (
            step_proximal_log_ratio,
            step_behavior_log_ratio,
            step_advantages,
        ) = _step_level_ppo_inputs(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_mask=aggregation_mask,
            cu_seqlens=cu_seqlens,
            step_boundary_mask=loss_mask,
            behavior_log_ratio=behavior_log_ratio,
        )

        # Compute one proximal ratio per step, then apply the same clipping branches
        # as AReaL's PPO loss. Broadcasting keeps every token gradient in a step
        # equally weighted while the mean log-ratio supplies length normalization.
        mask = aggregation_mask.bool()
        ratio = torch.where(mask, torch.exp(step_proximal_log_ratio), 0)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - eps_clip,
            1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
        )

        pg_loss1 = -step_advantages * ratio
        pg_loss2 = -step_advantages * clipped_ratio
        clip_mask = pg_loss1.detach() < pg_loss2.detach()
        pg_loss = torch.maximum(pg_loss1, pg_loss2)
        if c_clip is not None:
            assert c_clip > 1.0, c_clip
            pg_loss3 = torch.sign(step_advantages) * c_clip * step_advantages
            dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
            pg_loss = torch.minimum(pg_loss, pg_loss3)
        else:
            dual_clip_mask = torch.zeros_like(clip_mask)

        # Decoupled PPO multiplies both clipping branches by the detached behavior
        # correction emitted by rejection sampling, geometrically averaged per step.
        if rejection_result is not None:
            step_behavior_weight = torch.where(
                mask,
                torch.exp(step_behavior_log_ratio),
                0,
            )
            behave_approx_kl = proximal_logprobs.detach() - old_logprobs.detach()
            behave_mask = (step_behavior_weight > 0).logical_and(mask)
            behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)
            pg_loss = pg_loss * step_behavior_weight

        logging_loss = pg_loss.detach()
        pg_loss = torch.where(mask, pg_loss, 0).sum() / loss_mask_count
        clip_mask.logical_and_(mask)
        dual_clip_mask.logical_and_(mask)
        stats = {
            "loss": logging_loss,
            "importance_weight": ratio.detach(),
            # AReaL reports the original per-token KL proxy even when its clipping
            # ratio is aggregated at sequence level; retain that meaning for steps.
            "approx_kl": (logprobs - proximal_logprobs).detach(),
            "clip_mask": clip_mask,
            "dual_clip_mask": dual_clip_mask,
        }
        if rejection_result is not None:
            stats.update(
                behave_approx_kl=behave_approx_kl.detach(),
                behave_imp_weight=step_behavior_weight.detach(),
                behave_mask=behave_mask,
                filtered_fraction=rejection_result.filtered_fraction,
            )
        return pg_loss, stats

    step_level_ppo_loss_fn._searcherkit_step_level = True  # type: ignore[attr-defined]
    return step_level_ppo_loss_fn
