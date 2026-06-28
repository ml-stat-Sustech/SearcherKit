from __future__ import annotations

import torch

from areal.api.cli_args import RejectionSamplingConfig
import areal.utils.functional.functional as functional

_installed = False
_original_ppo_actor_loss_fn = functional.ppo_actor_loss_fn


def _compute_step_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate each contiguous valid-token span in loss_mask as one step."""
    mask = loss_mask.bool()
    starts = mask.clone()
    if log_ratio.ndim == 1:
        starts[1:] &= ~mask[:-1]
        if cu_seqlens is not None:
            packed_seq_len = log_ratio.numel()
            seq_starts = cu_seqlens[cu_seqlens < packed_seq_len].long()
            starts[seq_starts] = mask[seq_starts]
    else:
        starts[:, 1:] &= ~mask[:, :-1]

    flat_mask = mask.reshape(-1)
    flat_starts = starts.reshape(-1)
    flat_log_ratio = log_ratio.reshape(-1)
    flat_advantages = advantages.reshape(-1)

    segment_idx = flat_starts.to(torch.int64).cumsum(dim=0) - 1
    valid_segment_idx = segment_idx[flat_mask]
    num_segments = int(flat_starts.sum().item())
    if num_segments == 0:
        return torch.zeros_like(log_ratio), torch.zeros_like(advantages)

    log_ratio_sum = torch.zeros(
        num_segments, device=log_ratio.device, dtype=log_ratio.dtype
    ).scatter_add_(0, valid_segment_idx, flat_log_ratio[flat_mask])
    advantages_sum = torch.zeros(
        num_segments, device=advantages.device, dtype=advantages.dtype
    ).scatter_add_(0, valid_segment_idx, flat_advantages[flat_mask])
    counts = torch.zeros(
        num_segments, device=loss_mask.device, dtype=torch.int64
    ).scatter_add_(
        0,
        valid_segment_idx,
        torch.ones_like(valid_segment_idx, dtype=torch.int64),
    )

    ratio_per_segment = torch.exp(log_ratio_sum / counts.to(log_ratio.dtype))
    advantages_per_segment = advantages_sum / counts.to(advantages.dtype)

    flat_ratio = torch.zeros_like(flat_log_ratio)
    flat_step_advantages = torch.zeros_like(flat_advantages)
    flat_ratio[flat_mask] = ratio_per_segment[valid_segment_idx]
    flat_step_advantages[flat_mask] = advantages_per_segment[valid_segment_idx]

    return flat_ratio.reshape_as(log_ratio), flat_step_advantages.reshape_as(advantages)


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """AReaL PPO actor loss with step-level importance sampling support."""
    loss_mask_count = loss_mask.count_nonzero() or 1

    if rejection_sampling is not None:
        rs_result = functional.apply_rejection_sampling(
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            config=rejection_sampling,
        )
        loss_mask = rs_result.loss_mask
        behave_imp_weight = rs_result.behave_imp_weight
        filtered_fraction = rs_result.filtered_fraction
    else:
        filtered_fraction = 0.0

    if importance_sampling_level == "sequence":
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = functional._compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "step":
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_step_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token', 'sequence', or 'step'."
        )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    if rejection_sampling is not None:
        behave_approx_kl = proximal_logprobs.detach() - old_logprobs.detach()
        behave_mask = (behave_imp_weight > 0).logical_and(loss_mask.bool())
        behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)
        pg_loss = pg_loss * behave_imp_weight

    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = {
        "loss": logging_loss,
        "importance_weight": ratio.detach(),
        "approx_kl": (logprobs - proximal_logprobs).detach(),
        "clip_mask": clip_mask,
        "dual_clip_mask": dual_clip_mask,
    }
    if rejection_sampling is not None:
        stat.update(
            behave_approx_kl=behave_approx_kl.detach(),
            behave_imp_weight=behave_imp_weight.detach(),
            behave_mask=behave_mask,
            filtered_fraction=filtered_fraction,
        )
    return pg_loss, stat


def install() -> None:
    global _installed

    if _installed:
        return

    functional.ppo_actor_loss_fn = ppo_actor_loss_fn
    _installed = True
