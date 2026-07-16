from __future__ import annotations

import torch


def _step_level_tensors_for_sample(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if ppo_kl.shape != advantages.shape:
        raise ValueError(f"ppo_kl and advantages shape mismatch: {ppo_kl.shape} vs {advantages.shape}")
    if ppo_kl.shape != loss_mask.shape:
        raise ValueError(f"ppo_kl and loss_mask shape mismatch: {ppo_kl.shape} vs {loss_mask.shape}")

    mask = loss_mask.bool()
    step_kl = torch.zeros_like(ppo_kl)
    step_advantages = torch.zeros_like(advantages)
    step_ratio = torch.zeros_like(ppo_kl)
    step_token_count = torch.zeros_like(ppo_kl, dtype=torch.float32)
    if not mask.any():
        return step_kl, step_advantages, {
            "ppo_step_ratio": step_ratio,
            "ppo_step_token_count": step_token_count,
        }

    starts = mask.clone()
    starts[1:] &= ~mask[:-1]
    segment_idx = starts.to(torch.int64).cumsum(dim=0) - 1
    valid_segment_idx = segment_idx[mask]
    num_segments = int(starts.sum().item())

    kl_sums = torch.zeros(num_segments, dtype=ppo_kl.dtype, device=ppo_kl.device).scatter_add_(
        0,
        valid_segment_idx,
        ppo_kl[mask],
    )
    adv_sums = torch.zeros(num_segments, dtype=advantages.dtype, device=advantages.device).scatter_add_(
        0,
        valid_segment_idx,
        advantages[mask],
    )
    counts = torch.zeros(num_segments, dtype=torch.float32, device=ppo_kl.device).scatter_add_(
        0,
        valid_segment_idx,
        torch.ones_like(valid_segment_idx, dtype=torch.float32),
    )

    kl_per_segment = kl_sums / counts.to(ppo_kl.dtype)
    adv_per_segment = adv_sums / counts.to(advantages.dtype)
    ratio_per_segment = torch.exp(-kl_per_segment)

    step_kl[mask] = kl_per_segment[valid_segment_idx]
    step_advantages[mask] = adv_per_segment[valid_segment_idx]
    step_ratio[mask] = ratio_per_segment[valid_segment_idx]
    step_token_count[mask] = counts[valid_segment_idx]
    return step_kl, step_advantages, {
        "ppo_step_ratio": step_ratio,
        "ppo_step_token_count": step_token_count,
    }


def compute_step_level_ppo_kl_and_advantages(
    *,
    ppo_kl: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    step_kl: list[torch.Tensor] = []
    step_advantages: list[torch.Tensor] = []
    metrics: dict[str, list[torch.Tensor]] = {
        "ppo_step_ratio": [],
        "ppo_step_token_count": [],
    }
    for kl_item, adv_item, mask_item in zip(ppo_kl, advantages, loss_masks, strict=True):
        kl_out, adv_out, sample_metrics = _step_level_tensors_for_sample(
            kl_item,
            adv_item,
            mask_item.to(device=kl_item.device),
        )
        step_kl.append(kl_out)
        step_advantages.append(adv_out)
        for key, value in sample_metrics.items():
            metrics[key].append(value)
    return step_kl, step_advantages, metrics


def step_level_metric_tensors(metrics: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {}
    for key, values in metrics.items():
        if values:
            out[key] = torch.cat(values, dim=0).detach()
    return out
