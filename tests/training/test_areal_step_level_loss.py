from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from searcherkit.training.areal.step_level_loss import (
    _step_level_logprobs_and_advantages,
    make_step_level_ppo_loss,
)


def test_step_level_inputs_aggregate_contiguous_mask_spans() -> None:
    logprobs, advantages = _step_level_logprobs_and_advantages(
        logprobs=torch.tensor([[0.2, 0.4, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 4),
        advantages=torch.tensor([[1.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 0, 1]]),
        cu_seqlens=None,
    )

    assert logprobs[0, [0, 1, 3]].tolist() == pytest.approx([0.3, 0.3, -0.2])
    assert advantages[0, [0, 1, 3]].tolist() == pytest.approx([2.0, 2.0, 5.0])


def test_step_level_inputs_do_not_merge_adjacent_packed_sequences() -> None:
    logprobs, advantages = _step_level_logprobs_and_advantages(
        logprobs=torch.tensor([0.2, 0.4, -0.2, -0.4]),
        proximal_logprobs=torch.zeros(4),
        advantages=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        loss_mask=torch.ones(4),
        cu_seqlens=torch.tensor([0, 2, 4]),
    )

    assert logprobs.tolist() == pytest.approx([0.3, 0.3, -0.3, -0.3])
    assert advantages.tolist() == pytest.approx([2.0, 2.0, 6.0, 6.0])


def test_packed_step_level_inputs_require_sequence_boundaries() -> None:
    with pytest.raises(ValueError, match="cu_seqlens is required"):
        _step_level_logprobs_and_advantages(
            logprobs=torch.zeros(2),
            proximal_logprobs=torch.zeros(2),
            advantages=torch.zeros(2),
            loss_mask=torch.ones(2),
            cu_seqlens=None,
        )


def test_step_level_wrapper_passes_each_step_mean_to_ppo_clipping() -> None:
    received = {}

    def fake_ppo_loss(**kwargs):
        received.update(kwargs)
        return torch.tensor(0.0), {}

    loss_fn = make_step_level_ppo_loss(fake_ppo_loss)
    loss_fn(
        logprobs=torch.tensor([[0.2, 0.4, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 4),
        old_logprobs=torch.zeros(1, 4),
        advantages=torch.tensor([[1.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 0, 1]]),
        importance_sampling_level="step",
    )

    assert received["importance_sampling_level"] == "token"
    assert received["logprobs"][0, [0, 1, 3]].tolist() == pytest.approx(
        [0.3, 0.3, -0.2]
    )
    assert received["advantages"][0, [0, 1, 3]].tolist() == pytest.approx(
        [2.0, 2.0, 5.0]
    )


def test_step_level_wrapper_aggregates_after_rejection_mask(monkeypatch) -> None:
    received = {}

    def fake_apply_rejection_sampling(**kwargs):
        return SimpleNamespace(loss_mask=torch.tensor([[1, 0, 1, 0, 1]]))

    monkeypatch.setitem(
        sys.modules,
        "areal.utils.functional",
        SimpleNamespace(apply_rejection_sampling=fake_apply_rejection_sampling),
    )

    def fake_ppo_loss(**kwargs):
        received.update(kwargs)
        return torch.tensor(0.0), {}

    make_step_level_ppo_loss(fake_ppo_loss)(
        logprobs=torch.tensor([[0.2, 9.0, 0.6, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 5),
        old_logprobs=torch.zeros(1, 5),
        advantages=torch.tensor([[1.0, 99.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 1, 0, 1]]),
        rejection_sampling=SimpleNamespace(action="mask"),
        importance_sampling_level="step",
    )

    assert received["logprobs"][0, [0, 2, 4]].tolist() == pytest.approx(
        [0.4, 0.4, -0.2]
    )
    assert received["advantages"][0, [0, 2, 4]].tolist() == pytest.approx(
        [2.0, 2.0, 5.0]
    )
    assert received["loss_mask"].tolist() == [[1, 1, 1, 0, 1]]


def test_step_level_wrapper_keeps_full_step_for_rejection_clamp(monkeypatch) -> None:
    received = {}

    def fake_apply_rejection_sampling(**kwargs):
        return SimpleNamespace(loss_mask=kwargs["loss_mask"])

    monkeypatch.setitem(
        sys.modules,
        "areal.utils.functional",
        SimpleNamespace(apply_rejection_sampling=fake_apply_rejection_sampling),
    )

    def fake_ppo_loss(**kwargs):
        received.update(kwargs)
        return torch.tensor(0.0), {}

    make_step_level_ppo_loss(fake_ppo_loss)(
        logprobs=torch.tensor([[0.2, 0.4]]),
        proximal_logprobs=torch.zeros(1, 2),
        old_logprobs=torch.zeros(1, 2),
        advantages=torch.tensor([[1.0, 3.0]]),
        loss_mask=torch.ones(1, 2),
        rejection_sampling=SimpleNamespace(action="clamp"),
        importance_sampling_level="step",
    )

    assert received["logprobs"][0].tolist() == pytest.approx([0.3, 0.3])
    assert received["advantages"][0].tolist() == pytest.approx([2.0, 2.0])


def test_packed_rejection_does_not_merge_steps_across_samples() -> None:
    logprobs, advantages = _step_level_logprobs_and_advantages(
        logprobs=torch.tensor([0.2, 9.0, 0.6, -0.2, -0.4]),
        proximal_logprobs=torch.zeros(5),
        advantages=torch.tensor([1.0, 99.0, 3.0, 5.0, 7.0]),
        loss_mask=torch.tensor([1, 0, 1, 1, 1]),
        step_boundary_mask=torch.ones(5),
        cu_seqlens=torch.tensor([0, 3, 5]),
    )

    assert logprobs[[0, 2, 3, 4]].tolist() == pytest.approx([0.4, 0.4, -0.3, -0.3])
    assert advantages[[0, 2, 3, 4]].tolist() == pytest.approx([2.0, 2.0, 6.0, 6.0])


def test_step_level_wrapper_clips_steps_independently() -> None:
    def ppo_clipping_loss(**kwargs):
        ratio = torch.exp(kwargs["logprobs"] - kwargs["proximal_logprobs"])
        clipped_ratio = ratio.clamp(0.8, 1.2)
        advantages = kwargs["advantages"]
        token_loss = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
        loss = token_loss[kwargs["loss_mask"].bool()].mean()
        return loss, {"importance_weight": ratio.detach()}

    logprobs = torch.tensor([[0.4, 0.4, 0.0, -0.1]], requires_grad=True)
    loss, stats = make_step_level_ppo_loss(ppo_clipping_loss)(
        logprobs=logprobs,
        proximal_logprobs=torch.zeros(1, 4),
        old_logprobs=torch.zeros(1, 4),
        advantages=torch.ones(1, 4),
        loss_mask=torch.tensor([[1, 1, 0, 1]]),
        importance_sampling_level="step",
    )
    loss.backward()

    assert stats["importance_weight"][0, [0, 1, 3]].tolist() == pytest.approx(
        [torch.exp(torch.tensor(0.4)).item()] * 2
        + [torch.exp(torch.tensor(-0.1)).item()]
    )
    assert logprobs.grad[0, :2].tolist() == pytest.approx([0.0, 0.0])
    assert logprobs.grad[0, 3].item() < 0
