from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from searcherkit.training.areal.step_level_loss import (
    _step_level_ppo_inputs,
    make_step_level_ppo_loss,
)


def test_step_level_inputs_aggregate_contiguous_mask_spans() -> None:
    proximal_ratio, behavior_ratio, advantages = _step_level_ppo_inputs(
        logprobs=torch.tensor([[0.2, 0.4, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 4),
        old_logprobs=torch.zeros(1, 4),
        advantages=torch.tensor([[1.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 0, 1]]),
        cu_seqlens=None,
    )

    assert proximal_ratio[0, [0, 1, 3]].tolist() == pytest.approx([0.3, 0.3, -0.2])
    assert behavior_ratio[0, [0, 1, 3]].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert advantages[0, [0, 1, 3]].tolist() == pytest.approx([2.0, 2.0, 5.0])


def test_step_level_inputs_do_not_merge_adjacent_packed_sequences() -> None:
    proximal_ratio, behavior_ratio, advantages = _step_level_ppo_inputs(
        logprobs=torch.tensor([0.2, 0.4, -0.2, -0.4]),
        proximal_logprobs=torch.zeros(4),
        old_logprobs=torch.tensor([-0.2, -0.4, 0.2, 0.4]),
        advantages=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        loss_mask=torch.ones(4),
        cu_seqlens=torch.tensor([0, 2, 4]),
    )

    assert proximal_ratio.tolist() == pytest.approx([0.3, 0.3, -0.3, -0.3])
    assert behavior_ratio.tolist() == pytest.approx([0.3, 0.3, -0.3, -0.3])
    assert advantages.tolist() == pytest.approx([2.0, 2.0, 6.0, 6.0])


def test_packed_step_level_inputs_require_sequence_boundaries() -> None:
    with pytest.raises(ValueError, match="cu_seqlens is required"):
        _step_level_ppo_inputs(
            logprobs=torch.zeros(2),
            proximal_logprobs=torch.zeros(2),
            old_logprobs=torch.zeros(2),
            advantages=torch.zeros(2),
            loss_mask=torch.ones(2),
            cu_seqlens=None,
        )


def test_step_level_wrapper_computes_each_step_mean_without_downstream_call() -> None:
    def fake_ppo_loss(**kwargs):
        raise AssertionError("step mode must not call the original PPO loss")

    loss_fn = make_step_level_ppo_loss(fake_ppo_loss)
    loss, stats = loss_fn(
        logprobs=torch.tensor([[0.2, 0.4, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 4),
        old_logprobs=torch.zeros(1, 4),
        advantages=torch.tensor([[1.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 0, 1]]),
        importance_sampling_level="step",
    )

    assert stats["importance_weight"][0, [0, 1, 3]].tolist() == pytest.approx(
        [math.exp(0.3), math.exp(0.3), math.exp(-0.2)]
    )
    assert stats["approx_kl"][0].tolist() == pytest.approx([0.2, 0.4, 9.0, -0.2])
    expected_loss = (-2.0 * 1.2 * 2 - 5.0 * math.exp(-0.2)) / 3
    assert loss.item() == pytest.approx(expected_loss)


def test_non_step_mode_delegates_to_original_ppo_loss() -> None:
    received = {}

    def fake_ppo_loss(**kwargs):
        received.update(kwargs)
        return torch.tensor(7.0), {"delegated": True}

    loss, stats = make_step_level_ppo_loss(fake_ppo_loss)(
        logprobs=torch.zeros(1, 2),
        proximal_logprobs=torch.zeros(1, 2),
        old_logprobs=torch.zeros(1, 2),
        advantages=torch.ones(1, 2),
        loss_mask=torch.ones(1, 2),
        importance_sampling_level="sequence",
    )

    assert loss.item() == pytest.approx(7.0)
    assert stats == {"delegated": True}
    assert received["importance_sampling_level"] == "sequence"


def test_step_level_wrapper_uses_step_mean_for_behavior_importance_weight() -> None:
    def fake_apply_rejection_sampling(**kwargs):
        return SimpleNamespace(
            loss_mask=kwargs["loss_mask"],
            behave_imp_weight=torch.exp(
                kwargs["proximal_logprobs"] - kwargs["old_logprobs"]
            ),
            filtered_fraction=0.0,
        )

    functional_module = SimpleNamespace(
        apply_rejection_sampling=fake_apply_rejection_sampling
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "areal.utils.functional", functional_module)
        loss, stats = make_step_level_ppo_loss(
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call"))
        )(
            logprobs=torch.zeros(1, 2),
            proximal_logprobs=torch.zeros(1, 2),
            old_logprobs=torch.tensor([[-math.log(4.0), math.log(4.0)]]),
            advantages=torch.ones(1, 2),
            loss_mask=torch.ones(1, 2),
            rejection_sampling=SimpleNamespace(action="clamp"),
            importance_sampling_level="step",
        )

    assert loss.item() == pytest.approx(-1.0)
    assert stats["behave_imp_weight"][0].tolist() == pytest.approx([1.0, 1.0])


def test_step_level_wrapper_aggregates_after_rejection_mask(monkeypatch) -> None:
    rejection_calls = []

    def fake_apply_rejection_sampling(**kwargs):
        rejection_calls.append(kwargs.copy())
        behavior_weight = torch.exp(
            kwargs["proximal_logprobs"] - kwargs["old_logprobs"]
        )
        result_mask = kwargs["loss_mask"].bool() & (behavior_weight <= 5.0)
        return SimpleNamespace(
            loss_mask=result_mask.to(kwargs["loss_mask"].dtype),
            behave_imp_weight=torch.where(
                result_mask, behavior_weight, torch.zeros_like(behavior_weight)
            ),
            filtered_fraction=0.4,
        )

    monkeypatch.setitem(
        sys.modules,
        "areal.utils.functional",
        SimpleNamespace(apply_rejection_sampling=fake_apply_rejection_sampling),
    )

    loss, stats = make_step_level_ppo_loss(
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call"))
    )(
        logprobs=torch.tensor([[0.2, 9.0, 0.6, 9.0, -0.2]]),
        proximal_logprobs=torch.zeros(1, 5),
        old_logprobs=torch.tensor([[-0.2, -9.0, -0.6, -9.0, 0.2]]),
        advantages=torch.tensor([[1.0, 99.0, 3.0, 99.0, 5.0]]),
        loss_mask=torch.tensor([[1, 1, 1, 0, 1]]),
        rejection_sampling=SimpleNamespace(action="mask"),
        importance_sampling_level="step",
    )

    assert len(rejection_calls) == 1
    assert stats["importance_weight"][0].tolist() == pytest.approx(
        [math.exp(0.4), 0.0, math.exp(0.4), 0.0, math.exp(-0.2)]
    )
    assert stats["behave_imp_weight"][0].tolist() == pytest.approx(
        [math.exp(0.4), 0.0, math.exp(0.4), 0.0, math.exp(-0.2)]
    )
    assert stats["behave_mask"].tolist() == [[True, False, True, False, True]]
    assert stats["filtered_fraction"] == pytest.approx(0.4)
    assert rejection_calls[0]["old_logprobs"][0].tolist() == pytest.approx(
        [-0.2, -9.0, -0.6, -9.0, 0.2]
    )
    expected_loss = (
        -2.0 * 1.2 * math.exp(0.4) * 2 - 5.0 * math.exp(-0.2) * math.exp(-0.2)
    ) / 4
    assert loss.item() == pytest.approx(expected_loss)


def test_step_level_wrapper_keeps_full_step_for_rejection_clamp(monkeypatch) -> None:
    def fake_apply_rejection_sampling(**kwargs):
        behavior_weight = torch.exp(
            kwargs["proximal_logprobs"] - kwargs["old_logprobs"]
        ).clamp(max=2.0)
        return SimpleNamespace(
            loss_mask=kwargs["loss_mask"],
            behave_imp_weight=behavior_weight,
            filtered_fraction=0.5,
        )

    monkeypatch.setitem(
        sys.modules,
        "areal.utils.functional",
        SimpleNamespace(apply_rejection_sampling=fake_apply_rejection_sampling),
    )

    loss, stats = make_step_level_ppo_loss(
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call"))
    )(
        logprobs=torch.zeros(1, 2),
        proximal_logprobs=torch.zeros(1, 2),
        old_logprobs=torch.tensor([[-math.log(4.0), math.log(4.0)]]),
        advantages=torch.tensor([[1.0, 3.0]]),
        loss_mask=torch.ones(1, 2),
        rejection_sampling=SimpleNamespace(action="clamp"),
        importance_sampling_level="step",
    )

    expected_weight = math.sqrt(0.5)
    assert loss.item() == pytest.approx(-2.0 * expected_weight)
    assert stats["behave_imp_weight"][0].tolist() == pytest.approx(
        [expected_weight, expected_weight]
    )
    assert stats["filtered_fraction"] == pytest.approx(0.5)


def test_step_level_wrapper_accepts_preprocessed_non_ratio_metric(monkeypatch) -> None:
    received_configs = []

    def fake_apply_rejection_sampling(**kwargs):
        received_configs.append(kwargs["config"])
        return SimpleNamespace(
            loss_mask=kwargs["loss_mask"],
            behave_imp_weight=torch.ones_like(kwargs["loss_mask"]),
            filtered_fraction=0.0,
        )

    monkeypatch.setitem(
        sys.modules,
        "areal.utils.functional",
        SimpleNamespace(apply_rejection_sampling=fake_apply_rejection_sampling),
    )
    config = SimpleNamespace(metric="binary_kl")
    loss, _ = make_step_level_ppo_loss(
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected call"))
    )(
        logprobs=torch.zeros(1, 2),
        proximal_logprobs=torch.zeros(1, 2),
        old_logprobs=torch.zeros(1, 2),
        advantages=torch.ones(1, 2),
        loss_mask=torch.ones(1, 2),
        rejection_sampling=config,
        importance_sampling_level="step",
    )

    assert received_configs == [config]
    assert loss.item() == pytest.approx(-1.0)


def test_packed_rejection_does_not_merge_steps_across_samples() -> None:
    proximal_ratio, behavior_ratio, advantages = _step_level_ppo_inputs(
        logprobs=torch.tensor([0.2, 9.0, 0.6, -0.2, -0.4]),
        proximal_logprobs=torch.zeros(5),
        old_logprobs=torch.zeros(5),
        advantages=torch.tensor([1.0, 99.0, 3.0, 5.0, 7.0]),
        loss_mask=torch.tensor([1, 0, 1, 1, 1]),
        step_boundary_mask=torch.ones(5),
        cu_seqlens=torch.tensor([0, 3, 5]),
    )

    assert proximal_ratio[[0, 2, 3, 4]].tolist() == pytest.approx(
        [0.4, 0.4, -0.3, -0.3]
    )
    assert behavior_ratio.tolist() == pytest.approx([0.0] * 5)
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
