from types import SimpleNamespace

import pytest
import torch

from searcherkit.training.slime.actor_igpo import (
    actor_igpo_reward_metrics,
    build_actor_igpo_score_requests,
    prepare_actor_igpo_reward_tensor,
    scatter_actor_igpo_rewards,
)
from searcherkit.training.slime.client import concat_prompt_ids_with_parent
from searcherkit.training.slime.rollout import (
    _build_igpo_token_rewards,
    areal_outcome_reward_filter,
    areal_outcome_reward_post_process,
)
from searcherkit.training.slime.step_level_loss import compute_step_level_ppo_kl_and_advantages
from slime.agent.trajectory import TurnRecord, merge_turns


def test_igpo_token_rewards_follow_segment_turn_indices():
    token_rewards, reward_mask, raw_rewards = _build_igpo_token_rewards(
        response_length=8,
        spans=[(0, 2), (5, 8)],
        turn_indices=[0, 2],
        answer_logprobs=[-4.0, -3.5, -3.25, -2.0],
    )

    assert raw_rewards == [0.5, 0.25, 1.25]
    assert reward_mask == [0, 1, 0, 0, 0, 0, 0, 1]
    assert token_rewards == [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.25]


def test_merge_turns_concat_keeps_raw_parent_output_when_prompt_rerenders_assistant():
    first = TurnRecord(
        prompt_ids=[10, 11],
        output_ids=[101, 102, 103],
        output_log_probs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    second = TurnRecord(
        # The later chat-template prompt contains the same conversation but the
        # assistant message was parsed and re-rendered as [101, 999, 103].
        prompt_ids=[10, 11, 101, 999, 103, 151645, 201, 202],
        output_ids=[301, 302],
        output_log_probs=[-0.4, -0.5],
        finish_reason="stop",
    )

    segment = merge_turns([first, second], metadata={"label": "42"}, eos_token_id=151645)

    assert segment is not None
    assert segment.prompt_ids == [10, 11]
    assert segment.response_ids == [101, 102, 103, 151645, 201, 202, 301, 302]
    assert segment.loss_mask == [1, 1, 1, 0, 0, 0, 1, 1]
    assert segment.rollout_log_probs == [-0.1, -0.2, -0.3, 0.0, 0.0, 0.0, -0.4, -0.5]
    assert segment.metadata["train_model_output_spans"] == [(0, 3), (6, 8)]
    assert segment.metadata["train_model_output_turn_indices"] == [0, 1]
    assert segment.metadata["merge_prefix_drift_count"] == 1
    assert segment.metadata["merge_concat_recovered_count"] == 1
    assert segment.metadata["merge_truncated_response_tokens"] == 0


def test_concat_prompt_ids_with_parent_uses_raw_parent_tokens_and_child_suffix():
    parent = TurnRecord(
        prompt_ids=[10, 11],
        output_ids=[101, 102, 151645],
        output_log_probs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    rendered_child_prompt = [10, 11, 101, 999, 151645, 201, 202]

    prompt_ids = concat_prompt_ids_with_parent(
        rendered_child_prompt,
        parent=parent,
        eos_token_id=151645,
        pad_token_id=151643,
    )

    assert prompt_ids == [10, 11, 101, 102, 151645, 201, 202]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        assert tokenize is True
        assert add_generation_prompt is False
        content = messages[0]["content"]
        assert content == "\\boxed{42}"
        return [10, 11, 12, 42, 13, 14]


def test_build_actor_igpo_score_requests_use_response_spans_as_prefix_cuts():
    requests = build_actor_igpo_score_requests(
        tokenizer=FakeTokenizer(),
        tokens=[100, 101, 201, 202, 901, 902, 301, 903, 302],
        response_length=7,
        metadata={
            "ground_truth": "42",
            "ig_response_spans": [(0, 2), (4, 5), (6, 7)],
        },
    )

    assert [request.sample_index for request in requests] == [0, 0, 0]
    assert [request.reward_index for request in requests] == [-1, 0, 1]
    assert [request.response_pos for request in requests] == [-1, 1, 4]
    assert [request.answer_mask for request in requests] == [
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, True],
    ]
    assert [request.score_response_length for request in requests] == [1, 1, 1]
    assert [request.token_ids for request in requests] == [
        [100, 101, 10, 11, 12, 42],
        [100, 101, 201, 202, 901, 902, 10, 11, 12, 42],
        [100, 101, 201, 202, 901, 902, 301, 903, 10, 11, 12, 42],
    ]


def test_scatter_actor_igpo_rewards_uses_adjacent_answer_logprob_deltas():
    token_rewards, reward_mask, raw_rewards = scatter_actor_igpo_rewards(
        response_length=5,
        response_positions=[1, 4],
        answer_logprobs=[-4.0, -3.5, -2.0],
    )

    assert raw_rewards == [0.5, 1.5]
    assert reward_mask == [0, 1, 0, 0, 1]
    assert token_rewards == [0.0, 0.5, 0.0, 0.0, 1.5]


def test_actor_igpo_reward_metrics_summarize_scattered_rewards():
    metrics = actor_igpo_reward_metrics(
        token_rewards=[0.0, 0.5, 0.0, -0.25, 0.0],
        reward_mask=[0, 1, 0, 1, 0],
        expected_reward_count=2,
        score_request_count=3,
    )

    assert metrics == {
        "actor_ig_reward_mask_count": 2.0,
        "actor_ig_reward_abs_mean": 0.375,
        "actor_ig_reward_nonzero_ratio": 1.0,
        "actor_ig_expected_reward_count": 2.0,
        "actor_ig_reward_mask_coverage_ratio": 1.0,
        "actor_ig_score_request_count": 3.0,
    }


def test_prepare_actor_igpo_reward_tensor_slices_for_context_parallel(monkeypatch):
    calls = []

    def fake_slice_log_prob_with_cp(values, total_length, response_length, qkv_format, max_seq_len):
        calls.append((list(values), total_length, response_length, qkv_format, max_seq_len))
        return values[1:3]

    monkeypatch.setattr(
        "searcherkit.training.slime.actor_igpo._context_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "searcherkit.training.slime.actor_igpo._slice_log_prob_with_cp",
        fake_slice_log_prob_with_cp,
    )

    tensor = prepare_actor_igpo_reward_tensor(
        values=[0.0, 1.25, 0.0, -0.5],
        total_length=10,
        response_length=4,
        qkv_format="thd",
        max_seq_len=None,
        device="cpu",
    )

    assert tensor.tolist() == [1.25, 0.0]
    assert calls == [([0.0, 1.25, 0.0, -0.5], 10, 4, "thd", None)]


class FakeSample:
    def __init__(self, *, reward, outcome_score, overlong_penalty=0.0):
        self.reward = reward
        self.metadata = {
            "reward": reward,
            "outcome_score": outcome_score,
            "overlong_penalty": overlong_penalty,
            "outcome_reward": outcome_score + overlong_penalty,
        }

    def get_reward_value(self, args):
        return self.reward


def _igpo_reward_args(**overrides):
    defaults = {
        "advantage_estimator": "igpo",
        "rewards_normalization": True,
        "grpo_std_normalization": True,
        "n_samples_per_prompt": 2,
        "rollout_batch_size": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_areal_outcome_reward_filter_uses_outcome_reward_mean_not_shaped_reward():
    args = _igpo_reward_args()
    samples = [
        FakeSample(reward=0.2, outcome_score=0.0),
        FakeSample(reward=0.2, outcome_score=1.0),
    ]

    output = areal_outcome_reward_filter(args, samples)

    assert output.keep is True
    assert output.reason is None


def test_areal_outcome_reward_filter_drops_all_zero_or_all_one_groups():
    args = _igpo_reward_args()

    zero_output = areal_outcome_reward_filter(
        args,
        [
            FakeSample(reward=0.2, outcome_score=0.0),
            FakeSample(reward=0.2, outcome_score=0.0),
        ],
    )
    one_output = areal_outcome_reward_filter(
        args,
        [
            FakeSample(reward=0.2, outcome_score=1.0),
            FakeSample(reward=0.2, outcome_score=1.0),
        ],
    )

    assert zero_output.keep is False
    assert zero_output.reason == "outcome_mean_0.0"
    assert one_output.keep is False
    assert one_output.reason == "outcome_mean_1.0"


def test_areal_outcome_reward_post_process_normalizes_outcome_rewards_not_shaped_rewards():
    args = _igpo_reward_args()
    samples = [
        FakeSample(reward=0.2, outcome_score=0.0),
        FakeSample(reward=0.2, outcome_score=1.0),
    ]

    raw_rewards, rewards = areal_outcome_reward_post_process(args, samples)

    assert raw_rewards == [0.0, 1.0]
    assert rewards == pytest.approx([-0.7071067, 0.7071067], abs=1e-6)


def test_step_level_ppo_ratio_inputs_share_contiguous_loss_mask_spans():
    ppo_kl, advantages, metrics = compute_step_level_ppo_kl_and_advantages(
        ppo_kl=[torch.tensor([0.1, 0.5, 9.0, 0.8])],
        advantages=[torch.tensor([1.0, 3.0, 99.0, 5.0])],
        loss_masks=[torch.tensor([1, 1, 0, 1])],
    )

    assert ppo_kl[0].tolist() == pytest.approx([0.3, 0.3, 0.0, 0.8])
    assert advantages[0].tolist() == pytest.approx([2.0, 2.0, 0.0, 5.0])
    assert metrics["ppo_step_token_count"][0].tolist() == pytest.approx([2.0, 2.0, 0.0, 1.0])
    assert metrics["ppo_step_ratio"][0].tolist() == pytest.approx(
        [torch.exp(torch.tensor(-0.3)).item(), torch.exp(torch.tensor(-0.3)).item(), 0.0, torch.exp(torch.tensor(-0.8)).item()]
    )


def test_step_level_ppo_ratio_inputs_do_not_merge_across_samples():
    ppo_kl, advantages, metrics = compute_step_level_ppo_kl_and_advantages(
        ppo_kl=[torch.tensor([0.1]), torch.tensor([1.0, 3.0])],
        advantages=[torch.tensor([7.0]), torch.tensor([2.0, 6.0])],
        loss_masks=[torch.tensor([1]), torch.tensor([1, 1])],
    )

    assert ppo_kl[0].tolist() == pytest.approx([0.1])
    assert ppo_kl[1].tolist() == pytest.approx([2.0, 2.0])
    assert advantages[0].tolist() == pytest.approx([7.0])
    assert advantages[1].tolist() == pytest.approx([4.0, 4.0])
    assert metrics["ppo_step_token_count"][0].tolist() == pytest.approx([1.0])
    assert metrics["ppo_step_token_count"][1].tolist() == pytest.approx([2.0, 2.0])
