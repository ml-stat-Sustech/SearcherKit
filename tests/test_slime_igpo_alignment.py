from searchagent.training.slime.rollout import _build_igpo_token_rewards


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
