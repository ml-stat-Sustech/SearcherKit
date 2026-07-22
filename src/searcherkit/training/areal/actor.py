from typing import Any

import torch

from areal.trainer.ppo.actor import PPOActor
from areal.utils.functional import (
    reward_overlong_penalty,
)

from searcherkit.training.areal.termination import TerminationReason

# TODO: make answer format configurable instead of hardcoded
_ANSWER_FORMAT = r"\boxed{{{answer}}}"
# Qwen chat template wraps assistant content as
# "<|im_start|>assistant\n{content}<|im_end|>\n".
_ASSISTANT_BEGIN_TOKENS = 3
_ASSISTANT_END_TOKENS = 2


class SearchAgentPPOActor(PPOActor):
    def _punish_bad_last_turns(
        self,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        termination_reasons: list[str],
    ) -> None:
        if len(termination_reasons) != advantages.shape[0]:
            raise ValueError(
                "termination_reason must contain one value per trajectory"
            )

        for batch_index, raw_reason in enumerate(termination_reasons):
            try:
                reason = TerminationReason(raw_reason)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown termination_reason: {raw_reason!r}"
                ) from exc
            if reason is not TerminationReason.BAD_LAST_TURN:
                continue

            trainable_positions = torch.where(loss_mask[batch_index].bool())[0]
            if trainable_positions.numel() == 0:
                continue
            last_position = int(trainable_positions[-1].item())
            first_position = last_position
            while first_position > 0 and loss_mask[batch_index, first_position - 1]:
                first_position -= 1
            advantages[batch_index, first_position : last_position + 1] = -1.0

    def _compute_turn_end_pos(
        self, data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eos_id = self.engine.tokenizer.eos_token_id
        attn_mask = data["attention_mask"]
        input_ids = data["input_ids"]
        device = input_ids.device
        bs = input_ids.shape[0]

        is_eos = (input_ids == eos_id) & attn_mask.bool()
        turn_ends_list = []
        resp_ends_list = []
        max_turns = 0
        for i in range(bs):
            eos_pos = torch.where(is_eos[i])[0]
            # pretend an eos at end of sequences
            valid_pos = torch.where(attn_mask[i].bool())[0]
            if valid_pos.numel() != 0:
                last_pos = valid_pos[-1]
                if eos_pos.numel() == 0 or eos_pos[-1] != last_pos:
                    eos_pos = torch.cat([eos_pos, last_pos.unsqueeze(0)])

            # turn end = [p0, t0, t1, ...]  (even indices, skip answer eos at -1)
            turn_ends = eos_pos[0:-1:2]
            # response end = [p0, a0, a1, ...]  (p0 + odd indices, skip answer eos)
            resp_ends = torch.cat([eos_pos[:1], eos_pos[1:-1:2]])

            # Drop trailing abnormal response without corresponding tool/result turn.
            aligned_len = min(turn_ends.numel(), resp_ends.numel())
            turn_ends = turn_ends[:aligned_len]
            resp_ends = resp_ends[:aligned_len]

            turn_ends_list.append(turn_ends)
            resp_ends_list.append(resp_ends)
            max_turns = max(max_turns, len(turn_ends))

        turn_end_pos = torch.full((bs, max_turns), -1, dtype=torch.long, device=device)
        response_end_pos = torch.full(
            (bs, max_turns), -1, dtype=torch.long, device=device
        )
        for i in range(bs):
            n = len(turn_ends_list[i])
            if n > 0:
                turn_end_pos[i, :n] = turn_ends_list[i]
                response_end_pos[i, :n] = resp_ends_list[i]
        n_turns = torch.tensor(
            [max(len(e) - 1, 0) for e in turn_ends_list], device=device
        )
        return turn_end_pos, response_end_pos, n_turns

    def _prepare_answer_tokens(
        self, ground_truths: list[str], device: torch.device
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        tokenizer = self.engine.tokenizer
        answer_tokens_list = []
        answer_spans_list = []

        for text in ground_truths:
            answer_content = _ANSWER_FORMAT.format(answer=text)
            assistant_tokens = (
                tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": answer_content}],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )["input_ids"]
                .to(device)
                .squeeze(0)
            )
            answer_tokens_list.append(assistant_tokens)

            answer_spans_list.append(
                (_ASSISTANT_BEGIN_TOKENS, len(assistant_tokens) - _ASSISTANT_END_TOKENS)
            )

        return answer_tokens_list, answer_spans_list

    def _compute_ig_rewards(self, data: dict[str, Any]) -> torch.Tensor:
        # TODO: ensure data["ground_truth"] field exists with ground-truth answer strings
        ground_truths = data.get("ground_truth")
        if ground_truths is None:
            raise ValueError(
                "data['ground_truth'] is missing. "
                "The IGPO reward computation requires ground-truth answer strings. "
                "Ensure the workflow adds 'ground_truth' to the trajectory dict."
            )
        input_ids = data["input_ids"]
        turn_end_pos = data["turn_end_pos"]
        response_end_pos = data["response_end_pos"]
        n_turns = data["n_turns"]
        device = input_ids.device
        bs = input_ids.shape[0]
        seqlen = input_ids.shape[1]
        max_turns = turn_end_pos.shape[1]

        answer_tokens_list, answer_spans_list = self._prepare_answer_tokens(
            ground_truths, device
        )

        all_seqs = []
        all_answer_masks = []

        for i in range(bs):
            ans_tok = answer_tokens_list[i]
            ans_start, ans_end = answer_spans_list[i]
            for k in range(int(n_turns[i].item()) + 1):
                cut_pos = turn_end_pos[i, k].item()
                if cut_pos < 0:
                    break
                prefix = input_ids[i, : cut_pos + 1]
                full = torch.cat([prefix, ans_tok])
                mask = torch.zeros(len(full), dtype=torch.bool, device=device)
                # engine.forward() returns next-token logprobs: logprobs[t] scores full[t + 1].
                mask[len(prefix) + ans_start - 1 : len(prefix) + ans_end - 1] = True
                all_seqs.append(full)
                all_answer_masks.append(mask)

        n_seqs = len(all_seqs)
        if n_seqs == 0:
            return torch.zeros(bs, seqlen, dtype=torch.float32, device=device)

        max_seqlen = max(len(s) for s in all_seqs)
        batch_ids = torch.zeros(
            n_seqs, max_seqlen, dtype=input_ids.dtype, device=device
        )
        batch_attn = torch.zeros(n_seqs, max_seqlen, dtype=torch.bool, device=device)
        batch_ans_mask = torch.zeros(
            n_seqs, max_seqlen, dtype=torch.bool, device=device
        )
        for j in range(n_seqs):
            L = len(all_seqs[j])
            batch_ids[j, :L] = all_seqs[j]
            batch_attn[j, :L] = True
            batch_ans_mask[j, :L] = all_answer_masks[j]

        self.engine.eval()
        with torch.no_grad():
            logprobs = self.engine.forward(
                input_={"input_ids": batch_ids, "attention_mask": batch_attn},
                aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
            )
            logprobs = logprobs.to(dtype=torch.float32)

            # per_turn_logp[j] = mean log π(a | context up to cut_pos[j])
            # shape: (n_seqs,)  where n_seqs = Σ_i (n_turns[i] + 1)
            per_turn_logp = (logprobs * batch_ans_mask.float()).sum(
                dim=-1
            ) / batch_ans_mask.sum(dim=-1).clamp(min=1)

            # ig_reward = logp[k+1] - logp[k], vectorized across all items
            # all_diffs length = n_seqs - 1, 跨 item 边界的 diff 在下层循环跳过
            all_diffs = per_turn_logp[1:] - per_turn_logp[:-1]

            # scatter diffs back to (B, max_turns), skipping cross-item boundary diffs
            ig_rewards = torch.zeros(bs, max_turns, dtype=torch.float32, device=device)
            offset = 0
            for i in range(bs):
                n = int(n_turns[i].item())
                if n > 0:
                    ig_rewards[i, :n] = all_diffs[offset : offset + n]
                offset += n + 1  # skip the cross-item diff at position offset+n

            data["ig_rewards"] = ig_rewards

            # Group z-score normalize IG rewards (separate from outcome normalization)
            # NOTE: paper uses γ=1.0, so no per-turn discount accumulation is needed;
            # GAE (λ=1, values=0) handles the forward propagation automatically.
            valid_mask = torch.arange(max_turns, device=device) < n_turns[:, None]
            valid_ig = ig_rewards[valid_mask]
            if valid_ig.numel() > 0:
                mu_ig, sigma_ig = valid_ig.mean(), valid_ig.std()
            else:
                mu_ig, sigma_ig = (
                    torch.tensor(0.0, device=device),
                    torch.tensor(1.0, device=device),
                )
            ig_norm = torch.where(
                valid_mask,
                (ig_rewards - mu_ig) / (sigma_ig + 1e-8),
                torch.zeros_like(ig_rewards),
            )

            # Scatter normalized IG rewards to a_k eos positions (response_end[k+1]).
            # GAE (λ=1, values=0) will back-propagate each R̃_t to all tokens in turn t.
            # shape: (B, seqlen)
            ig_token_rewards = torch.zeros(
                bs, seqlen, dtype=torch.float32, device=device
            )
            for i in range(bs):
                for k in range(int(n_turns[i].item())):
                    pos = response_end_pos[i, k + 1].item()  # skip p0 at index 0
                    if pos >= 0:
                        ig_token_rewards[i, pos] = ig_norm[i, k]

        return ig_token_rewards.detach()

    def _compute_advantages(
        self, data: dict[str, Any], meta: Any | None = None
    ) -> dict[str, Any]:
        if not (
            self.config.enable_igpo_reward or self.config.punish_last_turn
        ):
            return super()._compute_advantages(data, meta=meta)

        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        ig_token_rewards = torch.zeros(
            bs, max_seqlen, dtype=torch.float32, device=data["input_ids"].device
        )
        if self.config.enable_igpo_reward:
            # IGPO: turn boundaries from eos token ordering
            turn_end_pos, response_end_pos, n_turns = self._compute_turn_end_pos(data)
            data["turn_end_pos"] = turn_end_pos
            data["response_end_pos"] = response_end_pos
            data["n_turns"] = n_turns

            # IGPO: compute information-gain rewards per turn
            ig_token_rewards = self._compute_ig_rewards(data)
            data["ig_token_rewards"] = ig_token_rewards

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        group_sizes = meta.traj_group_sizes if meta is not None else None
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score, group_sizes=group_sizes)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        # IGPO: inject turn-level reward at each response_end (a_k eos).
        # GAE (λ=1, values=0) will back-propagate to all turn tokens.
        rewards += ig_token_rewards

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        if self.config.punish_last_turn:
            termination_reasons = data.get("termination_reason")
            if termination_reasons is None:
                raise ValueError(
                    "data['termination_reason'] is required when "
                    "punish_last_turn=True"
                )
            self._punish_bad_last_turns(
                advantages, loss_mask, termination_reasons
            )

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask, group_sizes=group_sizes)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

        return data
