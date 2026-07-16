from __future__ import annotations

from typing import Any

import torch

from areal.trainer.ppo.actor import PPOActor
from areal.utils.functional import reward_overlong_penalty

_ANSWER_FORMAT = r"\boxed{{{answer}}}"
_ASSISTANT_BEGIN_TOKENS = 3
_ASSISTANT_END_TOKENS = 2


class IGPOActor(PPOActor):
    def _compute_turn_end_pos(
        self, data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eos_id = self.engine.tokenizer.eos_token_id
        attn_mask = data["attention_mask"]
        input_ids = data["input_ids"]
        device = input_ids.device
        batch_size = input_ids.shape[0]

        is_eos = (input_ids == eos_id) & attn_mask.bool()
        turn_ends_list = []
        response_ends_list = []
        max_turns = 0
        for i in range(batch_size):
            eos_pos = torch.where(is_eos[i])[0]
            valid_pos = torch.where(attn_mask[i].bool())[0]
            if valid_pos.numel() != 0:
                last_pos = valid_pos[-1]
                if eos_pos.numel() == 0 or eos_pos[-1] != last_pos:
                    eos_pos = torch.cat([eos_pos, last_pos.unsqueeze(0)])

            turn_ends = eos_pos[0:-1:2]
            response_ends = torch.cat([eos_pos[:1], eos_pos[1:-1:2]])

            aligned_len = min(turn_ends.numel(), response_ends.numel())
            turn_ends = turn_ends[:aligned_len]
            response_ends = response_ends[:aligned_len]

            turn_ends_list.append(turn_ends)
            response_ends_list.append(response_ends)
            max_turns = max(max_turns, len(turn_ends))

        turn_end_pos = torch.full(
            (batch_size, max_turns), -1, dtype=torch.long, device=device
        )
        response_end_pos = torch.full(
            (batch_size, max_turns), -1, dtype=torch.long, device=device
        )
        for i in range(batch_size):
            count = len(turn_ends_list[i])
            if count > 0:
                turn_end_pos[i, :count] = turn_ends_list[i]
                response_end_pos[i, :count] = response_ends_list[i]
        n_turns = torch.tensor(
            [max(len(ends) - 1, 0) for ends in turn_ends_list], device=device
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
            encoded = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": answer_content}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            if isinstance(encoded, dict):
                answer_tokens = encoded["input_ids"]
            else:
                answer_tokens = encoded
            answer_tokens = answer_tokens.to(device).squeeze(0)
            answer_tokens_list.append(answer_tokens)
            answer_spans_list.append(
                (
                    _ASSISTANT_BEGIN_TOKENS,
                    len(answer_tokens) - _ASSISTANT_END_TOKENS,
                )
            )

        return answer_tokens_list, answer_spans_list

    def _compute_ig_rewards(self, data: dict[str, Any]) -> torch.Tensor:
        ground_truths = data.get("ground_truth")
        if ground_truths is None:
            raise ValueError(
                "data['ground_truth'] is missing. IGPO reward computation "
                "requires the workflow to add ground-truth answer strings."
            )

        input_ids = data["input_ids"]
        turn_end_pos = data["turn_end_pos"]
        response_end_pos = data["response_end_pos"]
        n_turns = data["n_turns"]
        device = input_ids.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        max_turns = turn_end_pos.shape[1]

        answer_tokens_list, answer_spans_list = self._prepare_answer_tokens(
            ground_truths, device
        )

        all_seqs = []
        all_answer_masks = []
        for i in range(batch_size):
            answer_tokens = answer_tokens_list[i]
            answer_start, answer_end = answer_spans_list[i]
            for k in range(int(n_turns[i].item()) + 1):
                cut_pos = turn_end_pos[i, k].item()
                if cut_pos < 0:
                    break
                prefix = input_ids[i, : cut_pos + 1]
                full = torch.cat([prefix, answer_tokens])
                mask = torch.zeros(len(full), dtype=torch.bool, device=device)
                mask[
                    len(prefix) + answer_start - 1 : len(prefix) + answer_end - 1
                ] = True
                all_seqs.append(full)
                all_answer_masks.append(mask)

        n_seqs = len(all_seqs)
        if n_seqs == 0:
            return torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

        max_seq_len = max(len(seq) for seq in all_seqs)
        batch_ids = torch.zeros(
            n_seqs, max_seq_len, dtype=input_ids.dtype, device=device
        )
        batch_attn = torch.zeros(
            n_seqs, max_seq_len, dtype=torch.bool, device=device
        )
        batch_answer_mask = torch.zeros(
            n_seqs, max_seq_len, dtype=torch.bool, device=device
        )
        for j, seq in enumerate(all_seqs):
            length = len(seq)
            batch_ids[j, :length] = seq
            batch_attn[j, :length] = True
            batch_answer_mask[j, :length] = all_answer_masks[j]

        self.engine.eval()
        with torch.no_grad():
            logprobs = self.engine.forward(
                input_={"input_ids": batch_ids, "attention_mask": batch_attn},
                aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
            ).to(dtype=torch.float32)

            per_turn_logp = (
                (logprobs * batch_answer_mask.float()).sum(dim=-1)
                / batch_answer_mask.sum(dim=-1).clamp(min=1)
            )
            all_diffs = per_turn_logp[1:] - per_turn_logp[:-1]

            ig_rewards = torch.zeros(
                batch_size, max_turns, dtype=torch.float32, device=device
            )
            offset = 0
            for i in range(batch_size):
                turn_count = int(n_turns[i].item())
                if turn_count > 0:
                    ig_rewards[i, :turn_count] = all_diffs[offset : offset + turn_count]
                offset += turn_count + 1

            data["ig_rewards"] = ig_rewards

            valid_mask = torch.arange(max_turns, device=device) < n_turns[:, None]
            valid_ig = ig_rewards[valid_mask]
            if valid_ig.numel() > 0:
                mu_ig = valid_ig.mean()
                sigma_ig = valid_ig.std(unbiased=False)
            else:
                mu_ig = torch.tensor(0.0, device=device)
                sigma_ig = torch.tensor(1.0, device=device)
            ig_norm = torch.where(
                valid_mask,
                (ig_rewards - mu_ig) / (sigma_ig + 1e-8),
                torch.zeros_like(ig_rewards),
            )

            ig_token_rewards = torch.zeros(
                batch_size, seq_len, dtype=torch.float32, device=device
            )
            for i in range(batch_size):
                for k in range(int(n_turns[i].item())):
                    pos = response_end_pos[i, k + 1].item()
                    if pos >= 0:
                        ig_token_rewards[i, pos] = ig_norm[i, k]

        return ig_token_rewards.detach()

    def _compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        batch_size = data["input_ids"].shape[0]
        max_seq_len = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            batch_size, device=data["input_ids"].device, dtype=torch.long
        )

        turn_end_pos, response_end_pos, n_turns = self._compute_turn_end_pos(data)
        data["turn_end_pos"] = turn_end_pos
        data["response_end_pos"] = response_end_pos
        data["n_turns"] = n_turns

        ig_token_rewards = self._compute_ig_rewards(data)
        data["ig_token_rewards"] = ig_token_rewards

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

        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
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
                data["prox_logp"] = old_logp

        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        attn_mask = data["attention_mask"]
        seq_lens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seq_lens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        rewards[batch_indices, seq_lens - 1] = 0
        indices = torch.clip(seq_lens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        rewards += ig_token_rewards

        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        advantages_reversed = [
            torch.zeros(batch_size, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seq_len - 1] * seq_no_eos_mask
        for t in reversed(range(max_seq_len - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        data["returns"] = advantages + values

        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        return data
