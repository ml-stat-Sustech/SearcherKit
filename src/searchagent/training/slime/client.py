from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TurnRecord
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post

from searchagent.llm.base import Client
from searchagent.log import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SlimeCompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chat_template_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        return {}
    value = extra_body.get("chat_template_kwargs")
    return dict(value) if isinstance(value, dict) else {}


def _find_kth_token(tokens: list[int], target: int, count: int) -> int:
    if count <= 0:
        return -1
    seen = 0
    for idx, token in enumerate(tokens):
        if token == target:
            seen += 1
            if seen == count:
                return idx
    return -1


def _output_ids_without_stop(
    output_ids: list[int],
    *,
    finish_reason: str,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    if finish_reason in {"length", "abort"} or not output_ids:
        return list(output_ids)

    stop_token_ids = {token_id for token_id in (eos_token_id, pad_token_id) if token_id is not None}
    if not stop_token_ids:
        return list(output_ids)

    end = len(output_ids)
    while end > 0 and output_ids[end - 1] in stop_token_ids:
        end -= 1
    return list(output_ids[:end])


def concat_prompt_ids_with_parent(
    rendered_prompt_ids: list[int],
    *,
    parent: TurnRecord | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    """Build an AReal-like concat prompt from the previous raw token trace."""
    if parent is None or eos_token_id is None:
        return list(rendered_prompt_ids)

    parent_tokens = list(parent.prompt_ids) + _output_ids_without_stop(
        list(parent.output_ids),
        finish_reason=parent.finish_reason,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    parent_tokens.append(eos_token_id)

    parent_eos_count = parent_tokens.count(eos_token_id)
    child_cut = _find_kth_token(rendered_prompt_ids, eos_token_id, parent_eos_count)
    if child_cut < 0 or child_cut + 1 >= len(rendered_prompt_ids):
        logger.warning(
            "[slime-client] concat prompt alignment failed; using rendered prompt "
            "(parent_eos=%d rendered_eos=%d)",
            parent_eos_count,
            rendered_prompt_ids.count(eos_token_id),
        )
        return list(rendered_prompt_ids)
    return parent_tokens + list(rendered_prompt_ids[child_cut + 1 :])


class SlimeSGLangClient(Client):
    """SearchAgent client backed by slime's SGLang rollout router."""

    def __init__(
        self,
        *,
        args: Any,
        tokenizer: Any,
        sampling_params: dict[str, Any],
        default_kwargs: dict[str, Any] | None = None,
        use_provider_tools: bool = False,
        tool_call_parser: str | None = "qwen",
        reasoning_parser: str | None = "qwen3",
        max_context_tokens: int = 0,
        model_name: str = "default",
        session_id: str | None = None,
    ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.default_kwargs = default_kwargs or {}
        self.use_provider_tools = use_provider_tools
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name
        self.session_id = session_id
        self.turns: list[TurnRecord] = []
        self.context_truncated = False
        self.url = get_model_url(args, model_name, "/generate")

    async def complete(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return (await self.complete_with_usage(messages, session_id=session_id, **kwargs))[0]

    async def complete_with_usage(
        self,
        messages: Iterable[dict[str, Any]],
        session_id: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], SlimeCompletionUsage]:
        message_list = messages if isinstance(messages, list) else list(messages)
        payload = {**self.default_kwargs, **kwargs}
        tools = payload.pop("tools", None)
        prompt_ids = self._render_prompt(message_list, tools, payload)
        turn = await self._generate(prompt_ids, payload, session_id=session_id)
        self.turns.append(turn)

        raw_output = (
            self.tokenizer.decode(turn.output_ids, skip_special_tokens=False)
            if turn.output_ids
            else ""
        )
        message = self._assistant_message(raw_output, tools)
        usage = SlimeCompletionUsage(
            prompt_tokens=len(turn.prompt_ids),
            completion_tokens=len(turn.output_ids),
            total_tokens=len(turn.prompt_ids) + len(turn.output_ids),
        )
        logger.debug(
            "SGLang rollout turn finished prompt_tokens=%s completion_tokens=%s finish=%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            turn.finish_reason,
        )
        return message, usage

    def _render_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        payload: dict[str, Any],
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self.use_provider_tools and tools:
            kwargs["tools"] = tools
        template_kwargs = _chat_template_kwargs(payload)
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                **kwargs,
                **template_kwargs,
            )
        except TypeError:
            if not template_kwargs:
                raise
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            rendered_prompt_ids = list(encoded["input_ids"])
        else:
            rendered_prompt_ids = list(encoded)

        return concat_prompt_ids_with_parent(
            rendered_prompt_ids,
            parent=self.turns[-1] if self.turns else None,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )

    def _sampling_params(self, payload: dict[str, Any], prompt_len: int) -> dict[str, Any]:
        params = dict(self.sampling_params)
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("stop", "stop"),
            ("stop_token_ids", "stop_token_ids"),
            ("skip_special_tokens", "skip_special_tokens"),
        ):
            if payload.get(src) is not None:
                params[dst] = payload[src]
        for key in ("max_completion_tokens", "max_tokens", "max_new_tokens"):
            if payload.get(key) is not None:
                params["max_new_tokens"] = _as_int(payload[key], _as_int(params.get("max_new_tokens"), 4096))
                break
        params.setdefault("skip_special_tokens", False)
        params.setdefault("spaces_between_special_tokens", False)
        params.setdefault("no_stop_trim", True)
        params.setdefault("max_new_tokens", 4096)
        if self.max_context_tokens > 0:
            remaining = self.max_context_tokens - prompt_len
            if remaining <= 0:
                return {**params, "max_new_tokens": 0}
            params["max_new_tokens"] = min(_as_int(params.get("max_new_tokens"), remaining), remaining)
        return params

    async def _generate(
        self,
        prompt_ids: list[int],
        payload: dict[str, Any],
        *,
        session_id: int | None,
    ) -> TurnRecord:
        sampling_params = self._sampling_params(payload, len(prompt_ids))
        if _as_int(sampling_params.get("max_new_tokens"), 0) <= 0:
            self.context_truncated = True
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")

        headers = None
        routing_key = str(session_id) if session_id is not None else self.session_id
        if routing_key and getattr(self.args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": routing_key}

        output = await post(
            self.url,
            {
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            headers=headers,
        )
        if not isinstance(output, dict):
            raise TypeError(f"SGLang /generate returned {type(output).__name__}, expected dict")
        meta = output.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [item[1] for item in output_token_logprobs]
        output_log_probs = [float(item[0]) for item in output_token_logprobs]
        if not output_ids and output.get("text"):
            output_ids = self.tokenizer.encode(output["text"], add_special_tokens=False)
            output_log_probs = [0.0] * len(output_ids)
        finish_payload = meta.get("finish_reason") or {}
        finish_reason = (
            str(finish_payload.get("type", "stop"))
            if isinstance(finish_payload, dict)
            else str(finish_payload or "stop")
        )
        if finish_reason == "length":
            self.context_truncated = True
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=list(output_ids),
            finish_reason=finish_reason,
            output_log_probs=list(output_log_probs),
        )

    def _assistant_message(
        self,
        raw_output: str,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not self.use_provider_tools:
            return {"role": "assistant", "content": raw_output}

        parsed = parse_model_output(
            raw_output,
            tools_schema=tools,
            tool_parser_name=self.tool_call_parser if tools else None,
            reasoning_parser_name=self.reasoning_parser,
        )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.text or None,
        }
        if parsed.reasoning:
            message["reasoning"] = parsed.reasoning
            message["reasoning_content"] = parsed.reasoning
        if parsed.tool_uses:
            message["tool_calls"] = [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": str(tool_use.get("name", "")),
                        "arguments": json.dumps(
                            tool_use.get("input", {}),
                            ensure_ascii=False,
                        ),
                    },
                }
                for idx, tool_use in enumerate(parsed.tool_uses)
            ]
        return message
