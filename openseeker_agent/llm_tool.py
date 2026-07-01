"""OpenSeeker-style completion and tool loop for SFT BCP evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
import requests
from fastmcp.exceptions import FastMCPError

from prompts import DEVELOPER_PROMPT
from template import render_prompt
from tools import Search, Visit


GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
TOOL_CALLS_BLOCK_RE = re.compile(r"<tool_calls_begin>\s*(.*?)\s*</tool_calls_end>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max(0, int(max_chars * 0.7))
    tail = max_chars - head
    return f"{text[:head]}...<truncated {len(text) - max_chars} chars>...{text[-tail:] if tail > 0 else ''}"


def _print_colored(text: str, color: int) -> None:
    print(f"\033[{color}m{text}\033[0m", end="", flush=True)


def _tool_color(tool_name: str) -> int:
    if tool_name == "search":
        return 34
    if tool_name == "visit":
        return 33
    return 35


def _print_tool_call(tool_name: str, tool_args: Any, tool_response: str) -> None:
    max_chars = 800
    color = _tool_color(tool_name)
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        args_str = str(tool_args)
    _print_colored(f"\n[{tool_name}] args={_truncate_text(args_str, max_chars)}\n", color)
    _print_colored(f"[{tool_name}] response={_truncate_text(tool_response, max_chars)}\n", color)


def normalise_completions_url(base_or_full: str) -> str:
    value = (base_or_full or "").strip()
    if not value:
        raise ValueError("Empty base_url / completions_url")
    if value.endswith("/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/completions"
    if value.endswith("/v1/"):
        return value + "completions"
    if value.endswith("/"):
        return value + "v1/completions"
    return value + "/v1/completions"


def get_stream_response(
    completions_url: str,
    payload: dict[str, Any],
    *,
    print_stream: bool = True,
    max_retries: int = 3,
    connect_timeout: int = 10,
    read_timeout: int = 60,
    max_total_seconds: int = 1200,
    max_idle_seconds: int = 45,
) -> tuple[str, bool]:
    retry_backoff = 1.8
    retry_jitter = 0.4
    too_many_tokens_error = False
    result = ""

    for attempt in range(int(max_retries) + 1):
        start = time.monotonic()
        last = start
        chunks: list[str] = []
        got_done = False
        failed = False

        try:
            with requests.post(
                completions_url,
                json=payload,
                stream=True,
                timeout=(int(connect_timeout), int(read_timeout)),
            ) as response:
                response.raise_for_status()

                full_text_so_far = ""
                for raw_line in response.iter_lines(decode_unicode=False):
                    now = time.monotonic()
                    if now - start > float(max_total_seconds):
                        failed = True
                        break
                    if now - last > float(max_idle_seconds):
                        failed = True
                        break
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue

                    last = now
                    json_str = line.split("data:", 1)[1].strip()
                    if not json_str:
                        continue
                    if json_str == "[DONE]":
                        got_done = not failed
                        break

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, Mapping) and "error" in data:
                        err = data.get("error") or {}
                        if isinstance(err, Mapping):
                            message = str(err.get("message") or "").strip()
                        else:
                            message = str(err).strip()
                        if "maximum context length" in message or "context length" in message:
                            too_many_tokens_error = True
                        if message:
                            print(f"[LLM ERROR] {message}")
                        failed = True
                        break

                    choices = data.get("choices") if isinstance(data, Mapping) else None
                    choice0 = choices[0] if isinstance(choices, list) and choices else {}
                    if not isinstance(choice0, Mapping):
                        choice0 = {}
                    text_piece = choice0.get("text") or ""
                    if isinstance(text_piece, str) and text_piece:
                        if text_piece.startswith(full_text_so_far):
                            delta = text_piece[len(full_text_so_far) :]
                            full_text_so_far = text_piece
                        else:
                            delta = text_piece
                            full_text_so_far += text_piece
                        if delta:
                            chunks.append(delta)
                            if print_stream:
                                print(delta, end="", flush=True)

                    if choice0.get("finish_reason") or choice0.get("matched_stop"):
                        got_done = True
                        break
        except requests.exceptions.RequestException as exc:
            print(f"[LLM ERROR] Request failed: {type(exc).__name__}: {exc}")
            failed = True

        result = "".join(chunks)
        if got_done and not failed:
            return result, too_many_tokens_error
        if too_many_tokens_error:
            return result, too_many_tokens_error
        if attempt < int(max_retries):
            delay = min(10.0, retry_backoff**attempt) + random.uniform(0, retry_jitter)
            time.sleep(delay)
            continue
        return result, too_many_tokens_error

    return result, too_many_tokens_error


def _try_fix_incomplete_json(json_str: str) -> str:
    if not json_str or not str(json_str).strip():
        return json_str

    text = str(json_str).strip()
    text = re.sub(r'"\s+"', '", "', text)
    text = re.sub(r'"\s*\[', '", [', text)
    text = re.sub(r'\]\s*"', '], "', text)
    text = re.sub(r"\}\s*\{", "}, {", text)
    text = re.sub(r",\s*([\]}])", r"\1", text)

    text = text.rstrip()
    missing_brackets = text.count("[") - text.count("]")
    missing_braces = text.count("{") - text.count("}")
    if missing_brackets > 0:
        text += "]" * missing_brackets
    if missing_braces > 0:
        text += "}" * missing_braces
    return text


def _parse_payload(inner: str, errors: list[str]) -> Any:
    try:
        return json.loads(inner)
    except json.JSONDecodeError:
        fixed = _try_fix_incomplete_json(inner)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as exc:
            error_msg = "ERROR: JSON decode failed even after fix attempt"
            print(f"\033[91m{error_msg},{exc}\033[0m")
            errors.append(error_msg)
            return None


def _append_one_tool_call(tool_calls: list[dict[str, Any]], payload: Mapping[str, Any], errors: list[str]) -> None:
    name = payload.get("tool_name") or payload.get("name")
    arguments = payload.get("tool_args") or payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        errors.append("ERROR: Tool call missing or empty name field")
        return
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            errors.append("ERROR: Failed to parse arguments as JSON string")
            arguments = {}
    if not isinstance(arguments, Mapping):
        errors.append("ERROR: Arguments is not a dict")
        arguments = {}
    tool_calls.append({"function": {"name": name.strip(), "arguments": dict(arguments)}})


def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]], str | None]:
    if not text:
        error_msg = "ERROR: No text to parse tool calls"
        print(f"\033[91m{error_msg}\033[0m")
        return "", [], error_msg

    tool_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    blocks = [match.group(1) for match in TOOL_CALLS_BLOCK_RE.finditer(text)]
    scan_targets = blocks if blocks else [text]

    for chunk in scan_targets:
        for match in TOOL_CALL_RE.finditer(chunk):
            inner = (match.group(1) or "").strip()
            if not inner:
                errors.append("ERROR: Found empty <tool_call> tag (inner content is empty)")
                continue
            payload = _parse_payload(inner, errors)
            if isinstance(payload, Mapping):
                _append_one_tool_call(tool_calls, payload, errors)
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, Mapping):
                        _append_one_tool_call(tool_calls, item, errors)
                    else:
                        errors.append("ERROR: List item is not a dict")
            elif payload is not None:
                error_msg = "ERROR: Parsed object is neither dict nor list"
                print(f"\033[91m{error_msg}\033[0m")
                errors.append(error_msg)

    cleaned = TOOL_CALLS_BLOCK_RE.sub("", text)
    return cleaned, tool_calls, "\n".join(errors) if errors else None


def _has_answer_tag(text: str) -> bool:
    return bool(text and "</answer>" in text)


def _split_think_and_content(completion_text: str) -> tuple[str, str]:
    text = completion_text or ""
    if not text:
        return "", ""
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        stripped = stripped[len("<think>") :]
        text = stripped
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
        return reasoning.strip(), rest.lstrip("\n").lstrip()
    return "", text.strip()


def _execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name == "search":
        return Search().call(tool_args)
    if tool_name in ("visit", "visit_summary"):
        return Visit().call(tool_args)
    return "Unknown tool or call tool with incorrect format."


def call_llm_with_tool(
    item: dict[str, Any],
    args: argparse.Namespace,
    *,
    return_metrics: bool = False,
    return_trace: bool = False,
) -> Any:
    query = item["query"]

    base_url = os.getenv("OPENSEEKER_BASE_URL", "YOUR_OPENSEEKER_BASE_URL")
    if base_url == "YOUR_OPENSEEKER_BASE_URL":
        raise ValueError("OPENSEEKER_BASE_URL environment variable is required")
    completions_url = normalise_completions_url(base_url)

    model_name = os.getenv("OPENSEEKER_MODEL", "YOUR_MODEL_NAME")
    if model_name == "YOUR_MODEL_NAME":
        raise ValueError("OPENSEEKER_MODEL environment variable is required")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEVELOPER_PROMPT},
        {"role": "user", "content": query},
    ]
    trace: list[dict[str, Any]] = []
    step_num = 0
    tool_count = 0
    tool_count_max = int(getattr(args, "tool_count_max", 200))
    local_max_tokens = int(getattr(args, "max_tokens", 16384))
    disable_tools = False
    pending_assistant_prefix = ""

    while True:
        injected_prefix = pending_assistant_prefix
        if pending_assistant_prefix:
            prompt_text = render_prompt(messages, [], add_generation_prompt=False) + GEN_PROMPT + injected_prefix
        else:
            prompt_text = render_prompt(messages, [], add_generation_prompt=True)
        pending_assistant_prefix = ""

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": local_max_tokens,
            "stream": True,
            "skip_special_tokens": False,
        }
        completion_text, too_many_tokens_error = get_stream_response(
            completions_url,
            payload,
            print_stream=bool(getattr(args, "print_stream", False)),
        )

        if not completion_text and not too_many_tokens_error:
            raise RuntimeError(f"Empty response from LLM. completions_url={completions_url}")

        if too_many_tokens_error:
            new_local_max = int(local_max_tokens * 0.9)
            if new_local_max > 2048 and new_local_max < local_max_tokens:
                local_max_tokens = new_local_max
                continue
            if 128 < new_local_max <= 2048 and new_local_max < local_max_tokens:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = "I have used too many tokens, so I will conclude my answer.\n"
                disable_tools = True
                local_max_tokens = new_local_max
                continue
            completion_text = (
                GEN_PROMPT
                + "</think>\n\n\n<answer>\nThe max context length has been reached.</answer><|im_end|>\n"
            )
            too_many_tokens_error = False

        if injected_prefix:
            completion_text = GEN_PROMPT + injected_prefix + completion_text
        step_num += 1

        content_for_parse = completion_text.replace("<|im_end|>", "").replace(GEN_PROMPT, "")
        reasoning_content, content_raw = _split_think_and_content(content_for_parse)
        has_answer = _has_answer_tag(content_raw)
        cleaned_text, tool_calls, parse_error = parse_tool_calls_from_text(content_raw)
        if disable_tools and tool_calls:
            tool_calls = []

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content_for_parse}
        messages.append(assistant_msg)
        if return_trace:
            trace.append(
                {
                    "step": step_num,
                    "type": "model_message",
                    "content": {
                        "reasoning_content": reasoning_content,
                        "content": cleaned_text,
                        "tool_calls": tool_calls,
                    },
                }
            )

        if has_answer:
            break

        if not tool_calls:
            error_content = parse_error if parse_error else "ERROR: Tool call parsing failed"
            print(f"\033[91m {error_content}\033[0m")
            messages.append(
                {
                    "role": "tool",
                    "name": "unknown",
                    "content": error_content,
                    "tool_call_id": str(uuid.uuid4()),
                }
            )
            if return_trace:
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_response",
                        "content": {"content": error_content},
                    }
                )
            continue

        for tool_call in tool_calls:
            function = (tool_call or {}).get("function") or {}
            tool_name = str(function.get("name") or "").strip()
            tool_args = function.get("arguments") or {}
            if not isinstance(tool_args, Mapping):
                tool_args = {}
            tool_args = dict(tool_args)

            try:
                tool_output = _execute_tool(tool_name, tool_args)
            except (RuntimeError, ValueError, TypeError, requests.exceptions.RequestException, httpx.HTTPError, FastMCPError) as exc:
                tool_output = f"Error during tool execution: {type(exc).__name__}: {exc}"

            try:
                _print_tool_call(tool_name or "unknown", tool_args, tool_output)
            except (TypeError, ValueError, OSError):
                pass
            tool_count += 1
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output,
                    "tool_call_id": str(uuid.uuid4()),
                }
            )
            if return_trace:
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_call",
                        "content": {"tool_name": tool_name, "tool_args": tool_args},
                    }
                )
                trace.append(
                    {
                        "step": step_num,
                        "type": "tool_response",
                        "content": {"tool_name": tool_name, "tool_response": tool_output},
                    }
                )

            if tool_count >= tool_count_max:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = "I have used too many tools, so I will conclude my answer."
                disable_tools = True
                break

        if tool_count >= tool_count_max:
            break

    full_traj = render_prompt(messages, [], add_generation_prompt=False)
    metrics = {"tool_calls": tool_count, "context_chars": len(full_traj)}
    if return_metrics:
        if return_trace:
            return full_traj, metrics, trace
        return full_traj, metrics
    if return_trace:
        return full_traj, trace
    return full_traj


def get_last_assistant_answer_from_messages(full_traj: str) -> str:
    if not full_traj:
        return ""
    parts = full_traj.split("<|im_start|>assistant")
    if len(parts) < 2:
        return full_traj.strip()
    last = parts[-1]
    if "<|im_end|>" in last:
        last = last.split("<|im_end|>", 1)[0]
    last = TOOL_CALL_RE.sub("", last)
    last = re.sub(r"<tool_response>.*?</tool_response>", "", last, flags=re.DOTALL)
    if "</think>" in last:
        last = last.split("</think>", 1)[-1]
    return last.strip()


def _estimate_tokens_from_chars(n_chars: int) -> int:
    return max(1, int(n_chars / 4))


def solve_query_with_tools(
    query: str,
    *,
    max_tokens: int = 16384,
    tool_count_max: int = 200,
    print_stream: bool = False,
    return_full_traj: bool = True,
    return_trace: bool = True,
) -> dict[str, Any]:
    start = time.time()
    args = argparse.Namespace(
        max_tokens=int(max_tokens),
        tool_count_max=int(tool_count_max),
        print_stream=bool(print_stream),
    )
    full_traj, metrics, trace = call_llm_with_tool(
        {"query": query},
        args,
        return_metrics=True,
        return_trace=return_trace,
    )
    elapsed = time.time() - start
    answer = get_last_assistant_answer_from_messages(full_traj)
    context_chars = int(metrics.get("context_chars", len(full_traj)))
    tool_calls = int(metrics.get("tool_calls", 0))

    result: dict[str, Any] = {
        "answer": answer,
        "tool_calls": tool_calls,
        "elapsed_seconds": elapsed,
        "context_chars": context_chars,
        "context_est_tokens": _estimate_tokens_from_chars(context_chars),
    }
    if return_full_traj:
        result["full_traj"] = full_traj
    if return_trace:
        result["trace"] = trace
    return result
