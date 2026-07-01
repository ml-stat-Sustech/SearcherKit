"""OpenSeeker JSONL to ms-swift JSONL converter."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.DOTALL)
TOOL_WRAPPER_RE = re.compile(r"</?tool_calls_begin>\s*|</?tool_calls_end>\s*", re.DOTALL)


@dataclass(slots=True)
class ConversionStats:
    total: int = 0
    written: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"total": self.total, "written": self.written, "skipped": self.skipped}


def _parse_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    items: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _render_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": dict(arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any] | None:
    name = str(tool.get("name") or "")
    if not name:
        return None
    parameters = tool.get("parameters") or tool.get("inputSchema") or {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": dict(parameters),
        },
    }


def parse_tools(system_text: str) -> list[dict[str, Any]]:
    matches = [match.group(1) for match in TOOL_BLOCK_RE.finditer(system_text) if match.group(1).strip()]
    if not matches:
        return []
    tools: list[dict[str, Any]] = []
    for raw_tool in _parse_json_objects(matches[-1]):
        tool = _openai_tool(raw_tool)
        if tool is not None:
            tools.append(tool)
    return tools


def normalize_system(system_text: str) -> str:
    return system_text.split("# Tools", 1)[0].strip()


def normalize_tool_call(raw_text: str) -> str:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = {"name": "invalid_tool_call", "arguments": {"raw": raw_text}}
    if not isinstance(payload, Mapping):
        payload = {"name": "invalid_tool_call", "arguments": {"raw": raw_text}}
    name = str(payload.get("name") or "invalid_tool_call")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {"raw": arguments}
    return _render_tool_call(name, arguments)


def split_assistant_message(text: str) -> list[dict[str, str]]:
    cleaned = TOOL_WRAPPER_RE.sub("", text).strip()
    if "<tool_call>" not in cleaned:
        return [{"role": "assistant", "content": cleaned}] if cleaned else []

    messages: list[dict[str, str]] = []
    pos = 0
    for match in TOOL_CALL_RE.finditer(cleaned):
        prefix = cleaned[pos : match.start()].strip()
        if prefix:
            messages.append({"role": "assistant", "content": prefix})
        messages.append({"role": "tool_call", "content": normalize_tool_call(match.group(1).strip())})
        pos = match.end()
    suffix = cleaned[pos:].strip()
    if suffix:
        messages.append({"role": "assistant", "content": suffix})
    return messages


def convert_trajectory(trajectory: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for idx, message in enumerate(trajectory):
        role = message.get("role")
        if idx == 0 and role == "system":
            continue
        text = str(message.get("content", "") or "").strip()
        if not text:
            continue
        if role == "assistant":
            messages.extend(split_assistant_message(text))
            continue
        if role == "user":
            match = TOOL_RESPONSE_RE.match(text)
            if match:
                content = match.group(1).strip()
                if content:
                    messages.append({"role": "tool_response", "content": content})
            else:
                messages.append({"role": "user", "content": text})
    return messages


def convert_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any] | None:
    raw_trajectory = row.get("trajectory") or []
    if not isinstance(raw_trajectory, list) or not raw_trajectory:
        return None
    trajectory = [item for item in raw_trajectory if isinstance(item, Mapping)]
    if not trajectory:
        return None

    system_text = str(trajectory[0].get("content", "") or "") if trajectory[0].get("role") == "system" else ""
    messages = convert_trajectory(trajectory)
    if not messages:
        return None

    return {
        "messages": messages,
        "tools": json.dumps(parse_tools(system_text), ensure_ascii=False, separators=(",", ":")),
        "system": normalize_system(system_text),
        "id": f"OpenSeeker:{index}",
        "source": "OpenSeeker",
        "question": str(row.get("question", "") or ""),
        "answer": str(row.get("answer", "") or ""),
        "metadata": {
            "num_tool_calls": sum(1 for message in messages if message["role"] == "tool_call"),
            "trajectory_correctness": row.get("trajectory correctness"),
            "removed_repeated_search_turns": 0,
        },
    }


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    input_path = Path(path).expanduser()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {input_path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_records: int = 0,
) -> ConversionStats:
    stats = ConversionStats()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_jsonl(input_path)):
            stats.total += 1
            item = convert_row(row, index=index)
            if item is None:
                stats.skipped += 1
                continue
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats.written += 1
            if max_records and stats.written >= max_records:
                break
    return stats
