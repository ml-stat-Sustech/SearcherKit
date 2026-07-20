from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

import pytest

from searcherkit.common.json_schema import schema_from_signature


class Status(Enum):
    OPEN = "open"
    CLOSED = "closed"


def test_schema_from_signature_builds_object_schema() -> None:
    async def run(*, query: str, top_k: int = 5) -> str:
        return query

    schema = schema_from_signature(run)

    assert schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "additionalProperties": False,
        "required": ["query"],
    }


def test_schema_from_signature_supports_optional_and_literal_types() -> None:
    async def run(*, goal: str | None = None, mode: Literal["fast", "deep"] = "fast") -> str:
        return goal or mode

    schema = schema_from_signature(run)

    assert schema is not None
    assert schema["properties"]["goal"]["type"] == ["string", "null"]
    assert schema["properties"]["mode"] == {
        "enum": ["fast", "deep"],
        "type": "string",
    }
    assert "required" not in schema


def test_schema_from_signature_supports_container_types() -> None:
    async def run(*, tags: list[str], weights: dict[str, float], metadata: Mapping[str, Any]) -> str:
        return ",".join(tags)

    schema = schema_from_signature(run)

    assert schema is not None
    assert schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert schema["properties"]["weights"] == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }
    assert schema["properties"]["metadata"] == {
        "type": "object",
        "additionalProperties": {},
    }


def test_schema_from_signature_supports_enum_types() -> None:
    async def run(*, status: Status) -> str:
        return status.value

    schema = schema_from_signature(run)

    assert schema is not None
    assert schema["properties"]["status"] == {
        "enum": ["open", "closed"],
        "type": "string",
    }


def test_schema_from_signature_returns_none_for_non_keyword_signatures(
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run(query: str, **kwargs: Any) -> str:
        return query

    monkeypatch.setattr(logging.getLogger("searcherkit"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="searcherkit.common.json_schema")

    assert schema_from_signature(run) is None
    assert "unsupported kind variadic keyword" in caplog.text


def test_schema_from_signature_returns_none_when_type_hints_cannot_resolve(
    caplog: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run(*, item: "MissingType") -> str:  # noqa: F821
        return str(item)

    monkeypatch.setattr(logging.getLogger("searcherkit"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="searcherkit.common.json_schema")

    assert schema_from_signature(run) is None
    assert "type hints could not be resolved" in caplog.text
    assert "MissingType" in caplog.text
