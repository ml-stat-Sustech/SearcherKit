from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from types import NoneType, UnionType
from typing import Any, Literal, Union, Mapping as TypingMapping, get_args, get_origin, get_type_hints

from searcherkit.common.log import get_logger


logger = get_logger(__name__)


def schema_from_signature(fn: Any) -> Mapping[str, Any] | None:
    """Build a JSON Schema object from a keyword-callable signature."""
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError, AttributeError) as exc:
        logger.warning(
            "Cannot build JSON schema for %s because type hints could not be resolved: %s",
            _callable_name(fn),
            exc,
        )
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        # Only keyword-callable, fixed-shape signatures can be translated into
        # model-visible JSON object properties without guessing.
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            logger.warning(
                "Cannot build JSON schema for %s because parameter %r has unsupported kind %s",
                _callable_name(fn),
                parameter.name,
                parameter.kind.description,
            )
            return None
        annotation = type_hints.get(parameter.name, parameter.annotation)
        properties[parameter.name] = _schema_from_annotation(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _callable_name(fn: Any) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _schema_from_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if isinstance(annotation, str):
        return {}
    if annotation is None or annotation is NoneType:
        return {"type": "null"}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        return _union_schema(args)
    if origin is Literal:
        return _literal_schema(args)
    if origin in (list, tuple, set, frozenset):
        item_schema = _schema_from_annotation(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Mapping, TypingMapping):
        value_schema = _schema_from_annotation(args[1]) if len(args) >= 2 else {}
        return {"type": "object", "additionalProperties": value_schema}
    if origin is not None:
        return _schema_from_annotation(origin)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [item.value for item in annotation]
        schema = _literal_schema(tuple(values))
        return schema or {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list:
        return {"type": "array", "items": {}}
    if annotation is dict or annotation is Mapping or annotation is TypingMapping:
        return {"type": "object"}

    return {}


def _union_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    schemas = [_schema_from_annotation(arg) for arg in args]
    types: list[str] = []
    merged: list[dict[str, Any]] = []
    for schema in schemas:
        schema_type = schema.get("type")
        if isinstance(schema_type, str) and set(schema) == {"type"}:
            if schema_type not in types:
                types.append(schema_type)
            continue
        merged.append(schema)
    if not merged and types:
        return {"type": types[0] if len(types) == 1 else types}
    for schema_type in types:
        merged.append({"type": schema_type})
    if len(merged) == 1:
        return merged[0]
    return {"anyOf": merged}


def _literal_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    if not args:
        return {}
    schema: dict[str, Any] = {"enum": list(args)}
    literal_types = {
        _json_type_for_literal_value(value)
        for value in args
    }
    if None in literal_types:
        literal_types.discard(None)
    if literal_types:
        schema["type"] = (
            next(iter(literal_types))
            if len(literal_types) == 1
            else sorted(literal_types)
        )
    return schema


def _json_type_for_literal_value(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return None
