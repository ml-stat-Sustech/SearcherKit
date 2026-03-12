from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union, Literal

from pydantic import BaseModel, ConfigDict, create_model


class JsonSchemaConversionError(ValueError):
    """Raised when JSON Schema cannot be converted to a Pydantic model."""


def json_schema_to_pydantic(
    schema: Mapping[str, Any],
    *,
    model_name: str = "GeneratedModel",
) -> type[BaseModel]:
    """Convert a JSON Schema (subset) into a Pydantic model class.

    Supported (best-effort):
    - type: object/string/number/integer/boolean/array/null
    - properties + required
    - enum
    - items (array)
    - additionalProperties (bool)

    Notes:
    - advanced JSON Schema keywords are not supported (oneOf/anyOf/allOf/etc.).
    - additionalProperties schema is ignored if properties exist.
    """
    if not isinstance(schema, Mapping):
        raise JsonSchemaConversionError("schema must be a Mapping")

    schema_type = schema.get("type")
    if schema_type is None:
        raise JsonSchemaConversionError("schema.type is required")

    if schema_type != "object":
        # Wrap non-object schema as a root model.
        root_type = _schema_to_type(schema, fallback_name=f"{model_name}Root")
        return create_model(
            model_name,
            __base__=BaseModel,
            __config__=ConfigDict(extra="forbid"),
            __root__=(root_type, ...),
        )

    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        raise JsonSchemaConversionError("schema.properties must be a Mapping")

    required = schema.get("required") or []
    if not isinstance(required, list):
        raise JsonSchemaConversionError("schema.required must be a list")
    required_set = {str(name) for name in required}

    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop_schema in properties.items():
        if not isinstance(prop_schema, Mapping):
            raise JsonSchemaConversionError(
                f"schema.properties[{name!r}] must be a Mapping"
            )
        field_type = _schema_to_type(prop_schema, fallback_name=f"{model_name}_{name}")
        if name in required_set:
            fields[str(name)] = (field_type, ...)
        else:
            fields[str(name)] = (Optional[field_type], None)

    additional = schema.get("additionalProperties", True)
    if additional is False:
        config = ConfigDict(extra="forbid")
    else:
        config = ConfigDict(extra="allow")

    return create_model(
        model_name,
        __base__=BaseModel,
        __config__=config,
        **fields,
    )


def _schema_to_type(schema: Mapping[str, Any], *, fallback_name: str) -> Any:
    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list) or not enum_values:
            raise JsonSchemaConversionError("enum must be a non-empty list")
        return Literal[tuple(enum_values)]  # type: ignore[misc]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        types = [_schema_to_type({"type": t}, fallback_name=fallback_name) for t in schema_type]
        return Union[tuple(types)]  # type: ignore[misc]

    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "null":
        return type(None)
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            item_type = _schema_to_type(items, fallback_name=fallback_name + "Item")
        else:
            item_type = Any
        return list[item_type]  # type: ignore[misc]
    if schema_type == "object":
        return json_schema_to_pydantic(schema, model_name=fallback_name)

    raise JsonSchemaConversionError(f"Unsupported schema type: {schema_type!r}")
