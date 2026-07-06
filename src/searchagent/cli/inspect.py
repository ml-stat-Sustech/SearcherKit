from __future__ import annotations

import argparse
import types
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Sequence, get_args, get_origin

from omegaconf import DictConfig

from searchagent.cli.config import compose_config
from searchagent.runtime.runner import RunConfig


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin is types.UnionType or origin is getattr(__import__("typing"), "Union", None):
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return tp, False


def _resolve_field_type(target_type: type, field: Field) -> Any:
    tp = field.type
    if isinstance(tp, str):
        import typing

        hints = typing.get_type_hints(target_type)
        tp = hints.get(field.name, tp)
    return tp


def _get_dataclass_fields(target_type: type) -> dict[str, Field]:
    result: dict[str, Field] = {}
    for f in fields(target_type):
        result[f.name] = f
    return result


def validate_config(
    cfg: DictConfig,
    target_type: type,
    path: str = "",
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    if not is_dataclass(target_type):
        return issues

    if not hasattr(cfg, "keys"):
        return issues

    expected_fields = _get_dataclass_fields(target_type)
    cfg_keys = set(cfg.keys())
    expected_keys = set(expected_fields.keys())

    for k in sorted(cfg_keys - expected_keys):
        full_path = f"{path}.{k}" if path else k
        issues.append(
            {
                "path": full_path,
                "level": "ERROR",
                "detail": f"unexpected field in config (not in {target_type.__name__})",
            }
        )

    for k in sorted(expected_keys - cfg_keys):
        f = expected_fields[k]
        if f.default is not MISSING or f.default_factory is not MISSING:
            continue
        full_path = f"{path}.{k}" if path else k
        issues.append(
            {
                "path": full_path,
                "level": "ERROR",
                "detail": f"required field missing (expected by {target_type.__name__})",
            }
        )

    for k in sorted(cfg_keys & expected_keys):
        f = expected_fields[k]
        full_path = f"{path}.{k}" if path else k
        field_type = _resolve_field_type(target_type, f)

        core_type, is_optional = _unwrap_optional(field_type)

        value = cfg[k]
        if value is None and is_optional:
            continue

        origin = get_origin(core_type)
        if origin is list:
            args = get_args(core_type)
            elem_type = args[0] if args else None
            if (
                elem_type
                and is_dataclass(elem_type)
                and hasattr(value, "__iter__")
                and not isinstance(value, (str, bytes))
            ):
                for idx, item in enumerate(value):
                    issues.extend(
                        validate_config(item, elem_type, f"{full_path}[{idx}]")
                    )
        elif is_dataclass(core_type):
            if hasattr(value, "keys"):
                issues.extend(validate_config(value, core_type, full_path))

    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchagent inspect",
        description="Recursively validate config fields against structured types.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Directory containing the Hydra config. Defaults to the packaged searchagent config directory.",
    )
    parser.add_argument("--config-name", default="config")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
    )
    issues = validate_config(cfg, RunConfig)
    if not issues:
        print("Config validation passed. No issues found.")
        return 0

    errors = [i for i in issues if i["level"] == "ERROR"]
    warnings = [i for i in issues if i["level"] == "WARNING"]
    print(f"Config validation found {len(errors)} error(s), {len(warnings)} warning(s):\n")
    for issue in issues:
        print(f"  [{issue['level']:<7}] {issue['path']}: {issue['detail']}")
    return 1 if errors else 0
