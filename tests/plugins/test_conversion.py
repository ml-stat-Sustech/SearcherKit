import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from searchagent.plugins.conversion.main import load_config
from searchagent.plugins.conversion.convert import convert_file


FIXTURE_DIR = Path("tests/fixtures/datasets")
OPENSEEKER_PATH = FIXTURE_DIR / "openseeker.jsonl"
MS_SWIFT_PATH = FIXTURE_DIR / "openseeker_ms_swift.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _convert_direct(output_path: Path, tmp_path: Path) -> dict[str, int]:
    return convert_file(OPENSEEKER_PATH, output_path).to_dict()


def _convert_from_config(output_path: Path, tmp_path: Path) -> dict[str, int]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "openseeker_ms_swift.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_path: {OPENSEEKER_PATH}",
                f"output_path: {output_path}",
                "max_records: 0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path=config_dir)
    return convert_file(
        config.input_path,
        config.output_path,
        max_records=config.max_records,
    ).to_dict()


@pytest.mark.parametrize("convert", [_convert_direct, _convert_from_config])
def test_openseeker_conversion_matches_ms_swift_fixture(
    tmp_path: Path,
    convert: Callable[[Path, Path], dict[str, int]],
) -> None:
    output_path = tmp_path / "openseeker_ms_swift.jsonl"

    stats = convert(output_path, tmp_path)

    actual = _read_jsonl(output_path)
    expected = _read_jsonl(MS_SWIFT_PATH)
    assert stats == {"total": 10, "written": 10, "skipped": 0}
    assert actual == expected

    first = actual[0]
    assert first["id"] == "OpenSeeker:0"
    assert first["source"] == "OpenSeeker"
    assert first["answer"] == "Carl von Ossietzky"
    assert first["metadata"]["num_tool_calls"] == 4
    assert {tool["function"]["name"] for tool in json.loads(first["tools"])} == {"search", "visit"}
    assert [message["role"] for message in first["messages"]].count("tool_call") == 4
    assert [message["role"] for message in first["messages"]].count("tool_response") == 4

    second = actual[1]
    assert second["id"] == "OpenSeeker:1"
    assert second["answer"] == "Defense Distinguished Service Medal"
    assert second["metadata"]["num_tool_calls"] == 4
    assert {tool["function"]["name"] for tool in json.loads(second["tools"])} == {"search", "visit"}
    assert [message["role"] for message in second["messages"]].count("tool_call") == 4
    assert second["messages"][-1]["content"].endswith(
        "The decoration described is the **Defense Distinguished Service Medal** of the United States.\n</answer>"
    )
