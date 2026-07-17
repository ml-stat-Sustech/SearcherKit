import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from searcherkit.plugins.conversion.main import main as conversion_main


FIXTURE_DIR = Path("tests/fixtures/datasets")
OPENSEEKER_PATH = FIXTURE_DIR / "openseeker.jsonl"
MS_SWIFT_PATH = FIXTURE_DIR / "openseeker_ms_swift.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_conversion_cli(args: list[str]) -> dict[str, int]:
    stdout = StringIO()
    with redirect_stdout(stdout):
        status = conversion_main(args, prog="searcher plugins convert")
    assert status == 0
    return json.loads(stdout.getvalue())


def _convert_from_cli(output_path: Path) -> dict[str, int]:
    return _run_conversion_cli(
        [
            str(OPENSEEKER_PATH),
            str(output_path),
        ]
    )


def test_openseeker_conversion_matches_ms_swift_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "openseeker_ms_swift.jsonl"

    stats = _convert_from_cli(output_path)

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


def test_conversion_cli_max_records(tmp_path: Path) -> None:
    output_path = tmp_path / "limited.jsonl"

    stats = _run_conversion_cli(
        [
            str(OPENSEEKER_PATH),
            str(output_path),
            "--max-records",
            "1",
        ]
    )

    actual = _read_jsonl(output_path)
    assert stats == {"total": 1, "written": 1, "skipped": 0}
    assert len(actual) == 1
    assert actual[0]["id"] == "OpenSeeker:0"
