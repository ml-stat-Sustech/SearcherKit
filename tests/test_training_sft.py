from __future__ import annotations

import json
from pathlib import Path

from searchagent.plugins.conversion.config import load_config
from searchagent.plugins.conversion.convert import convert_file, convert_row
from searchagent.training.sft.config import (
    SFTBackendConfig,
    SFTTrainingConfig,
    SFTTrainingDataConfig,
    load_training_config,
)
from searchagent.training.sft.train import build_swift_command


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sft"


def test_openseeker_converter_matches_ms_swift_golden_fixture() -> None:
    row = json.loads((FIXTURE_DIR / "openseeker.jsonl").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURE_DIR / "openseeker_ms_swift.jsonl").read_text(encoding="utf-8"))

    actual = convert_row(row, index=0)

    assert actual == expected


def test_openseeker_converter_writes_ms_swift_golden_fixture(tmp_path: Path) -> None:
    output_path = tmp_path / "ms_swift.jsonl"

    stats = convert_file(FIXTURE_DIR / "openseeker.jsonl", output_path)

    assert stats.to_dict() == {"skipped": 0, "total": 1, "written": 1}
    actual = json.loads(output_path.read_text(encoding="utf-8"))
    expected = json.loads((FIXTURE_DIR / "openseeker_ms_swift.jsonl").read_text(encoding="utf-8"))
    assert actual == expected


def test_sft_conversion_config_loads_paths() -> None:
    cfg = load_config()

    assert cfg.input_path
    assert cfg.output_path
    assert cfg.max_records == 0


def test_build_swift_command_uses_explicit_cli_args() -> None:
    config = SFTTrainingConfig(
        data=SFTTrainingDataConfig(train_path="data/train.jsonl"),
        backend=SFTBackendConfig(
            swift_bin="/bin/swift",
            model="Qwen/Qwen3",
            rope_scaling={"type": "yarn", "factor": 2},
            extra_args={"warmup_ratio": 0.05},
        ),
    )

    command = build_swift_command(config, config.data.train_path)

    assert command[:2] == ["/bin/swift", "sft"]
    assert "--dataset" in command
    assert "data/train.jsonl" in command
    assert "--rope_scaling" in command
    assert '{"type":"yarn","factor":2}' in command
    assert "--warmup_ratio" in command


def test_split_sft_training_config_loads_train_path() -> None:
    cfg = load_training_config("scripts/sft/sft_training.yaml")

    assert cfg.data.train_path
    assert cfg.backend.command == "sft"
