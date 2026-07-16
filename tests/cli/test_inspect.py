from __future__ import annotations

from pathlib import Path

import pytest

from searcherkit.cli import main as cli_main


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cli"


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main.main(
        ["inspect", "--config-path", str(FIXTURE_DIR), "--config-name", "test_config"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config validation passed" in captured.out


def test_inspect_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main.main(
        [
            "inspect",
            "--config-path",
            str(FIXTURE_DIR),
            "--config-name",
            "test_config_invalid",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config validation found 1 error(s)" in captured.out
    assert "unexpected_root" in captured.out


def test_config_path_must_be_directory() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist or is not a directory"):
        cli_main.main(["inspect", "--config-path", str(FIXTURE_DIR / "test_config.yaml")])
