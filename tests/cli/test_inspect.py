from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from searcherkit.cli import main as cli_main


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cli"


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main.main(["inspect", "--config-path", str(FIXTURE_DIR / "test_config.yaml")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config validation passed" in captured.out


def test_inspect_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main.main(
        [
            "inspect",
            "--config-path",
            str(FIXTURE_DIR / "test_config_invalid.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Config validation found 1 error(s)" in captured.out
    assert "unexpected_root" in captured.out


def test_config_path_must_be_file() -> None:
    with pytest.raises(FileNotFoundError, match="config file does not exist"):
        cli_main.main(["inspect", "--config-path", str(FIXTURE_DIR)])


def test_cli_works_without_uvloop(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_uvloop(name: str, *args: object, **kwargs: object) -> object:
        if name == "uvloop":
            raise ImportError("uvloop unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_uvloop)
    with pytest.warns(RuntimeWarning, match="uvloop is not installed"):
        with pytest.raises(SystemExit, match="0"):
            cli_main.main(["--help"])
