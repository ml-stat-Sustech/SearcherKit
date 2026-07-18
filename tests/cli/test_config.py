from __future__ import annotations

from pathlib import Path

import pytest

from searcherkit.cli.config import resolve_config_file


def test_resolve_config_file_defaults_to_packaged_config(tmp_path: Path) -> None:
    packaged = tmp_path / "package-config"
    expected = packaged / "config.yaml"
    packaged.mkdir()
    expected.touch()

    assert resolve_config_file(None, default=packaged) == expected


@pytest.mark.parametrize("value", ["custom.yaml", "custom"])
def test_resolve_config_file_prefers_local_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    packaged = tmp_path / "package-config"
    packaged.mkdir()
    (packaged / "custom.yaml").touch()
    local = tmp_path / "custom.yaml"
    local.touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_config_file(value, default=packaged) == local


def test_resolve_config_file_falls_back_to_packaged_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged = tmp_path / "package-config"
    expected = packaged / "examples" / "custom.yaml"
    expected.parent.mkdir(parents=True)
    expected.touch()
    monkeypatch.chdir(tmp_path)

    assert resolve_config_file("examples/custom", default=packaged) == expected


def test_resolve_config_file_rejects_missing_config(tmp_path: Path) -> None:
    packaged = tmp_path / "package-config"
    packaged.mkdir()

    with pytest.raises(FileNotFoundError, match="config file does not exist"):
        resolve_config_file("missing", default=packaged)
