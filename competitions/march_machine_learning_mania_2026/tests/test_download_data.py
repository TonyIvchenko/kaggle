from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from competitions.march_machine_learning_mania_2026.scripts.download_data import (
    extract_archives,
    kaggle_credentials_configured,
    kaggle_credentials_path,
)


def test_kaggle_credentials_path_uses_config_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert kaggle_credentials_path() == tmp_path / "kaggle.json"


def test_kaggle_credentials_configured_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert kaggle_credentials_configured() is True


def test_extract_archives_unzips_files(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/example.csv", "a,b\n1,2\n")

    extracted = extract_archives(tmp_path)

    assert tmp_path.joinpath("nested", "example.csv").exists()
    assert any(path.name == "example.csv" for path in extracted)
