from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from competitions.march_machine_learning_mania_2026.scripts.download_data import (
    build_download_error_message,
    extract_archives,
    kaggle_access_token_path,
    kaggle_credentials_configured,
    kaggle_credentials_path,
)


def test_kaggle_credentials_path_uses_config_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert kaggle_credentials_path() == tmp_path / "kaggle.json"
    assert kaggle_access_token_path() == tmp_path / "access_token"


def test_kaggle_credentials_configured_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert kaggle_credentials_configured() is True


def test_kaggle_credentials_configured_from_api_token_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    assert kaggle_credentials_configured() is True


def test_extract_archives_unzips_files(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/example.csv", "a,b\n1,2\n")

    extracted = extract_archives(tmp_path)

    assert tmp_path.joinpath("nested", "example.csv").exists()
    assert any(path.name == "example.csv" for path in extracted)


def test_build_download_error_message_401_mentions_credentials():
    message = build_download_error_message("demo-comp", "401 Client Error: Unauthorized")
    assert "credentials are invalid or expired" in message
    assert "kaggle competitions list -s titanic" in message


def test_build_download_error_message_403_mentions_rules_and_verification():
    message = build_download_error_message("demo-comp", "403 Client Error: Forbidden")
    assert "Accept Rules" in message
    assert "identity verification" in message
