from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from competitions.llm_classification_finetuning.scripts.download_data import (
    build_download_commands,
    build_download_error_message,
    extract_archives,
    parse_file_list,
)


def test_parse_file_list_deduplicates_and_ignores_whitespace():
    files = parse_file_list("train.csv, test.csv, train.csv, ,sample_submission.csv")
    assert files == ["train.csv", "test.csv", "sample_submission.csv"]


def test_build_download_commands_uses_selected_files(tmp_path: Path):
    commands = build_download_commands(
        competition="demo-comp",
        raw_dir=tmp_path,
        kaggle_exe="/usr/bin/kaggle",
        files=["train.csv", "test.csv"],
        force=False,
        all_files=False,
    )
    assert commands == [
        ["/usr/bin/kaggle", "competitions", "download", "-c", "demo-comp", "-p", str(tmp_path), "-f", "train.csv"],
        ["/usr/bin/kaggle", "competitions", "download", "-c", "demo-comp", "-p", str(tmp_path), "-f", "test.csv"],
    ]


def test_build_download_commands_all_files_mode(tmp_path: Path):
    commands = build_download_commands(
        competition="demo-comp",
        raw_dir=tmp_path,
        kaggle_exe="/usr/bin/kaggle",
        files=["train.csv"],
        force=True,
        all_files=True,
    )
    assert commands == [
        ["/usr/bin/kaggle", "competitions", "download", "-c", "demo-comp", "-p", str(tmp_path), "--force"]
    ]


def test_extract_archives_unzips_files(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/example.csv", "a,b\n1,2\n")

    extracted = extract_archives(tmp_path)

    assert tmp_path.joinpath("nested", "example.csv").exists()
    assert any(path.name == "example.csv" for path in extracted)


def test_build_download_error_message_401_mentions_credentials():
    message = build_download_error_message("demo-comp", "401 Client Error: Unauthorized")
    assert "401 Unauthorized" in message
    assert "kaggle competitions list -s titanic" in message


def test_build_download_error_message_403_mentions_verification():
    message = build_download_error_message("demo-comp", "403 Client Error: Forbidden")
    assert "403 Forbidden" in message
    assert "verification" in message

