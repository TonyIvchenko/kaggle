from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from competitions.gan_getting_started.scripts.download_data import (
    build_download_commands,
    build_download_error_message,
    extract_archives,
    parse_file_list,
)


def test_parse_file_list_deduplicates():
    files = parse_file_list("a.jpg, b.jpg, a.jpg, ,c.jpg")
    assert files == ["a.jpg", "b.jpg", "c.jpg"]


def test_build_download_commands_selected_files(tmp_path: Path):
    commands = build_download_commands(
        competition="gan-getting-started",
        raw_dir=tmp_path,
        kaggle_exe="/usr/bin/kaggle",
        files=["photo_jpg/abc.jpg", "monet_jpg/xyz.jpg"],
        force=False,
        all_files=False,
    )
    assert commands == [
        [
            "/usr/bin/kaggle",
            "competitions",
            "download",
            "-c",
            "gan-getting-started",
            "-p",
            str(tmp_path),
            "-f",
            "photo_jpg/abc.jpg",
        ],
        [
            "/usr/bin/kaggle",
            "competitions",
            "download",
            "-c",
            "gan-getting-started",
            "-p",
            str(tmp_path),
            "-f",
            "monet_jpg/xyz.jpg",
        ],
    ]


def test_build_download_commands_all_files(tmp_path: Path):
    commands = build_download_commands(
        competition="gan-getting-started",
        raw_dir=tmp_path,
        kaggle_exe="/usr/bin/kaggle",
        files=[],
        force=True,
        all_files=True,
    )
    assert commands == [
        ["/usr/bin/kaggle", "competitions", "download", "-c", "gan-getting-started", "-p", str(tmp_path), "--force"]
    ]


def test_extract_archives(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("photo_jpg/test.jpg", b"abc")
    extracted = extract_archives(tmp_path)
    assert tmp_path.joinpath("photo_jpg", "test.jpg").exists()
    assert any(path.name == "test.jpg" for path in extracted)


def test_download_error_message_401():
    message = build_download_error_message("gan-getting-started", "401 Client Error: Unauthorized")
    assert "401 Unauthorized" in message


def test_download_error_message_403():
    message = build_download_error_message("gan-getting-started", "403 Client Error: Forbidden")
    assert "403 Forbidden" in message

