from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from competitions.gan_getting_started.scripts.build_submission import (
    collect_source_images,
    write_submission_zip,
)


def _touch_jpg(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_collect_source_images(tmp_path: Path):
    _touch_jpg(tmp_path / "photo_jpg" / "a.jpg", b"a")
    _touch_jpg(tmp_path / "photo_jpg" / "b.jpg", b"b")
    images = collect_source_images(tmp_path, "photo_jpg")
    assert [path.name for path in images] == ["a.jpg", "b.jpg"]


def test_write_submission_zip_repeats_if_needed(tmp_path: Path):
    src_a = tmp_path / "src" / "a.jpg"
    src_b = tmp_path / "src" / "b.jpg"
    _touch_jpg(src_a, b"a")
    _touch_jpg(src_b, b"b")

    out_zip = tmp_path / "images.zip"
    write_submission_zip([src_a, src_b], num_images=5, output_zip=out_zip)

    with ZipFile(out_zip) as archive:
        names = sorted(archive.namelist())
    assert names == ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg", "00004.jpg"]

