"""Build a valid baseline submission archive for GAN Getting Started."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_SUBMISSION_DIR = COMPETITION_ROOT / "submissions"
DEFAULT_OUTPUT_ZIP = DEFAULT_SUBMISSION_DIR / "images.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline `images.zip` submission.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with raw competition files.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="photo_jpg",
        help="Source image directory under raw dir (e.g. photo_jpg or monet_jpg).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=7000,
        help="Number of images to include in submission archive.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help="Path to output zip file (should be images.zip for submission).",
    )
    return parser.parse_args()


def collect_source_images(raw_dir: Path, source_dir: str) -> list[Path]:
    source_root = raw_dir / source_dir
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")
    images = sorted(path for path in source_root.rglob("*.jpg") if path.is_file())
    if not images:
        raise FileNotFoundError(f"No .jpg images found under {source_root}")
    return images


def write_submission_zip(source_images: list[Path], num_images: int, output_zip: Path) -> None:
    if num_images <= 0:
        raise ValueError("num_images must be positive")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    with ZipFile(output_zip, mode="w", compression=ZIP_DEFLATED) as archive:
        for idx in range(num_images):
            src = source_images[idx % len(source_images)]
            arcname = f"{idx:05d}.jpg"
            archive.write(src, arcname=arcname)


def main() -> None:
    args = parse_args()
    images = collect_source_images(args.raw_dir, args.source_dir)
    write_submission_zip(images, args.num_images, args.output_zip)
    print(f"Created submission zip: {args.output_zip}")
    print(f"Source directory: {args.source_dir}")
    print(f"Source images available: {len(images)}")
    print(f"Images written: {args.num_images}")


if __name__ == "__main__":
    main()

