"""Baseline training wrapper for GAN Getting Started."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_SUBMISSION_ZIP = COMPETITION_ROOT / "submissions" / "images.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a baseline submission archive for GAN Getting Started.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with raw competition files.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="photo_jpg",
        help="Source image directory under raw dir.",
    )
    parser.add_argument("--num-images", type=int, default=7000, help="Number of images to include in archive.")
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_SUBMISSION_ZIP, help="Output zip path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = COMPETITION_ROOT / "scripts" / "build_submission.py"
    cmd = [
        sys.executable,
        str(script),
        "--raw-dir",
        str(args.raw_dir),
        "--source-dir",
        args.source_dir,
        "--num-images",
        str(args.num_images),
        "--output-zip",
        str(args.output_zip),
    ]
    subprocess.run(cmd, check=True)
    print("Baseline submission build complete.")
    print(f"Submission path: {args.output_zip}")


if __name__ == "__main__":
    main()

