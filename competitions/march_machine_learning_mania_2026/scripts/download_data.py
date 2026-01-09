"""Download March Machine Learning Mania 2026 files with the Kaggle CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile


COMPETITION_SLUG = "march-machine-learning-mania-2026"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download March Mania 2026 competition data.")
    parser.add_argument("--competition", default=COMPETITION_SLUG, help="Kaggle competition slug.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where downloaded and extracted files are stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to the Kaggle CLI download command.",
    )
    return parser.parse_args()


def kaggle_credentials_path() -> Path:
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return config_dir / "kaggle.json"


def kaggle_credentials_configured() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return kaggle_credentials_path().exists()


def ensure_kaggle_cli_available() -> str:
    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise RuntimeError(
            "Kaggle CLI was not found on PATH. Activate the conda env or install the `kaggle` package first."
        )
    return kaggle_exe


def ensure_kaggle_credentials() -> None:
    if kaggle_credentials_configured():
        return
    raise RuntimeError(
        "Kaggle credentials were not found. Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY."
    )


def extract_archives(raw_dir: Path) -> list[Path]:
    extracted_files: list[Path] = []
    for archive_path in sorted(raw_dir.glob("*.zip")):
        with ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)
            extracted_files.extend(raw_dir / name for name in archive.namelist())
    return extracted_files


def download_competition_data(competition: str, raw_dir: Path, force: bool = False) -> None:
    kaggle_exe = ensure_kaggle_cli_available()
    ensure_kaggle_credentials()
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [kaggle_exe, "competitions", "download", "-c", competition, "-p", str(raw_dir)]
    if force:
        command.append("--force")

    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    download_competition_data(competition=args.competition, raw_dir=args.raw_dir, force=args.force)
    extracted = extract_archives(args.raw_dir)
    csv_files = sorted(path for path in args.raw_dir.rglob("*.csv"))

    print(f"Downloaded competition files into: {args.raw_dir}")
    if extracted:
        print(f"Extracted {len(extracted)} files from zip archives.")
    print("CSV files discovered:")
    for path in csv_files:
        print(f" - {path.relative_to(args.raw_dir)}")


if __name__ == "__main__":
    main()
