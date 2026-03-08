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


def kaggle_access_token_path() -> Path:
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return config_dir / "access_token"


def kaggle_credentials_configured() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if kaggle_access_token_path().exists():
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
        "Kaggle credentials were not found. Configure one of: "
        "~/.kaggle/access_token, ~/.kaggle/kaggle.json, KAGGLE_API_TOKEN, or KAGGLE_USERNAME/KAGGLE_KEY."
    )


def extract_archives(raw_dir: Path) -> list[Path]:
    extracted_files: list[Path] = []
    for archive_path in sorted(raw_dir.glob("*.zip")):
        with ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)
            extracted_files.extend(raw_dir / name for name in archive.namelist())
    return extracted_files


def build_download_error_message(competition: str, output: str) -> str:
    if "401" in output and "Unauthorized" in output:
        return (
            "Kaggle CLI returned 401 Unauthorized while downloading competition data.\n"
            "Most likely causes:\n"
            "1. Your Kaggle credentials are invalid or expired.\n"
            "2. You generated the wrong token type and need to regenerate it from Kaggle settings.\n"
            "3. The competition rules have not been accepted in the browser yet.\n\n"
            "Debug this in order:\n"
            "  - Run: kaggle competitions list -s titanic\n"
            "    If that also returns 401, regenerate credentials in https://www.kaggle.com/settings\n"
            "  - If list works, open https://www.kaggle.com/competitions/"
            f"{competition} and accept the competition rules.\n\n"
            f"Original CLI output:\n{output}"
        )
    if "403" in output and "Forbidden" in output:
        return (
            "Kaggle CLI returned 403 Forbidden while downloading competition data.\n"
            "Your credentials are probably valid, but this competition still is not downloadable for your account.\n\n"
            "For March Machine Learning Mania 2026, Kaggle marks the competition as requiring both rules acceptance "
            "and identity verification.\n"
            "Check these in order:\n"
            "  - Open https://www.kaggle.com/competitions/"
            f"{competition} in the browser.\n"
            "  - Click Join / Accept Rules if Kaggle shows that prompt.\n"
            "  - Complete any requested account or identity verification on Kaggle.\n"
            "  - Retry the CLI download after that.\n\n"
            "The outdated-version warning is secondary; it is worth upgrading the kaggle package, but it is not the "
            "main cause of a 403.\n\n"
            f"Original CLI output:\n{output}"
        )
    return f"Kaggle download failed:\n{output}"


def download_competition_data(competition: str, raw_dir: Path, force: bool = False) -> None:
    kaggle_exe = ensure_kaggle_cli_available()
    ensure_kaggle_credentials()
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [kaggle_exe, "competitions", "download", "-c", competition, "-p", str(raw_dir)]
    if force:
        command.append("--force")

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    combined = "\n".join(part for part in [stdout, stderr] if part)
    raise RuntimeError(build_download_error_message(competition=competition, output=combined))


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
