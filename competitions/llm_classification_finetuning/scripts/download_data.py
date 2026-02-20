"""Download LLM Classification Finetuning files with the Kaggle CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile


COMPETITION_SLUG = "llm-classification-finetuning"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LLM Classification Finetuning data.")
    parser.add_argument("--competition", default=COMPETITION_SLUG, help="Kaggle competition slug.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory where downloaded and extracted files are stored.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="",
        help="Optional comma-separated file list (e.g. train.csv,test.csv,sample_submission.csv).",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all competition files instead of the default minimal set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to the Kaggle CLI download command.",
    )
    return parser.parse_args()


def parse_file_list(value: str) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for part in value.split(","):
        candidate = part.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        files.append(candidate)
    return files


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


def build_download_commands(
    competition: str,
    raw_dir: Path,
    kaggle_exe: str,
    files: list[str],
    force: bool,
    all_files: bool,
) -> list[list[str]]:
    base = [kaggle_exe, "competitions", "download", "-c", competition, "-p", str(raw_dir)]
    if force:
        base = [*base, "--force"]

    if all_files or not files:
        return [base]
    return [[*base, "-f", filename] for filename in files]


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
            "Check credentials in ~/.kaggle and accept competition rules in browser first.\n\n"
            "Debug:\n"
            "  - Run: kaggle competitions list -s titanic\n"
            f"  - Open: https://www.kaggle.com/competitions/{competition}\n\n"
            f"Original CLI output:\n{output}"
        )
    if "403" in output and "Forbidden" in output:
        return (
            "Kaggle CLI returned 403 Forbidden while downloading competition data.\n"
            "Accept rules / complete verification in Kaggle settings and retry.\n\n"
            f"Original CLI output:\n{output}"
        )
    return f"Kaggle download failed:\n{output}"


def download_competition_data(
    competition: str,
    raw_dir: Path,
    files: list[str] | None = None,
    force: bool = False,
    all_files: bool = False,
) -> None:
    kaggle_exe = ensure_kaggle_cli_available()
    ensure_kaggle_credentials()
    raw_dir.mkdir(parents=True, exist_ok=True)

    requested_files = files or []
    commands = build_download_commands(
        competition=competition,
        raw_dir=raw_dir,
        kaggle_exe=kaggle_exe,
        files=requested_files,
        force=force,
        all_files=all_files,
    )

    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            continue
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        raise RuntimeError(build_download_error_message(competition=competition, output=combined))


def main() -> None:
    args = parse_args()
    requested_files = parse_file_list(args.files)
    if not args.all_files and not requested_files:
        requested_files = list(DEFAULT_FILES)

    download_competition_data(
        competition=args.competition,
        raw_dir=args.raw_dir,
        files=requested_files,
        force=args.force,
        all_files=args.all_files,
    )
    extracted = extract_archives(args.raw_dir)
    csv_files = sorted(path for path in args.raw_dir.rglob("*.csv"))

    print(f"Downloaded competition files into: {args.raw_dir}")
    if args.all_files:
        print("Download mode: all files")
    else:
        print(f"Download mode: selected files ({', '.join(requested_files)})")
    if extracted:
        print(f"Extracted {len(extracted)} files from zip archives.")
    print("CSV files discovered:")
    for path in csv_files:
        print(f" - {path.relative_to(args.raw_dir)}")


if __name__ == "__main__":
    main()

