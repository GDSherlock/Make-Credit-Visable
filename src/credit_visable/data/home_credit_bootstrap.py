"""Bootstrap Kaggle competition data into the project raw data directory."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from credit_visable.config import Settings, load_settings
from credit_visable.utils.paths import get_paths


DEFAULT_COMPETITION = "home-credit-default-risk"


class BootstrapError(RuntimeError):
    """Raised when the Kaggle bootstrap flow cannot complete."""


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    """Summary of the bootstrap run."""

    competition: str
    download_dir: Path
    raw_dir: Path
    downloaded: bool
    copied_files: tuple[Path, ...]
    extracted_files: tuple[Path, ...]
    skipped_files: tuple[Path, ...]


def _default_download_dir(competition: str) -> Path:
    return get_paths().data_external / "kaggle" / competition


def _default_raw_dir() -> Path:
    return get_paths().data_raw


def _resolve_project_path(path_value: str | Path | None, default: Path) -> Path:
    if path_value is None:
        return default

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return get_paths().root / candidate


def _has_kaggle_auth() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    return any(
        [
            (kaggle_dir / "kaggle.json").exists(),
            (kaggle_dir / "access_token").exists(),
            bool(os.getenv("KAGGLE_API_TOKEN")),
        ]
    )


def _assert_kaggle_command_available() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise BootstrapError(
            "The `kaggle` command is not available. Install the Kaggle CLI outside "
            "this project environment and ensure it is on PATH."
        )
    return executable


def _assert_kaggle_auth_available() -> None:
    if _has_kaggle_auth():
        return

    raise BootstrapError(
        "Kaggle authentication was not found. Provide ~/.kaggle/kaggle.json, "
        "~/.kaggle/access_token, or KAGGLE_API_TOKEN before downloading."
    )


def _run_kaggle_command(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        return

    detail = completed.stderr.strip() or completed.stdout.strip()
    message = (
        "Kaggle download failed. Confirm the competition rules were accepted on "
        "the Kaggle website before retrying."
    )
    if detail:
        message = f"{message} CLI output: {detail}"
    raise BootstrapError(message)


def download_competition_data(
    competition: str = DEFAULT_COMPETITION,
    download_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download Kaggle competition files into the external data directory."""

    resolved_dir = _resolve_project_path(download_dir, _default_download_dir(competition))
    resolved_dir.mkdir(parents=True, exist_ok=True)

    kaggle_executable = _assert_kaggle_command_available()
    _assert_kaggle_auth_available()

    command = [
        kaggle_executable,
        "competitions",
        "download",
        competition,
        "-p",
        str(resolved_dir),
    ]
    if force:
        command.append("--force")

    _run_kaggle_command(command)
    return resolved_dir


def _copy_file(source: Path, target: Path, force: bool) -> Path | None:
    if target.exists() and not force:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _extract_csv_members(archive_path: Path, raw_dir: Path, force: bool) -> tuple[Path, ...]:
    extracted: list[Path] = []

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            if member_path.suffix.lower() != ".csv":
                continue

            target = raw_dir / member_path.name
            if target.exists() and not force:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(target)

    return tuple(extracted)


def materialize_downloads(
    download_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Copy direct CSV downloads and extract archive members into raw data."""

    resolved_download_dir = _resolve_project_path(
        download_dir, _default_download_dir(DEFAULT_COMPETITION)
    )
    resolved_raw_dir = _resolve_project_path(raw_dir, _default_raw_dir())
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    extracted_files: list[Path] = []
    skipped_files: list[Path] = []

    if not resolved_download_dir.exists():
        return (), (), ()

    for artifact in sorted(resolved_download_dir.rglob("*")):
        if not artifact.is_file():
            continue

        if artifact.suffix.lower() == ".zip":
            extracted_before = set(extracted_files)
            extracted_now = _extract_csv_members(artifact, resolved_raw_dir, force)
            extracted_files.extend(extracted_now)

            with ZipFile(artifact) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    if Path(member.filename).suffix.lower() != ".csv":
                        continue
                    target = resolved_raw_dir / Path(member.filename).name
                    if target.exists() and not force and target not in extracted_before:
                        if target not in extracted_now:
                            skipped_files.append(target)
            continue

        if artifact.suffix.lower() != ".csv":
            continue

        target = resolved_raw_dir / artifact.name
        copied = _copy_file(artifact, target, force)
        if copied is None:
            skipped_files.append(target)
            continue
        copied_files.append(copied)

    return tuple(copied_files), tuple(extracted_files), tuple(skipped_files)


def validate_expected_tables(
    raw_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> tuple[str, ...]:
    """Return configured table names that are still missing from the raw directory."""

    resolved_settings = settings or load_settings()
    resolved_raw_dir = _resolve_project_path(raw_dir, _default_raw_dir())

    missing: list[str] = []
    for table_name, file_name in resolved_settings.expected_tables.items():
        if not (resolved_raw_dir / file_name).exists():
            missing.append(table_name)
    return tuple(missing)


def bootstrap_home_credit_data(
    competition: str = DEFAULT_COMPETITION,
    download_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    skip_download: bool = False,
    force: bool = False,
    settings: Settings | None = None,
) -> BootstrapSummary:
    """Download, materialize, and validate competition data for the project."""

    resolved_download_dir = _resolve_project_path(download_dir, _default_download_dir(competition))
    resolved_raw_dir = _resolve_project_path(raw_dir, _default_raw_dir())

    downloaded = False
    if not skip_download:
        download_competition_data(
            competition=competition,
            download_dir=resolved_download_dir,
            force=force,
        )
        downloaded = True

    copied_files, extracted_files, skipped_files = materialize_downloads(
        download_dir=resolved_download_dir,
        raw_dir=resolved_raw_dir,
        force=force,
    )

    missing = validate_expected_tables(raw_dir=resolved_raw_dir, settings=settings)
    if missing:
        raise BootstrapError(
            "Missing expected tables in raw data directory: "
            f"{', '.join(missing)}. Check configs/base.yaml and the downloaded artifacts."
        )

    return BootstrapSummary(
        competition=competition,
        download_dir=resolved_download_dir,
        raw_dir=resolved_raw_dir,
        downloaded=downloaded,
        copied_files=copied_files,
        extracted_files=extracted_files,
        skipped_files=skipped_files,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Kaggle bootstrap flow."""

    parser = argparse.ArgumentParser(
        description="Download and materialize Kaggle Home Credit competition data."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug. Defaults to home-credit-default-risk.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory for original Kaggle downloads. Defaults to data/external/kaggle/<competition>.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory for CSV files used by the project. Defaults to data/raw.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Kaggle download step and only materialize or validate existing artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow download artifacts and raw CSV files to be overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = bootstrap_home_credit_data(
            competition=args.competition,
            download_dir=args.download_dir,
            raw_dir=args.raw_dir,
            skip_download=args.skip_download,
            force=args.force,
        )
    except BootstrapError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Competition: {summary.competition}")
    print(f"Download dir: {summary.download_dir}")
    print(f"Raw dir: {summary.raw_dir}")
    print(f"Download executed: {'yes' if summary.downloaded else 'no'}")
    print(f"Copied CSV files: {len(summary.copied_files)}")
    print(f"Extracted CSV files: {len(summary.extracted_files)}")
    print(f"Skipped existing files: {len(summary.skipped_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
