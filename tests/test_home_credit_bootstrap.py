"""Tests for the Kaggle bootstrap CLI and helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.config import Settings
from credit_visable.data.home_credit_bootstrap import (
    BootstrapError,
    bootstrap_home_credit_data,
    download_competition_data,
    main,
)


def test_main_reports_missing_kaggle_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap.shutil.which",
        lambda _name: None,
    )

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "kaggle" in captured.err
    assert "Install the Kaggle CLI outside this project environment" in captured.err


def test_main_reports_missing_authentication(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap.shutil.which",
        lambda _name: "/usr/local/bin/kaggle",
    )
    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap._has_kaggle_auth",
        lambda: False,
    )

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "~/.kaggle/kaggle.json" in captured.err
    assert "~/.kaggle/access_token" in captured.err
    assert "KAGGLE_API_TOKEN" in captured.err


@pytest.mark.parametrize("force", [False, True])
def test_download_command_uses_expected_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, force: bool
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap._assert_kaggle_command_available",
        lambda: "kaggle",
    )
    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap._assert_kaggle_auth_available",
        lambda: None,
    )

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
        commands.append(command)

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return Result()

    monkeypatch.setattr(
        "credit_visable.data.home_credit_bootstrap.subprocess.run",
        fake_run,
    )

    resolved_dir = download_competition_data(
        competition="home-credit-default-risk",
        download_dir=tmp_path,
        force=force,
    )

    assert resolved_dir == tmp_path
    assert commands == [
        [
            "kaggle",
            "competitions",
            "download",
            "home-credit-default-risk",
            "-p",
            str(tmp_path),
            *(["--force"] if force else []),
        ]
    ]


def test_bootstrap_materializes_zip_and_csv_artifacts(tmp_path: Path) -> None:
    download_dir = tmp_path / "external"
    raw_dir = tmp_path / "raw"
    download_dir.mkdir(parents=True)

    archive_path = download_dir / "home-credit-default-risk.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("application_train.csv", "SK_ID_CURR,TARGET\n1,0\n")
        archive.writestr("nested/application_test.csv", "SK_ID_CURR\n2\n")

    (download_dir / "bureau.csv").write_text("SK_ID_BUREAU\n10\n", encoding="utf-8")

    settings = Settings(
        expected_tables={
            "application_train": "application_train.csv",
            "application_test": "application_test.csv",
            "bureau": "bureau.csv",
        }
    )

    summary = bootstrap_home_credit_data(
        download_dir=download_dir,
        raw_dir=raw_dir,
        skip_download=True,
        settings=settings,
    )

    assert summary.downloaded is False
    assert (raw_dir / "application_train.csv").exists()
    assert (raw_dir / "application_test.csv").exists()
    assert (raw_dir / "bureau.csv").exists()
    assert len(summary.extracted_files) == 2
    assert len(summary.copied_files) == 1


def test_bootstrap_reports_missing_expected_tables(tmp_path: Path) -> None:
    download_dir = tmp_path / "external"
    raw_dir = tmp_path / "raw"
    download_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    (download_dir / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n1,0\n",
        encoding="utf-8",
    )

    settings = Settings(
        expected_tables={
            "application_train": "application_train.csv",
            "bureau": "bureau.csv",
        }
    )

    with pytest.raises(BootstrapError, match="bureau"):
        bootstrap_home_credit_data(
            download_dir=download_dir,
            raw_dir=raw_dir,
            skip_download=True,
            settings=settings,
        )
