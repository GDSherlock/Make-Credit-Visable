"""Tests for table loading and availability summaries."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.config import Settings
from credit_visable.data.load_data import list_available_tables, summarize_table_availability


def _settings() -> Settings:
    return Settings(
        expected_tables={
            "application_train": "application_train.csv",
            "application_test": "application_test.csv",
            "bureau": "bureau.csv",
        }
    )


def test_summarize_table_availability_for_empty_directory(tmp_path: Path) -> None:
    summary = summarize_table_availability(data_dir=tmp_path, settings=_settings())

    assert summary.columns.tolist() == [
        "table_name",
        "file_name",
        "resolved_path",
        "available",
    ]
    assert summary["table_name"].tolist() == [
        "application_train",
        "application_test",
        "bureau",
    ]
    assert summary["available"].tolist() == [False, False, False]


def test_summarize_table_availability_for_partial_directory(tmp_path: Path) -> None:
    (tmp_path / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n1,0\n",
        encoding="utf-8",
    )

    summary = summarize_table_availability(data_dir=tmp_path, settings=_settings())

    availability = dict(zip(summary["table_name"], summary["available"]))
    assert availability == {
        "application_train": True,
        "application_test": False,
        "bureau": False,
    }
    assert list_available_tables(data_dir=tmp_path, settings=_settings()) == {
        "application_train": tmp_path / "application_train.csv"
    }


def test_summarize_table_availability_for_full_directory(tmp_path: Path) -> None:
    for file_name in _settings().expected_tables.values():
        (tmp_path / file_name).write_text("col\n1\n", encoding="utf-8")

    summary = summarize_table_availability(data_dir=tmp_path, settings=_settings())

    assert summary["available"].tolist() == [True, True, True]


def test_table_availability_supports_custom_data_dir_override(tmp_path: Path) -> None:
    raw_dir = tmp_path / "custom-raw"
    raw_dir.mkdir()
    (raw_dir / "bureau.csv").write_text("SK_ID_BUREAU\n10\n", encoding="utf-8")

    available = list_available_tables(data_dir=raw_dir, settings=_settings())
    summary = summarize_table_availability(data_dir=raw_dir, settings=_settings())

    assert available == {"bureau": raw_dir / "bureau.csv"}
    assert summary.loc[summary["table_name"] == "bureau", "resolved_path"].item() == str(
        raw_dir / "bureau.csv"
    )
    assert summary.loc[summary["table_name"] == "bureau", "available"].item() is True


def test_table_availability_ignores_extra_csv_files(tmp_path: Path) -> None:
    (tmp_path / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n1,0\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "SK_ID_CURR,TARGET\n100001,0.1\n",
        encoding="utf-8",
    )
    (tmp_path / "HomeCredit_columns_description.csv").write_text(
        "Table,Column,Description\napplication_train,TARGET,target label\n",
        encoding="utf-8",
    )

    summary = summarize_table_availability(data_dir=tmp_path, settings=_settings())

    assert summary["table_name"].tolist() == [
        "application_train",
        "application_test",
        "bureau",
    ]
    assert summary["available"].tolist() == [True, False, False]
    assert list_available_tables(data_dir=tmp_path, settings=_settings()) == {
        "application_train": tmp_path / "application_train.csv"
    }
