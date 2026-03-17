"""Utilities for loading Kaggle tables from the project data directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_visable.config import Settings, load_settings
from credit_visable.utils.paths import get_paths


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the raw data directory used by the loader."""

    return Path(data_dir) if data_dir is not None else get_paths().data_raw


def list_available_tables(
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Return configured table names that currently exist on disk."""

    resolved_settings = settings or load_settings()
    resolved_dir = _resolve_data_dir(data_dir)

    available: dict[str, Path] = {}
    for table_name, file_name in resolved_settings.expected_tables.items():
        candidate = resolved_dir / file_name
        if candidate.exists():
            available[table_name] = candidate
    return available


def load_table(
    table_name: str,
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a configured CSV table by logical name.

    Notes
    -----
    This starter assumes CSV inputs because the Kaggle release is CSV-based.
    TODO: Extend to parquet or explicit schema validation when the pipeline matures.
    """

    resolved_settings = settings or load_settings()
    resolved_dir = _resolve_data_dir(data_dir)
    file_name = resolved_settings.expected_tables.get(table_name, table_name)
    file_path = resolved_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Table '{table_name}' was not found at {file_path}. "
            "Place the Kaggle CSV files under data/raw/."
        )

    return pd.read_csv(file_path, **read_csv_kwargs)


def load_application_train(
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Convenience loader for `application_train.csv`."""

    return load_table(
        "application_train",
        data_dir=data_dir,
        settings=settings,
        **read_csv_kwargs,
    )


def load_application_test(
    data_dir: str | Path | None = None,
    settings: Settings | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Convenience loader for `application_test.csv`."""

    return load_table(
        "application_test",
        data_dir=data_dir,
        settings=settings,
        **read_csv_kwargs,
    )
