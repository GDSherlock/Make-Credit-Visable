"""Small memory helpers for exploratory work on wide credit datasets."""

from __future__ import annotations

import pandas as pd


def memory_usage_mb(frame: pd.DataFrame) -> float:
    """Return total DataFrame memory usage in megabytes."""

    return float(frame.memory_usage(deep=True).sum() / (1024**2))


def downcast_numeric_types(frame: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    This is intentionally conservative and only applies pandas downcasting rules.
    TODO: Add nullable dtype handling once the preprocessing pipeline is stabilized.
    """

    result = frame if inplace else frame.copy()

    integer_columns = result.select_dtypes(include=["integer"]).columns
    float_columns = result.select_dtypes(include=["floating"]).columns

    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], downcast="integer")

    for column in float_columns:
        result[column] = pd.to_numeric(result[column], downcast="float")

    return result
