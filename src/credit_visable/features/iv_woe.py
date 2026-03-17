"""Starter placeholders for IV / WOE analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_iv_summary(
    frame: pd.DataFrame,
    target_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Return a lightweight placeholder IV summary table.

    This function intentionally does not implement full IV logic yet.
    It produces a schema-ready summary that can be extended later.
    """

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    feature_columns = [column for column in frame.columns if column != target_column]
    summary = pd.DataFrame(
        {
            "feature": feature_columns,
            "iv": np.nan,
            "bins_requested": bins,
            "status": "TODO: implement IV calculation",
        }
    )
    return summary


def fit_woe_placeholder(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    """Return metadata for a future WOE transformer implementation."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    selected_features = feature_columns or [
        column for column in frame.columns if column != target_column
    ]

    return {
        "target_column": target_column,
        "features": selected_features,
        "fitted": False,
        "notes": "TODO: replace this placeholder with a reusable WOE transformer.",
    }
