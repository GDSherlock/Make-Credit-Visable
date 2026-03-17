"""Starter fairness summaries for grouped outcome inspection."""

from __future__ import annotations

import pandas as pd


def fairness_report_placeholder(
    frame: pd.DataFrame,
    target_column: str,
    protected_columns: list[str],
) -> pd.DataFrame:
    """Create a simple grouped summary for future fairness analysis.

    This is not a full fairness audit. It only prepares group counts and target rates.
    """

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    missing = [column for column in protected_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Protected columns not found: {missing}")

    outputs: list[pd.DataFrame] = []
    for column in protected_columns:
        summary = (
            frame.groupby(column, dropna=False)[target_column]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={column: "group", "mean": "target_rate"})
        )
        summary.insert(0, "protected_attribute", column)
        outputs.append(summary)

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return result
