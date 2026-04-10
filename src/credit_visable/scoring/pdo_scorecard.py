"""Placeholder PDO helpers plus policy-profit utilities for Phase 6."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_scorecard_placeholder(
    base_score: int = 600,
    base_odds: float = 50.0,
    points_to_double_odds: int = 20,
) -> dict[str, float | int | str]:
    """Return starter scorecard metadata without implementing score scaling yet."""

    return {
        "base_score": base_score,
        "base_odds": base_odds,
        "points_to_double_odds": points_to_double_odds,
        "ready": False,
        "notes": "TODO: implement PDO scaling once WOE features and calibration are finalized.",
    }


def build_profit_assumption_config(
    approve_good: float = 1.0,
    approve_bad: float = -5.0,
    reject_good: float = -0.2,
    reject_bad: float = 0.0,
) -> dict[str, float]:
    """Return a standardized unit-economics assumption set for Phase 6."""

    return {
        "approve_good": float(approve_good),
        "approve_bad": float(approve_bad),
        "reject_good": float(reject_good),
        "reject_bad": float(reject_bad),
    }


def compute_threshold_profit_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    thresholds: list[float] | np.ndarray,
    profit_assumptions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute approval, bad-rate, and profit outcomes across thresholds."""

    resolved_profit = profit_assumptions or build_profit_assumption_config()
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    if y_true_array.shape[0] != y_score_array.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    total_bad = float(y_true_array.sum())
    rows = []
    for threshold in np.asarray(thresholds, dtype=float):
        approved_mask = y_score_array < threshold
        rejected_mask = ~approved_mask

        approved_good_count = int(np.sum(approved_mask & (y_true_array == 0)))
        approved_bad_count = int(np.sum(approved_mask & (y_true_array == 1)))
        rejected_good_count = int(np.sum(rejected_mask & (y_true_array == 0)))
        rejected_bad_count = int(np.sum(rejected_mask & (y_true_array == 1)))

        total_profit = (
            approved_good_count * resolved_profit["approve_good"]
            + approved_bad_count * resolved_profit["approve_bad"]
            + rejected_good_count * resolved_profit["reject_good"]
            + rejected_bad_count * resolved_profit["reject_bad"]
        )
        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": float(approved_mask.mean()),
                "reject_rate": float(rejected_mask.mean()),
                "approved_good_count": approved_good_count,
                "approved_bad_count": approved_bad_count,
                "rejected_good_count": rejected_good_count,
                "rejected_bad_count": rejected_bad_count,
                "approved_bad_rate": (
                    float(approved_bad_count / approved_mask.sum())
                    if approved_mask.any()
                    else 0.0
                ),
                "rejected_bad_capture_rate": (
                    float(rejected_bad_count / total_bad) if total_bad > 0 else 0.0
                ),
                "total_profit": float(total_profit),
                "profit_per_applicant": float(total_profit / len(y_true_array))
                if len(y_true_array) > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def select_optimal_profit_threshold(profit_curve: pd.DataFrame) -> dict[str, float | int]:
    """Return the best threshold row from a profit curve frame."""

    if profit_curve.empty:
        raise ValueError("profit_curve must not be empty.")
    if "total_profit" not in profit_curve.columns:
        raise KeyError("profit_curve must contain a total_profit column.")

    best_row = profit_curve.sort_values(
        by=["total_profit", "profit_per_applicant", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return {
        key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
        if isinstance(value, (np.integer, int, np.floating, float))
        else value
        for key, value in best_row.to_dict().items()
    }
