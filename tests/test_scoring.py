"""Tests for Phase 6 scoring and profit helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.scoring import (
    build_profit_assumption_config,
    compute_threshold_profit_curve,
    select_optimal_profit_threshold,
)


def test_compute_threshold_profit_curve_returns_expected_columns() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.10, 0.20, 0.30, 0.80, 0.15, 0.65])

    profit_curve = compute_threshold_profit_curve(
        y_true=y_true,
        y_score=y_score,
        thresholds=[0.25, 0.50, 0.75],
        profit_assumptions=build_profit_assumption_config(),
    )

    assert set(profit_curve.columns) == {
        "threshold",
        "approval_rate",
        "reject_rate",
        "approved_good_count",
        "approved_bad_count",
        "rejected_good_count",
        "rejected_bad_count",
        "approved_bad_rate",
        "rejected_bad_capture_rate",
        "total_profit",
        "profit_per_applicant",
    }
    assert np.allclose(profit_curve["approval_rate"] + profit_curve["reject_rate"], 1.0)


def test_select_optimal_profit_threshold_picks_highest_total_profit() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_score = np.array([0.05, 0.10, 0.25, 0.90, 0.20, 0.70, 0.15, 0.18])

    profit_curve = compute_threshold_profit_curve(
        y_true=y_true,
        y_score=y_score,
        thresholds=[0.15, 0.30, 0.60],
        profit_assumptions=build_profit_assumption_config(),
    )
    best = select_optimal_profit_threshold(profit_curve)

    assert best["threshold"] in {0.15, 0.30, 0.60}
    assert best["total_profit"] == float(profit_curve["total_profit"].max())
