"""Tests for IV / WOE helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.features import compute_iv_summary, compute_woe_detail


def test_compute_woe_detail_returns_expected_schema() -> None:
    frame = pd.DataFrame(
        {
            "TARGET": [0, 0, 1, 1, 0, 1, 0, 1],
            "AGE_YEARS": [23, 29, 34, 41, 47, 53, None, 61],
        }
    )

    detail = compute_woe_detail(
        frame=frame,
        target_column="TARGET",
        feature_column="AGE_YEARS",
        bins=4,
    )

    assert set(detail.columns) == {
        "feature",
        "feature_type",
        "bin_order",
        "bin",
        "total_count",
        "good_count",
        "bad_count",
        "distribution_share",
        "event_rate",
        "good_share",
        "bad_share",
        "woe",
        "iv_component",
        "status",
    }
    assert detail["feature"].nunique() == 1
    assert detail["bin_order"].tolist() == sorted(detail["bin_order"].tolist())
    assert (detail["good_count"] + detail["bad_count"]).tolist() == detail["total_count"].tolist()


def test_compute_iv_summary_ranks_predictive_feature_above_noise() -> None:
    frame = pd.DataFrame(
        {
            "TARGET": [0, 0, 0, 0, 1, 1, 1, 1],
            "predictive_feature": [10, 11, 12, 13, 80, 82, 85, 90],
            "weak_feature": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )

    summary = compute_iv_summary(frame=frame, target_column="TARGET", bins=4)

    assert summary.iloc[0]["feature"] == "predictive_feature"
    assert summary.loc[summary["feature"] == "predictive_feature", "iv"].iloc[0] > 0.0
    assert set(summary["status"]) == {"ok"}
