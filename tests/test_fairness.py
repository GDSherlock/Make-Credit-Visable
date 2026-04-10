"""Tests for Phase 5 fairness and grouped governance helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.governance import (
    build_group_fairness_metric_summary,
    build_group_fairness_summary,
    build_grouped_operational_summary,
    collapse_rare_categories,
    derive_age_band_from_days_birth,
)


def test_derive_age_band_from_days_birth_uses_fixed_phase5_bins() -> None:
    days_birth = pd.Series([-22 * 365.25, -31 * 365.25, -67 * 365.25, None])

    age_band = derive_age_band_from_days_birth(days_birth)

    assert age_band.tolist() == ["[20,25)", "[25,35)", "[65,70)", "Missing"]


def test_collapse_rare_categories_keeps_top_n_and_missing() -> None:
    values = pd.Series(["A", "A", "B", "C", None, "D"])

    collapsed = collapse_rare_categories(values, top_n=2)

    assert collapsed.tolist() == ["A", "A", "B", "Other", "Missing", "Other"]


def test_build_grouped_operational_summary_returns_phase5_metrics() -> None:
    frame = pd.DataFrame(
        {
            "group_name": ["A", "A", "B", "B", "C", "C"],
            "TARGET": [0, 1, 1, 0, 0, 0],
            "predicted_pd": [0.10, 0.70, 0.20, 0.80, 0.30, 0.40],
        }
    )

    summary = build_grouped_operational_summary(
        frame=frame,
        group_column="group_name",
        target_column="TARGET",
        score_column="predicted_pd",
        threshold=0.5,
        protected_attribute="demo_group",
    )

    assert set(summary.columns) == {
        "protected_attribute",
        "group",
        "count",
        "population_share",
        "bad_count",
        "actual_default_rate",
        "mean_predicted_pd",
        "approval_rate",
        "reject_rate",
        "approved_count",
        "rejected_count",
        "approved_bad_rate",
        "rejected_bad_capture_rate",
        "approval_rate_vs_best_group",
        "threshold",
    }
    group_a = summary.loc[summary["group"] == "A"].iloc[0]
    group_c = summary.loc[summary["group"] == "C"].iloc[0]
    assert group_a["approval_rate"] == 0.5
    assert group_a["approved_bad_rate"] == 0.0
    assert group_a["rejected_bad_capture_rate"] == 0.5
    assert group_c["approval_rate"] == 1.0
    assert group_a["approval_rate_vs_best_group"] == 0.5


def test_build_group_fairness_summary_handles_age_bands_and_top_categories() -> None:
    frame = pd.DataFrame(
        {
            "TARGET": [0, 1, 0, 1, 0],
            "predicted_pd": [0.10, 0.80, 0.20, 0.60, 0.30],
            "DAYS_BIRTH": [-22 * 365.25, -31 * 365.25, -40 * 365.25, -52 * 365.25, None],
            "ORGANIZATION_TYPE": ["OrgA", "OrgA", "OrgB", None, "OrgC"],
        }
    )
    group_specs = [
        {
            "protected_attribute": "age_band",
            "source_column": "DAYS_BIRTH",
            "group_column": "age_band",
            "kind": "age_band",
        },
        {
            "protected_attribute": "organization_group",
            "source_column": "ORGANIZATION_TYPE",
            "group_column": "organization_group",
            "kind": "top_categories",
            "top_n": 1,
        },
    ]

    summary = build_group_fairness_summary(
        frame=frame,
        target_column="TARGET",
        score_column="predicted_pd",
        group_specs=group_specs,
        threshold=0.5,
        top_n_categories=8,
    )

    assert set(summary["protected_attribute"]) == {"age_band", "organization_group"}
    assert "group_column" in summary.columns
    assert "source_column" in summary.columns
    organization_groups = summary.loc[
        summary["protected_attribute"] == "organization_group",
        "group",
    ].tolist()
    assert "OrgA" in organization_groups
    assert "Other" in organization_groups
    assert "Missing" in organization_groups


def test_build_group_fairness_metric_summary_returns_regulator_style_metrics() -> None:
    frame = pd.DataFrame(
        {
            "TARGET": [0, 0, 1, 1, 0, 1, 0, 1],
            "predicted_pd": [0.10, 0.20, 0.30, 0.80, 0.15, 0.65, 0.55, 0.40],
            "NAME_FAMILY_STATUS": [
                "Married",
                "Married",
                "Married",
                "Married",
                "Single",
                "Single",
                "Single",
                "Single",
            ],
        }
    )
    group_specs = [
        {
            "protected_attribute": "family_status_group",
            "source_column": "NAME_FAMILY_STATUS",
            "group_column": "family_status_group",
            "kind": "identity",
        }
    ]

    summary = build_group_fairness_metric_summary(
        frame=frame,
        target_column="TARGET",
        score_column="predicted_pd",
        group_specs=group_specs,
        threshold=0.5,
    )

    assert set(summary.columns) >= {
        "protected_attribute",
        "demographic_parity_diff",
        "demographic_parity_ratio",
        "approval_disparity_ratio",
        "equal_opportunity_diff",
        "equalized_odds_gap",
    }
    row = summary.iloc[0]
    assert row["protected_attribute"] == "family_status_group"
    assert row["demographic_parity_diff"] >= 0.0
    assert 0.0 <= row["demographic_parity_ratio"] <= 1.0
    assert row["equalized_odds_gap"] >= 0.0
