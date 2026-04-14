"""Monitoring helpers for governed scorecard artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_visable.governance.fairness import build_group_fairness_metric_summary
from credit_visable.scoring.calibration import compute_calibration_monitoring_metrics


def build_monitoring_baseline(
    *,
    reference_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    target_column: str,
    calibrated_pd_column: str,
    score_column: str,
    group_specs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Build production-style monitoring diagnostics between two populations."""

    calibration_metrics = compute_calibration_monitoring_metrics(
        y_true=comparison_frame[target_column],
        calibrated_probabilities=comparison_frame[calibrated_pd_column],
        reference_scores=reference_frame[score_column],
        comparison_scores=comparison_frame[score_column],
    )
    reference_default_rate = float(
        pd.to_numeric(reference_frame[target_column], errors="coerce").mean()
    )
    comparison_default_rate = float(
        pd.to_numeric(comparison_frame[target_column], errors="coerce").mean()
    )

    reference_fairness = build_group_fairness_metric_summary(
        frame=reference_frame,
        target_column=target_column,
        score_column=calibrated_pd_column,
        group_specs=group_specs,
        threshold=threshold,
    )
    comparison_fairness = build_group_fairness_metric_summary(
        frame=comparison_frame,
        target_column=target_column,
        score_column=calibrated_pd_column,
        group_specs=group_specs,
        threshold=threshold,
    )
    fairness_drift = reference_fairness.merge(
        comparison_fairness,
        on=["protected_attribute", "group_column", "source_column"],
        how="outer",
        suffixes=("_reference", "_comparison"),
    )
    if fairness_drift.empty:
        fairness_drift = pd.DataFrame(
            columns=[
                "protected_attribute",
                "group_column",
                "source_column",
                "demographic_parity_diff_reference",
                "demographic_parity_diff_comparison",
                "equalized_odds_gap_reference",
                "equalized_odds_gap_comparison",
                "demographic_parity_diff_delta",
                "equalized_odds_gap_delta",
            ]
        )
    else:
        fairness_drift["demographic_parity_diff_delta"] = (
            pd.to_numeric(
                fairness_drift["demographic_parity_diff_comparison"],
                errors="coerce",
            )
            - pd.to_numeric(
                fairness_drift["demographic_parity_diff_reference"],
                errors="coerce",
            )
        )
        fairness_drift["equalized_odds_gap_delta"] = (
            pd.to_numeric(
                fairness_drift["equalized_odds_gap_comparison"],
                errors="coerce",
            )
            - pd.to_numeric(
                fairness_drift["equalized_odds_gap_reference"],
                errors="coerce",
            )
        )

    summary = {
        "reference_population_size": int(len(reference_frame)),
        "comparison_population_size": int(len(comparison_frame)),
        "reference_default_rate": reference_default_rate,
        "comparison_default_rate": comparison_default_rate,
        "default_rate_drift": float(comparison_default_rate - reference_default_rate),
        "mean_score_reference": float(pd.to_numeric(reference_frame[score_column], errors="coerce").mean()),
        "mean_score_comparison": float(pd.to_numeric(comparison_frame[score_column], errors="coerce").mean()),
        "mean_score_drift": float(
            pd.to_numeric(comparison_frame[score_column], errors="coerce").mean()
            - pd.to_numeric(reference_frame[score_column], errors="coerce").mean()
        ),
        "calibration_monitoring": calibration_metrics,
        "max_abs_fairness_drift": float(
            np.nanmax(
                np.abs(
                    pd.concat(
                        [
                            fairness_drift["demographic_parity_diff_delta"],
                            fairness_drift["equalized_odds_gap_delta"],
                        ],
                        ignore_index=True,
                    )
                )
            )
        )
        if not fairness_drift.empty
        else 0.0,
    }

    return {
        "summary": summary,
        "fairness_drift": fairness_drift,
        "reference_fairness": reference_fairness,
        "comparison_fairness": comparison_fairness,
    }
