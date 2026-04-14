"""Scorecard, calibration, and decision-economics helpers."""

from __future__ import annotations

from typing import Any

from credit_visable.scoring.calibration import (
    apply_platt_calibrator,
    build_calibration_table,
    compute_calibration_monitoring_metrics,
    compute_population_stability_index,
    fit_platt_calibrator,
)
from credit_visable.scoring.economics import (
    build_unit_economics_frame,
    build_unit_economics_summary,
    compute_expected_value_frame,
    evaluate_cutoff_curve,
    run_cutoff_sensitivity_analysis,
    select_final_scenario_cutoff,
    select_optimal_cutoff,
)
from credit_visable.scoring.pdo_scorecard import (
    apply_cutoff_policy,
    assign_frozen_risk_bands,
    assign_hybrid_risk_bands,
    assign_risk_band_from_pd,
    assign_risk_band_from_score,
    freeze_risk_band_thresholds,
    build_operational_risk_band_table,
    build_profit_assumption_config,
    build_risk_band_table,
    build_score_cutoff_grid,
    build_scorecard_metadata,
    build_scorecard_placeholder,
    compute_threshold_profit_curve,
    odds_to_score,
    pd_to_score,
    resolve_scaling_metadata,
    score_to_odds,
    score_to_pd,
    select_optimal_profit_threshold,
)


def refresh_phase6_artifacts(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper to avoid importing the reporting stack during package init."""

    from credit_visable.scoring.phase6_reporting import refresh_phase6_artifacts as _impl

    return _impl(*args, **kwargs)


def run_governed_application_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper to avoid circular imports during package initialization."""

    from credit_visable.scoring.phase6_reporting import (
        run_governed_application_pipeline as _impl,
    )

    return _impl(*args, **kwargs)

__all__ = [
    "apply_cutoff_policy",
    "apply_platt_calibrator",
    "assign_frozen_risk_bands",
    "assign_hybrid_risk_bands",
    "assign_risk_band_from_pd",
    "assign_risk_band_from_score",
    "build_calibration_table",
    "build_operational_risk_band_table",
    "build_profit_assumption_config",
    "build_risk_band_table",
    "build_score_cutoff_grid",
    "build_scorecard_metadata",
    "build_scorecard_placeholder",
    "build_unit_economics_frame",
    "build_unit_economics_summary",
    "compute_calibration_monitoring_metrics",
    "compute_expected_value_frame",
    "compute_population_stability_index",
    "compute_threshold_profit_curve",
    "evaluate_cutoff_curve",
    "fit_platt_calibrator",
    "freeze_risk_band_thresholds",
    "odds_to_score",
    "pd_to_score",
    "refresh_phase6_artifacts",
    "run_governed_application_pipeline",
    "resolve_scaling_metadata",
    "run_cutoff_sensitivity_analysis",
    "score_to_odds",
    "score_to_pd",
    "select_final_scenario_cutoff",
    "select_optimal_cutoff",
    "select_optimal_profit_threshold",
]
