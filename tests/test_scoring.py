"""Tests for production scorecard helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.config import load_scorecard_settings
from credit_visable.scoring import (
    apply_cutoff_policy,
    apply_platt_calibrator,
    assign_frozen_risk_bands,
    assign_hybrid_risk_bands,
    assign_risk_band_from_pd,
    assign_risk_band_from_score,
    build_calibration_table,
    build_operational_risk_band_table,
    build_profit_assumption_config,
    build_risk_band_table,
    build_score_cutoff_grid,
    build_scorecard_metadata,
    build_unit_economics_frame,
    build_unit_economics_summary,
    compute_calibration_monitoring_metrics,
    compute_expected_value_frame,
    compute_threshold_profit_curve,
    evaluate_cutoff_curve,
    fit_platt_calibrator,
    freeze_risk_band_thresholds,
    pd_to_score,
    run_governed_application_pipeline,
    run_cutoff_sensitivity_analysis,
    score_to_pd,
    select_final_scenario_cutoff,
    select_optimal_cutoff,
    select_optimal_profit_threshold,
)


def _scorecard_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "calibrated_pd": [0.02, 0.04, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60],
            "TARGET": [0, 0, 0, 1, 0, 1, 1, 1],
            "score": [655, 635, 600, 570, 545, 530, 505, 470],
            "AMT_CREDIT": [100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 220_000, 240_000],
            "AMT_ANNUITY": [8_000, 8_500, 9_000, 9_500, 10_000, 11_000, 12_000, 13_000],
            "AMT_INCOME_TOTAL": [240_000, 240_000, 240_000, 240_000, 240_000, 240_000, 240_000, 240_000],
            "AMT_GOODS_PRICE": [95_000, 110_000, 130_000, 150_000, 175_000, 190_000, 215_000, 235_000],
        }
    )


def test_pdo_score_round_trip_is_stable() -> None:
    metadata = build_scorecard_metadata(base_score=600, base_odds=20, points_to_double_odds=40)
    pd_values = pd.Series([0.02, 0.05, 0.10, 0.20], name="pd")

    scores = pd_to_score(pd_values, metadata)
    reconstructed_pd = score_to_pd(scores, metadata)

    assert np.allclose(pd_values.to_numpy(), reconstructed_pd.to_numpy(), atol=1e-9)


def test_risk_band_assignment_uses_configured_thresholds() -> None:
    scorecard_settings = load_scorecard_settings()
    pd_values = pd.Series([0.015, 0.03, 0.06, 0.10, 0.20])
    score_values = pd.Series([660, 620, 590, 550, 500])

    bands_from_pd = assign_risk_band_from_pd(pd_values, scorecard_settings.risk_bands.thresholds)
    bands_from_score = assign_risk_band_from_score(score_values, scorecard_settings.risk_bands.thresholds)

    assert bands_from_pd.tolist() == ["A", "B", "C", "D", "E"]
    assert bands_from_score.tolist() == ["A", "B", "C", "D", "E"]
    assert set(build_risk_band_table(scorecard_settings.risk_bands.thresholds)["risk_band"]) == {"A", "B", "C", "D", "E"}


def test_hybrid_risk_band_assignment_is_monotonic_on_phase6_fixture() -> None:
    scorecard_settings = load_scorecard_settings()
    fixture = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "scorecard_cutoff"
        / "xgboost_traditional_plus_proxy"
        / "calibrated_validation_scores.csv"
    )

    assigned_bands, summary = assign_hybrid_risk_bands(
        frame=fixture,
        calibrated_pd_column="calibrated_pd",
        score_column="score",
        risk_band_config=scorecard_settings.risk_bands,
        target_column="TARGET",
    )
    band_table = build_operational_risk_band_table(summary)

    assert not assigned_bands.empty
    assert assigned_bands.isna().sum() == 0
    assert summary["actual_default_rate"].is_monotonic_increasing
    assert summary["mean_calibrated_pd"].is_monotonic_increasing
    assert float(summary.loc[summary["risk_band"].isin(["A", "B", "C"]), "population_share"].sum()) > 0.5
    assert band_table["risk_band"].tolist() == ["A", "B", "C", "D", "E"]


def test_freeze_risk_band_thresholds_are_reusable() -> None:
    frame = _scorecard_test_frame()
    scorecard_settings = load_scorecard_settings()
    frozen_band_table = freeze_risk_band_thresholds(
        frame,
        calibrated_pd_column="calibrated_pd",
        score_column="score",
        risk_band_config=scorecard_settings.risk_bands,
    )
    assigned_bands, summary = assign_frozen_risk_bands(
        frame,
        score_column="score",
        calibrated_pd_column="calibrated_pd",
        band_threshold_table=frozen_band_table,
        target_column="TARGET",
    )

    assert assigned_bands.tolist()[0] == "A"
    assert assigned_bands.tolist()[-1] == "E"
    assert summary["risk_band"].tolist() == frozen_band_table.sort_values(
        "min_score", ascending=False
    )["risk_band"].tolist()


def test_apply_cutoff_policy_returns_expected_labels() -> None:
    decisions = apply_cutoff_policy(pd.Series([650, 590, 530]), approve_min_score=610, review_min_score=540)
    assert decisions.tolist() == ["approve", "review", "reject"]


def test_platt_calibration_outputs_finite_probabilities() -> None:
    raw_pd = np.array([0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60, 0.75])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    calibrator = fit_platt_calibrator(raw_pd, y_true)
    calibrated = apply_platt_calibrator(raw_pd, calibrator)
    metrics = compute_calibration_monitoring_metrics(y_true, calibrated, raw_pd, calibrated, bins=4)
    calibration_table = build_calibration_table(y_true, calibrated, bins=4)

    assert np.isfinite(calibrated).all()
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert metrics["brier_score"] >= 0.0
    assert metrics["score_psi"] is not None
    assert not calibration_table.empty


def test_legacy_profit_curve_helpers_remain_available() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.10, 0.20, 0.30, 0.80, 0.15, 0.65])

    profit_curve = compute_threshold_profit_curve(
        y_true=y_true,
        y_score=y_score,
        thresholds=[0.25, 0.50, 0.75],
        profit_assumptions=build_profit_assumption_config(),
    )
    best = select_optimal_profit_threshold(profit_curve)

    assert set(profit_curve.columns) >= {"threshold", "approval_rate", "total_profit"}
    assert best["total_profit"] == float(profit_curve["total_profit"].max())


def test_unit_economics_are_finite_and_monotonic() -> None:
    frame = _scorecard_test_frame()
    scorecard_settings = load_scorecard_settings()
    economics = build_unit_economics_frame(frame, scorecard_settings.unit_economics)

    assert np.isfinite(economics.to_numpy()).all()
    assert economics.loc[7, "approve_good"] > economics.loc[0, "approve_good"]
    assert economics.loc[7, "reject_bad"] > economics.loc[0, "reject_bad"]
    assert economics.loc[0, "approve_good"] > economics.loc[0, "reject_good"]
    assert economics.loc[0, "reject_bad"] > economics.loc[0, "approve_bad"]
    assert economics.loc[7, "lgd"] >= economics.loc[0, "lgd"]


def test_expected_values_and_cutoff_curve_respect_guardrails() -> None:
    frame = _scorecard_test_frame()
    scorecard_settings = load_scorecard_settings()
    economics = build_unit_economics_frame(frame, scorecard_settings.unit_economics)
    expected_values = compute_expected_value_frame(frame, "calibrated_pd", economics)
    cutoff_grid = build_score_cutoff_grid(frame["score"], step=5)
    cutoff_curve = evaluate_cutoff_curve(
        frame=frame,
        score_column="score",
        calibrated_pd_column="calibrated_pd",
        unit_economics_frame=economics,
        cutoff_grid=cutoff_grid,
        review_buffer_points=scorecard_settings.cutoff_strategy.review_buffer_points,
        target_column="TARGET",
        guardrails=scorecard_settings.cutoff_strategy.guardrails,
    )
    best_row = select_optimal_cutoff(cutoff_curve)

    assert set(expected_values.columns) == {"ev_approve", "ev_reject", "ev_review"}
    assert not cutoff_curve.empty
    assert "passes_guardrails" in cutoff_curve.columns
    assert best_row["score_cutoff"] in cutoff_curve["score_cutoff"].tolist()


def test_sensitivity_analysis_returns_final_cutoff() -> None:
    frame = _scorecard_test_frame()
    scorecard_settings = load_scorecard_settings()
    cutoff_grid = build_score_cutoff_grid(frame["score"], step=5)

    sensitivity_summary, combined_curve = run_cutoff_sensitivity_analysis(
        frame=frame,
        score_column="score",
        calibrated_pd_column="calibrated_pd",
        unit_economics_config=scorecard_settings.unit_economics,
        cutoff_strategy_config=scorecard_settings.cutoff_strategy,
        sensitivity_config=scorecard_settings.sensitivity_analysis,
        cutoff_grid=cutoff_grid,
        target_column="TARGET",
    )
    final_cutoff = select_final_scenario_cutoff(sensitivity_summary)
    summary_frame = build_unit_economics_summary(
        build_unit_economics_frame(frame, scorecard_settings.unit_economics)
    )

    assert not sensitivity_summary.empty
    assert not combined_curve.empty
    assert isinstance(final_cutoff, int)
    assert "population_size" in summary_frame["metric"].tolist()


def test_governed_pipeline_symbol_is_importable() -> None:
    assert callable(run_governed_application_pipeline)
