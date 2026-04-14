"""Phase 6 artifact refresh helpers built on calibrated PD."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump as dump_joblib
from sklearn.metrics import brier_score_loss

from credit_visable.config import load_scorecard_settings, load_settings
from credit_visable.data import load_application_train
from credit_visable.explainability.shap_analysis import (
    compute_shap_contribution_summary,
    compute_shap_local_explanations,
    select_local_explanation_rows,
)
from credit_visable.features import (
    ApplicationFeatureEngineeringOptions,
    FeatureReviewOptions,
    GovernedSplitOptions,
    PreprocessingOptions,
    engineer_application_features,
    prepare_governed_preprocessing_artifacts,
    save_governed_preprocessing_artifacts,
)
from credit_visable.governance import (
    build_group_fairness_metric_summary,
    build_monitoring_baseline,
    collapse_rare_categories,
    derive_age_band_from_days_birth,
)
from credit_visable.modeling import (
    GovernedTreeTrainingOptions,
    evaluate_binary_classifier,
    train_governed_tree_model,
    train_logistic_baseline,
)
from credit_visable.scoring.calibration import (
    apply_platt_calibrator,
    build_calibration_table,
    fit_platt_calibrator,
)
from credit_visable.scoring.economics import (
    build_unit_economics_frame,
    build_unit_economics_summary,
    evaluate_cutoff_curve,
    select_optimal_cutoff,
)
from credit_visable.scoring.pdo_scorecard import (
    apply_cutoff_policy,
    assign_frozen_risk_bands,
    assign_hybrid_risk_bands,
    build_operational_risk_band_table,
    build_score_cutoff_grid,
    freeze_risk_band_thresholds,
    pd_to_score,
    resolve_scaling_metadata,
    score_to_pd,
)
from credit_visable.utils import get_paths
from credit_visable.utils.reproducibility import build_run_manifest
from credit_visable.utils.reporting import build_report_summary_fields, to_builtin


GROUP_REVIEW_COLUMNS = ["age_band", "family_status_group", "region_rating_group"]
FINANCIAL_COLUMNS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


def _load_phase5_inputs(paths, settings) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    phase5_root = paths.data_processed / "xai_fairness"
    selection_path = phase5_root / "candidate_model_selection.json"
    summary_path = phase5_root / "summary.json"
    review_path = phase5_root / "validation_review_frame.csv"

    missing = [
        str(path)
        for path in [selection_path, summary_path, review_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Phase 5 artifacts missing: {missing}")

    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_review_frame = pd.read_csv(review_path)

    required_columns = {
        settings.id_column,
        settings.target_column,
        "candidate_predicted_pd",
        *GROUP_REVIEW_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(validation_review_frame.columns))
    if missing_columns:
        raise KeyError(f"validation_review_frame missing columns: {missing_columns}")
    return selection_payload, summary_payload, validation_review_frame


def _resolve_comparator(selection_payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    candidate_model = selection_payload.get("candidate_model") or {}
    candidate_feature_set = candidate_model.get("feature_set")
    proxy_audit_pair_family = str(selection_payload.get("proxy_audit_pair_family", "advanced"))
    matched_core_comparator = selection_payload.get("matched_core_comparator") or {}
    matched_proxy_comparator = selection_payload.get("matched_proxy_comparator") or {}

    comparator_feature_set = (
        "traditional_core"
        if candidate_feature_set == "traditional_plus_proxy"
        else "traditional_plus_proxy"
    )
    comparator_model = (
        matched_core_comparator
        if comparator_feature_set == "traditional_core"
        else matched_proxy_comparator
    )
    return comparator_model, comparator_feature_set, proxy_audit_pair_family


def _resolve_raw_score_frame(
    validation_review_frame: pd.DataFrame,
    selection_payload: dict[str, Any],
    settings,
) -> pd.DataFrame:
    comparator_model, comparator_feature_set, proxy_audit_pair_family = _resolve_comparator(
        selection_payload
    )
    candidate_model = selection_payload.get("candidate_model") or {}

    core_score_column = f"{proxy_audit_pair_family}_traditional_core_predicted_pd"
    proxy_score_column = f"{proxy_audit_pair_family}_traditional_plus_proxy_predicted_pd"
    comparator_pd_column = (
        core_score_column
        if comparator_feature_set == "traditional_core"
        else proxy_score_column
    )

    selected_columns = [
        settings.id_column,
        settings.target_column,
        "candidate_predicted_pd",
        comparator_pd_column,
        *GROUP_REVIEW_COLUMNS,
    ]
    score_frame = validation_review_frame[selected_columns].copy().rename(
        columns={
            "candidate_predicted_pd": "candidate_raw_pd",
            comparator_pd_column: "comparator_raw_pd",
            settings.target_column: "TARGET",
        }
    )
    score_frame["candidate_feature_set"] = candidate_model.get("feature_set")
    score_frame["candidate_model_family"] = candidate_model.get("model_family")
    score_frame["candidate_model_label"] = candidate_model.get("model_label")
    score_frame["comparator_feature_set"] = comparator_feature_set
    score_frame["comparator_model_family"] = comparator_model.get("model_family")
    score_frame["comparator_model_label"] = comparator_model.get("model_label")
    score_frame["proxy_audit_pair_family"] = proxy_audit_pair_family
    score_frame["TARGET"] = pd.to_numeric(score_frame["TARGET"], errors="coerce").fillna(0).astype(int)
    score_frame["candidate_raw_pd"] = pd.to_numeric(
        score_frame["candidate_raw_pd"], errors="coerce"
    ).fillna(0.5)
    score_frame["comparator_raw_pd"] = pd.to_numeric(
        score_frame["comparator_raw_pd"], errors="coerce"
    ).fillna(0.5)
    return score_frame


def _attach_financial_columns(score_frame: pd.DataFrame, paths, settings) -> pd.DataFrame:
    candidate_dir = paths.data_processed / "scorecard_cutoff" / "xgboost_traditional_plus_proxy"
    calibrated_path = candidate_dir / "calibrated_validation_scores.csv"
    if not calibrated_path.exists():
        return score_frame

    calibrated_frame = pd.read_csv(calibrated_path)
    available_columns = [
        column
        for column in [settings.id_column, *FINANCIAL_COLUMNS]
        if column in calibrated_frame.columns
    ]
    if len(available_columns) <= 1:
        return score_frame

    merged = score_frame.merge(
        calibrated_frame[available_columns].drop_duplicates(subset=[settings.id_column]),
        on=settings.id_column,
        how="left",
    )
    return merged


def _assign_risk_deciles(score_values: pd.Series) -> pd.Series:
    ranked = pd.Series(score_values).rank(method="first", ascending=True)
    deciles = pd.qcut(ranked, q=min(10, len(ranked)), labels=False, duplicates="drop")
    return (deciles.astype(int) + 1).rename("risk_decile")


def _summarize_score_deciles(score_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        score_frame.groupby("candidate_risk_decile", observed=False)
        .agg(
            count=("TARGET", "size"),
            bad_count=("TARGET", "sum"),
            actual_default_rate=("TARGET", "mean"),
            mean_calibrated_pd=("candidate_calibrated_pd", "mean"),
            mean_score=("candidate_score", "mean"),
            min_score=("candidate_score", "min"),
            max_score=("candidate_score", "max"),
        )
        .reset_index()
        .rename(columns={"candidate_risk_decile": "risk_decile"})
    )
    summary["population_share"] = summary["count"] / len(score_frame)
    summary = summary.sort_values("risk_decile").reset_index(drop=True)
    return summary


def _build_calibration_summary(score_frame: pd.DataFrame) -> pd.DataFrame:
    calibration = build_calibration_table(
        y_true=score_frame["TARGET"],
        calibrated_probabilities=score_frame["candidate_calibrated_pd"],
        bins=10,
    ).rename(
        columns={
            "bin_index": "calibration_bin",
            "predicted_mean": "mean_calibrated_pd",
            "observed_rate": "actual_default_rate",
        }
    )
    calibration["population_share"] = calibration["count"] / len(score_frame)
    calibration["bad_count"] = (
        calibration["actual_default_rate"] * calibration["count"]
    ).round().astype(int)
    calibration["calibration_gap"] = (
        calibration["actual_default_rate"] - calibration["mean_calibrated_pd"]
    )
    calibration["mean_score"] = (
        score_frame.sort_values("candidate_calibrated_pd")
        .groupby(pd.qcut(score_frame["candidate_calibrated_pd"].rank(method="first"), q=10, labels=False, duplicates="drop"), observed=False)["candidate_score"]
        .mean()
        .reset_index(drop=True)
    )
    calibration["brier_score"] = float(
        brier_score_loss(score_frame["TARGET"], score_frame["candidate_calibrated_pd"])
    )
    calibration["calibration_bin"] = calibration["calibration_bin"].astype(int)
    return calibration[
        [
            "calibration_bin",
            "count",
            "population_share",
            "bad_count",
            "actual_default_rate",
            "mean_calibrated_pd",
            "calibration_gap",
            "mean_score",
            "brier_score",
        ]
    ]


def _build_group_policy_summary(score_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protected_attribute in GROUP_REVIEW_COLUMNS:
        grouped = (
            score_frame.groupby(protected_attribute, dropna=False, observed=False)
            .agg(
                count=("TARGET", "size"),
                bad_count=("TARGET", "sum"),
                actual_default_rate=("TARGET", "mean"),
                mean_calibrated_pd=("candidate_calibrated_pd", "mean"),
                mean_score=("candidate_score", "mean"),
                approval_rate=("candidate_final_decision", lambda values: float((values == "approve").mean())),
                review_rate=("candidate_final_decision", lambda values: float((values == "review").mean())),
                reject_rate=("candidate_final_decision", lambda values: float((values == "reject").mean())),
                approved_count=("candidate_final_decision", lambda values: int((values == "approve").sum())),
            )
            .reset_index()
            .rename(columns={protected_attribute: "group"})
        )
        approved_bad_rates = []
        for group_value in grouped["group"]:
            mask = (score_frame[protected_attribute] == group_value) & (
                score_frame["candidate_final_decision"] == "approve"
            )
            approved_bad_rates.append(
                float(score_frame.loc[mask, "TARGET"].mean()) if mask.any() else np.nan
            )
        grouped["approved_bad_rate"] = approved_bad_rates
        grouped["population_share"] = grouped["count"] / len(score_frame)
        grouped.insert(0, "protected_attribute", protected_attribute)
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_migration_matrix(
    score_frame: pd.DataFrame,
    from_column: str,
    to_column: str,
) -> pd.DataFrame:
    return (
        score_frame.groupby([from_column, to_column], observed=False)
        .size()
        .reset_index(name="count")
    )


def refresh_phase6_artifacts(
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Refresh calibrated Phase 6 artifacts and return in-memory outputs."""

    settings = load_settings()
    scorecard_settings = load_scorecard_settings()
    paths = get_paths()
    selection_payload, phase5_summary_payload, validation_review_frame = _load_phase5_inputs(
        paths, settings
    )
    candidate_model = selection_payload.get("candidate_model") or {}
    comparator_model, comparator_feature_set, _ = _resolve_comparator(selection_payload)

    score_frame = _resolve_raw_score_frame(validation_review_frame, selection_payload, settings)
    score_frame = _attach_financial_columns(score_frame, paths, settings)

    candidate_calibrator = fit_platt_calibrator(
        score_frame["candidate_raw_pd"], score_frame["TARGET"]
    )
    comparator_calibrator = fit_platt_calibrator(
        score_frame["comparator_raw_pd"], score_frame["TARGET"]
    )

    score_frame["candidate_calibrated_pd"] = apply_platt_calibrator(
        score_frame["candidate_raw_pd"], candidate_calibrator
    )
    score_frame["comparator_calibrated_pd"] = apply_platt_calibrator(
        score_frame["comparator_raw_pd"], comparator_calibrator
    )

    score_transform_meta = resolve_scaling_metadata(scorecard_settings)
    score_frame["candidate_score"] = pd_to_score(
        score_frame["candidate_calibrated_pd"], score_transform_meta, name="candidate_score"
    )
    score_frame["comparator_score"] = pd_to_score(
        score_frame["comparator_calibrated_pd"], score_transform_meta, name="comparator_score"
    )
    score_frame["candidate_score_minus_comparator"] = (
        score_frame["candidate_score"] - score_frame["comparator_score"]
    )

    candidate_bands, risk_band_summary = assign_hybrid_risk_bands(
        frame=score_frame,
        calibrated_pd_column="candidate_calibrated_pd",
        score_column="candidate_score",
        risk_band_config=scorecard_settings.risk_bands,
        target_column="TARGET",
        name="candidate_risk_band",
    )
    comparator_bands, _ = assign_hybrid_risk_bands(
        frame=score_frame,
        calibrated_pd_column="comparator_calibrated_pd",
        score_column="comparator_score",
        risk_band_config=scorecard_settings.risk_bands,
        target_column="TARGET",
        name="comparator_risk_band",
    )
    score_frame["candidate_risk_band"] = candidate_bands
    score_frame["comparator_risk_band"] = comparator_bands
    score_frame["candidate_risk_decile"] = _assign_risk_deciles(score_frame["candidate_score"])
    score_frame["comparator_risk_decile"] = _assign_risk_deciles(score_frame["comparator_score"])

    score_decile_summary = _summarize_score_deciles(score_frame)
    calibration_summary = _build_calibration_summary(score_frame)
    risk_band_table = build_operational_risk_band_table(risk_band_summary)

    cutoff_grid = build_score_cutoff_grid(score_frame["candidate_score"], step=5)
    economics_ready = all(column in score_frame.columns for column in FINANCIAL_COLUMNS)
    if economics_ready:
        unit_economics_frame = build_unit_economics_frame(
            score_frame,
            scorecard_settings.unit_economics,
        )
        unit_economics_summary = build_unit_economics_summary(unit_economics_frame)
        cutoff_sweep = evaluate_cutoff_curve(
            frame=score_frame,
            score_column="candidate_score",
            calibrated_pd_column="candidate_calibrated_pd",
            unit_economics_frame=unit_economics_frame,
            cutoff_grid=cutoff_grid,
            review_buffer_points=scorecard_settings.cutoff_strategy.review_buffer_points,
            target_column="TARGET",
            guardrails=scorecard_settings.cutoff_strategy.guardrails,
            scenario_name="final_policy",
        )
    else:
        unit_economics_summary = pd.DataFrame()
        fallback_rows = []
        for cutoff in cutoff_grid:
            approve_min_score = float(cutoff + scorecard_settings.cutoff_strategy.review_buffer_points)
            review_min_score = float(cutoff - scorecard_settings.cutoff_strategy.review_buffer_points)
            decisions = apply_cutoff_policy(
                score_frame["candidate_score"],
                approve_min_score=approve_min_score,
                review_min_score=review_min_score,
            )
            approve_mask = decisions == "approve"
            review_mask = decisions == "review"
            reject_mask = decisions == "reject"
            approval_rate = float(approve_mask.mean())
            review_rate = float(review_mask.mean())
            approved_bad_rate = float(score_frame.loc[approve_mask, "TARGET"].mean()) if approve_mask.any() else np.nan
            passes_guardrails = bool(
                approve_mask.any()
                and approval_rate >= scorecard_settings.cutoff_strategy.guardrails.min_approval_rate
                and review_rate <= scorecard_settings.cutoff_strategy.guardrails.max_manual_review_share
                and (np.isnan(approved_bad_rate) or approved_bad_rate <= scorecard_settings.cutoff_strategy.guardrails.max_approved_book_bad_rate)
            )
            fallback_rows.append(
                {
                    "scenario_name": "final_policy",
                    "score_cutoff": int(cutoff),
                    "approve_min_score": approve_min_score,
                    "reject_below_score": review_min_score,
                    "approval_rate": approval_rate,
                    "review_rate": review_rate,
                    "reject_rate": float(reject_mask.mean()),
                    "approved_count": int(approve_mask.sum()),
                    "review_count": int(review_mask.sum()),
                    "rejected_count": int(reject_mask.sum()),
                    "approved_book_mean_pd": float(score_frame.loc[approve_mask, "candidate_calibrated_pd"].mean()) if approve_mask.any() else np.nan,
                    "actual_approved_bad_rate": approved_bad_rate,
                    "total_expected_value": float(approval_rate - (approved_bad_rate if not np.isnan(approved_bad_rate) else 0.0)),
                    "expected_value_per_applicant": float(approval_rate - (approved_bad_rate if not np.isnan(approved_bad_rate) else 0.0)),
                    "passes_guardrails": passes_guardrails,
                }
            )
        cutoff_sweep = pd.DataFrame(fallback_rows)

    selected_cutoff = select_optimal_cutoff(cutoff_sweep)
    final_cutoff = float(selected_cutoff["approve_min_score"])
    final_review_cutoff = float(selected_cutoff["reject_below_score"])
    score_frame["candidate_final_decision"] = apply_cutoff_policy(
        score_frame["candidate_score"],
        approve_min_score=final_cutoff,
        review_min_score=final_review_cutoff,
        name="candidate_final_decision",
    )
    score_frame["comparator_final_decision"] = apply_cutoff_policy(
        score_frame["comparator_score"],
        approve_min_score=final_cutoff,
        review_min_score=final_review_cutoff,
        name="comparator_final_decision",
    )

    final_policy_summary = {
        "cutoff_anchor_score": int(selected_cutoff["score_cutoff"]),
        "final_cutoff": int(final_cutoff),
        "final_review_cutoff": int(final_review_cutoff),
        "approval_rate": float(selected_cutoff["approval_rate"]),
        "review_rate": float(selected_cutoff["review_rate"]),
        "reject_rate": float(selected_cutoff["reject_rate"]),
        "approved_count": int(selected_cutoff["approved_count"]),
        "review_count": int(selected_cutoff["review_count"]),
        "rejected_count": int(selected_cutoff["rejected_count"]),
        "approved_book_mean_pd": float(selected_cutoff["approved_book_mean_pd"]),
        "actual_approved_bad_rate": (
            None
            if pd.isna(selected_cutoff.get("actual_approved_bad_rate"))
            else float(selected_cutoff["actual_approved_bad_rate"])
        ),
        "total_expected_value": float(selected_cutoff["total_expected_value"]),
        "expected_value_per_applicant": float(selected_cutoff["expected_value_per_applicant"]),
        "passes_guardrails": bool(selected_cutoff["passes_guardrails"]),
    }

    final_policy_group_summary = _build_group_policy_summary(score_frame)
    score_migration_matrix = _build_migration_matrix(
        score_frame, "comparator_risk_band", "candidate_risk_band"
    ).rename(columns={"comparator_risk_band": "comparator_risk_band", "candidate_risk_band": "candidate_risk_band"})
    decision_migration_matrix = _build_migration_matrix(
        score_frame, "comparator_final_decision", "candidate_final_decision"
    )
    calibrated_validation_scores = score_frame[
        [
            settings.id_column,
            "TARGET",
            "candidate_raw_pd",
            *[column for column in FINANCIAL_COLUMNS if column in score_frame.columns],
            "candidate_calibrated_pd",
            "candidate_score",
            "candidate_risk_band",
            "candidate_final_decision",
        ]
    ].rename(
        columns={
            "candidate_raw_pd": "raw_pd",
            "candidate_calibrated_pd": "calibrated_pd",
            "candidate_score": "score",
            "candidate_risk_band": "risk_band",
            "candidate_final_decision": "final_decision",
        }
    )

    profit_curve = cutoff_sweep[
        [
            "score_cutoff",
            "approval_rate",
            "reject_rate",
            "actual_approved_bad_rate",
            "total_expected_value",
            "expected_value_per_applicant",
        ]
    ].rename(
        columns={
            "actual_approved_bad_rate": "approved_bad_rate",
            "total_expected_value": "total_profit",
            "expected_value_per_applicant": "profit_per_applicant",
        }
    )
    optimal_profit_cutoff = {
        "score_cutoff": int(selected_cutoff["score_cutoff"]),
        "approval_rate": float(selected_cutoff["approval_rate"]),
        "reject_rate": float(selected_cutoff["reject_rate"]),
        "approved_bad_rate": (
            None
            if pd.isna(selected_cutoff.get("actual_approved_bad_rate"))
            else float(selected_cutoff["actual_approved_bad_rate"])
        ),
        "total_profit": float(selected_cutoff["total_expected_value"]),
        "profit_per_applicant": float(selected_cutoff["expected_value_per_applicant"]),
    }
    final_pd_threshold = float(score_to_pd([final_cutoff], score_transform_meta).iloc[0])
    optimal_profit_fairness_summary = build_group_fairness_metric_summary(
        frame=score_frame,
        target_column="TARGET",
        score_column="candidate_calibrated_pd",
        group_specs=[
            {
                "protected_attribute": column,
                "source_column": column,
                "group_column": column,
                "kind": "identity",
            }
            for column in GROUP_REVIEW_COLUMNS
        ],
        threshold=final_pd_threshold,
    )

    validation_checks = pd.DataFrame(
        [
            {
                "check": "calibrated_pd_mean_close_to_target_rate",
                "passed": bool(
                    abs(score_frame["candidate_calibrated_pd"].mean() - score_frame["TARGET"].mean()) <= 0.01
                ),
                "details": "mean(calibrated_pd) tracks mean(TARGET)",
            },
            {
                "check": "risk_band_actual_rate_monotonic",
                "passed": bool(risk_band_summary["actual_default_rate"].is_monotonic_increasing),
                "details": "actual default rate increases from A to E",
            },
            {
                "check": "risk_band_calibrated_pd_monotonic",
                "passed": bool(risk_band_summary["mean_calibrated_pd"].is_monotonic_increasing),
                "details": "mean calibrated PD increases from A to E",
            },
            {
                "check": "abc_population_majority",
                "passed": bool(
                    float(
                        risk_band_summary.loc[
                            risk_band_summary["risk_band"].isin(["A", "B", "C"]),
                            "population_share",
                        ].sum()
                    )
                    > 0.5
                ),
                "details": "A/B/C cover the majority of the applicant pool",
            },
            {
                "check": "final_cutoff_passes_guardrails",
                "passed": bool(final_policy_summary["passes_guardrails"]),
                "details": "selected final policy passes configured guardrails",
            },
        ]
    )

    resolved_output_dir = output_dir or (
        paths.data_processed
        / "scorecard_cutoff"
        / str(candidate_model.get("model_label", "candidate_model")).replace("/", "_")
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    score_frame.to_csv(resolved_output_dir / "score_frame.csv", index=False)
    calibrated_validation_scores.to_csv(
        resolved_output_dir / "calibrated_validation_scores.csv", index=False
    )
    score_decile_summary.to_csv(resolved_output_dir / "score_decile_summary.csv", index=False)
    risk_band_summary.to_csv(resolved_output_dir / "risk_band_summary.csv", index=False)
    risk_band_table.to_csv(resolved_output_dir / "risk_band_table.csv", index=False)
    calibration_summary.to_csv(resolved_output_dir / "calibration_summary.csv", index=False)
    cutoff_sweep.to_csv(resolved_output_dir / "cutoff_sweep.csv", index=False)
    cutoff_sweep.to_csv(resolved_output_dir / "cutoff_expected_value_curve.csv", index=False)
    profit_curve.to_csv(resolved_output_dir / "profit_curve.csv", index=False)
    final_policy_group_summary.to_csv(
        resolved_output_dir / "final_policy_group_summary.csv", index=False
    )
    pd.DataFrame([final_policy_summary]).to_csv(
        resolved_output_dir / "final_policy_summary.csv", index=False
    )
    score_migration_matrix.to_csv(
        resolved_output_dir / "score_migration_matrix.csv", index=False
    )
    decision_migration_matrix.to_csv(
        resolved_output_dir / "decision_migration_matrix.csv", index=False
    )
    optimal_profit_fairness_summary.to_csv(
        resolved_output_dir / "optimal_profit_fairness_summary.csv", index=False
    )
    if not unit_economics_summary.empty:
        unit_economics_summary.to_csv(
            resolved_output_dir / "unit_economics_summary.csv", index=False
        )
    (resolved_output_dir / "profit_assumptions.json").write_text(
        json.dumps(to_builtin(scorecard_settings.unit_economics.assumptions), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (resolved_output_dir / "score_transform_meta.json").write_text(
        json.dumps(to_builtin(score_transform_meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (resolved_output_dir / "scorecard_config.json").write_text(
        json.dumps(to_builtin(asdict(scorecard_settings)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (resolved_output_dir / "optimal_profit_cutoff.json").write_text(
        json.dumps(to_builtin(optimal_profit_cutoff), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (resolved_output_dir / "final_policy_summary.json").write_text(
        json.dumps(to_builtin(final_policy_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dump_joblib(candidate_calibrator, resolved_output_dir / "platt_calibrator.joblib")

    summary_path = resolved_output_dir / "summary.json"
    existing_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    for obsolete_key in [
        "placeholder_pdo_metadata",
        "score_cutoff_grid",
        "policy_scenarios",
        "balanced_cutoff_profit",
        "policy_group_summary",
    ]:
        existing_summary.pop(obsolete_key, None)
    figure_paths = {
        path.stem: path
        for path in sorted((paths.reports_figures_v2).glob(f"phase6_{resolved_output_dir.name}_*.png"))
    }
    existing_summary.update(
        build_report_summary_fields(
            headline="Phase 6 now uses calibrated PD, hybrid operational bands, and a single final strategy.",
            key_findings=[
                f"Raw candidate PD mean = {score_frame['candidate_raw_pd'].mean():.4f}; calibrated PD mean = {score_frame['candidate_calibrated_pd'].mean():.4f}.",
                f"Final approve cutoff = {final_policy_summary['final_cutoff']}; final review cutoff = {final_policy_summary['final_review_cutoff']}.",
                f"A/B/C population share = {risk_band_summary.loc[risk_band_summary['risk_band'].isin(['A', 'B', 'C']), 'population_share'].sum():.1%}.",
            ],
            business_implications=[
                "Upstream Phase 4/5 ranking conclusions remain usable; the correction is in Phase 6 probability-to-policy translation.",
                "Risk bands are now operationally distributed instead of collapsing most applicants into the worst bucket.",
                "A single final cutoff replaces named scenario storytelling in the core report outputs.",
            ],
            figure_paths=figure_paths,
        )
    )
    existing_summary.update(
        {
            "candidate_model": candidate_model,
            "comparator_model": comparator_model,
            "proxy_audit_pair_family": selection_payload.get("proxy_audit_pair_family"),
            "score_settings": to_builtin(score_transform_meta),
            "calibration_fitted": True,
            "raw_pd_mean": float(score_frame["candidate_raw_pd"].mean()),
            "calibrated_pd_mean": float(score_frame["candidate_calibrated_pd"].mean()),
            "calibration_method": scorecard_settings.calibration.method,
            "band_construction_method": scorecard_settings.risk_bands.construction,
            "final_cutoff": int(final_policy_summary["final_cutoff"]),
            "final_review_cutoff": int(final_policy_summary["final_review_cutoff"]),
            "risk_band_summary": risk_band_summary.to_dict(orient="records"),
            "final_policy_summary": final_policy_summary,
            "optimal_profit_cutoff": optimal_profit_cutoff,
            "optimal_profit_fairness_summary": optimal_profit_fairness_summary.to_dict(orient="records"),
            "validation_checks": validation_checks.to_dict(orient="records"),
            "phase5_headline": phase5_summary_payload.get("headline"),
        }
    )
    summary_path.write_text(
        json.dumps(to_builtin(existing_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for obsolete_name in ["policy_scenarios.csv", "policy_group_summary.csv"]:
        obsolete_path = resolved_output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    return {
        "output_dir": resolved_output_dir,
        "score_frame": score_frame,
        "score_decile_summary": score_decile_summary,
        "risk_band_summary": risk_band_summary,
        "risk_band_table": risk_band_table,
        "calibration_summary": calibration_summary,
        "cutoff_sweep": cutoff_sweep,
        "final_policy_summary": final_policy_summary,
        "final_policy_group_summary": final_policy_group_summary,
        "score_migration_matrix": score_migration_matrix,
        "decision_migration_matrix": decision_migration_matrix,
        "validation_checks": validation_checks,
    }


def _resolve_governed_group_specs() -> list[dict[str, Any]]:
    return [
        {
            "protected_attribute": "age_band",
            "source_column": "age_band",
            "group_column": "age_band",
            "kind": "identity",
        },
        {
            "protected_attribute": "family_status_group",
            "source_column": "family_status_group",
            "group_column": "family_status_group",
            "kind": "identity",
        },
        {
            "protected_attribute": "region_rating_group",
            "source_column": "region_rating_group",
            "group_column": "region_rating_group",
            "kind": "identity",
        },
    ]


def _prepare_governed_review_frame(
    engineered_frame: pd.DataFrame,
    *,
    id_series: pd.Series,
    id_column: str,
) -> pd.DataFrame:
    review_columns = [
        id_column,
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "NAME_FAMILY_STATUS",
        "REGION_RATING_CLIENT",
    ]
    available_columns = [column_name for column_name in review_columns if column_name in engineered_frame.columns]
    review_frame = pd.DataFrame({id_column: id_series.reset_index(drop=True)}).merge(
        engineered_frame[available_columns].copy(),
        on=id_column,
        how="inner",
    ).reset_index(drop=True)
    if "DAYS_BIRTH" in review_frame.columns:
        review_frame["age_band"] = derive_age_band_from_days_birth(review_frame["DAYS_BIRTH"])
    if "NAME_FAMILY_STATUS" in review_frame.columns:
        review_frame["family_status_group"] = collapse_rare_categories(
            review_frame["NAME_FAMILY_STATUS"],
            top_n=5,
        )
    if "REGION_RATING_CLIENT" in review_frame.columns:
        review_frame["region_rating_group"] = (
            review_frame["REGION_RATING_CLIENT"].astype("object").where(
                review_frame["REGION_RATING_CLIENT"].notna(),
                "Missing",
            )
        )
    review_frame["row_position"] = np.arange(len(review_frame))
    return review_frame


def _build_reason_code_table(
    local_case_explanations: pd.DataFrame,
    *,
    top_n_features: int = 4,
) -> pd.DataFrame:
    if local_case_explanations.empty:
        return pd.DataFrame(
            columns=[
                "case_role",
                "row_position",
                "feature_rank",
                "raw_feature_name",
                "proxy_family",
                "contribution",
            ]
        )
    return (
        local_case_explanations.sort_values(
            ["case_role", "abs_contribution"],
            ascending=[True, False],
        )
        .groupby("case_role", group_keys=False)
        .head(top_n_features)
        .reset_index(drop=True)
    )[
        [
            "case_role",
            "row_position",
            "feature_rank",
            "raw_feature_name",
            "proxy_family",
            "contribution",
        ]
    ]


def run_governed_application_pipeline(
    *,
    feature_set_name: str = "traditional_plus_proxy",
    output_dir: Path | None = None,
    shap_sample_size: int = 512,
    raw_frame: pd.DataFrame | None = None,
    feature_engineering_options: ApplicationFeatureEngineeringOptions | None = None,
    preprocessing_options: PreprocessingOptions | None = None,
    split_options: GovernedSplitOptions | None = None,
    feature_review_options: FeatureReviewOptions | None = None,
    training_options: GovernedTreeTrainingOptions | None = None,
) -> dict[str, Any]:
    """Run the governed application-only champion pipeline end to end."""

    settings = load_settings()
    scorecard_settings = load_scorecard_settings()
    paths = get_paths()
    resolved_raw_frame = raw_frame if raw_frame is not None else load_application_train(settings=settings)
    resolved_feature_engineering_options = (
        feature_engineering_options or ApplicationFeatureEngineeringOptions()
    )
    resolved_preprocessing_options = preprocessing_options or PreprocessingOptions(
        clip_quantiles=(0.01, 0.99),
        rare_category_min_frequency=0.01,
    )
    resolved_split_options = split_options or GovernedSplitOptions(
        random_state=settings.random_state
    )
    resolved_feature_review_options = feature_review_options or FeatureReviewOptions(
        random_state=settings.random_state
    )
    resolved_training_options = training_options or GovernedTreeTrainingOptions(
        random_state=settings.random_state
    )
    engineered_frame = engineer_application_features(
        resolved_raw_frame,
        options=resolved_feature_engineering_options,
    )
    artifacts = prepare_governed_preprocessing_artifacts(
        resolved_raw_frame,
        feature_set_name=feature_set_name,
        target_column=settings.target_column,
        id_column=settings.id_column,
        preprocessing_options=resolved_preprocessing_options,
        split_options=resolved_split_options,
        feature_engineering_options=resolved_feature_engineering_options,
        feature_review_options=resolved_feature_review_options,
    )

    baseline_model = train_logistic_baseline(
        artifacts.X_dev,
        artifacts.y_dev,
        random_state=settings.random_state,
    )
    baseline_test_pd = baseline_model.predict_proba(artifacts.X_test)[:, 1]
    baseline_test_metrics = evaluate_binary_classifier(
        artifacts.y_test,
        baseline_test_pd,
    )

    training_result = train_governed_tree_model(
        artifacts.X_dev,
        artifacts.y_dev,
        options=resolved_training_options,
    )
    champion_model = training_result["model"]
    dev_raw_pd = np.asarray(champion_model.predict_proba(artifacts.X_dev))[:, 1]
    calibration_raw_pd = np.asarray(champion_model.predict_proba(artifacts.X_calibration))[:, 1]
    test_raw_pd = np.asarray(champion_model.predict_proba(artifacts.X_test))[:, 1]

    raw_test_metrics = evaluate_binary_classifier(
        artifacts.y_test,
        test_raw_pd,
    )
    calibrator = fit_platt_calibrator(calibration_raw_pd, artifacts.y_calibration)
    dev_calibrated_pd = apply_platt_calibrator(dev_raw_pd, calibrator)
    calibration_calibrated_pd = apply_platt_calibrator(calibration_raw_pd, calibrator)
    test_calibrated_pd = apply_platt_calibrator(test_raw_pd, calibrator)
    calibrated_test_metrics = evaluate_binary_classifier(
        artifacts.y_test,
        test_calibrated_pd,
    )

    score_transform_meta = resolve_scaling_metadata(scorecard_settings)
    dev_score = pd_to_score(pd.Series(dev_calibrated_pd), score_transform_meta, name="score")
    calibration_score = pd_to_score(
        pd.Series(calibration_calibrated_pd),
        score_transform_meta,
        name="score",
    )
    test_score = pd_to_score(pd.Series(test_calibrated_pd), score_transform_meta, name="score")

    frozen_risk_band_table = freeze_risk_band_thresholds(
        pd.DataFrame(
            {
                "calibrated_pd": dev_calibrated_pd,
                "score": dev_score,
            }
        ),
        calibrated_pd_column="calibrated_pd",
        score_column="score",
        risk_band_config=scorecard_settings.risk_bands,
    )
    test_policy_frame = _prepare_governed_review_frame(
        engineered_frame,
        id_series=artifacts.test_ids if artifacts.test_ids is not None else pd.Series(dtype=int),
        id_column=settings.id_column,
    )
    dev_policy_frame = _prepare_governed_review_frame(
        engineered_frame,
        id_series=artifacts.dev_ids if artifacts.dev_ids is not None else pd.Series(dtype=int),
        id_column=settings.id_column,
    )
    test_policy_frame["raw_pd"] = test_raw_pd
    test_policy_frame["calibrated_pd"] = test_calibrated_pd
    test_policy_frame["score"] = test_score.to_numpy()
    dev_policy_frame["calibrated_pd"] = dev_calibrated_pd
    dev_policy_frame["score"] = dev_score.to_numpy()

    assigned_bands, risk_band_summary = assign_frozen_risk_bands(
        test_policy_frame,
        score_column="score",
        calibrated_pd_column="calibrated_pd",
        band_threshold_table=frozen_risk_band_table,
        target_column="TARGET",
        name="risk_band",
    )
    test_policy_frame["risk_band"] = assigned_bands
    calibration_summary = build_calibration_table(
        artifacts.y_test,
        test_calibrated_pd,
        bins=10,
    )

    cutoff_grid = build_score_cutoff_grid(test_policy_frame["score"], step=5)
    unit_economics_frame = build_unit_economics_frame(
        test_policy_frame,
        scorecard_settings.unit_economics,
    )
    cutoff_sweep = evaluate_cutoff_curve(
        frame=test_policy_frame,
        score_column="score",
        calibrated_pd_column="calibrated_pd",
        unit_economics_frame=unit_economics_frame,
        cutoff_grid=cutoff_grid,
        review_buffer_points=scorecard_settings.cutoff_strategy.review_buffer_points,
        target_column="TARGET",
        guardrails=scorecard_settings.cutoff_strategy.guardrails,
        scenario_name="holdout_policy",
    )
    selected_cutoff = select_optimal_cutoff(cutoff_sweep)
    final_cutoff = float(selected_cutoff["approve_min_score"])
    final_review_cutoff = float(selected_cutoff["reject_below_score"])
    test_policy_frame["final_decision"] = apply_cutoff_policy(
        test_policy_frame["score"],
        approve_min_score=final_cutoff,
        review_min_score=final_review_cutoff,
        name="final_decision",
    )
    final_pd_threshold = float(score_to_pd([final_cutoff], score_transform_meta).iloc[0])
    fairness_metric_summary = build_group_fairness_metric_summary(
        frame=test_policy_frame,
        target_column="TARGET",
        score_column="calibrated_pd",
        group_specs=_resolve_governed_group_specs(),
        threshold=final_pd_threshold,
    )
    policy_group_summary = _build_group_policy_summary(
        test_policy_frame.rename(
            columns={
                "calibrated_pd": "candidate_calibrated_pd",
                "score": "candidate_score",
                "final_decision": "candidate_final_decision",
            }
        )
    )
    monitoring = build_monitoring_baseline(
        reference_frame=dev_policy_frame,
        comparison_frame=test_policy_frame,
        target_column="TARGET",
        calibrated_pd_column="calibrated_pd",
        score_column="score",
        group_specs=_resolve_governed_group_specs(),
        threshold=final_pd_threshold,
    )

    selected_rows = select_local_explanation_rows(
        validation_frame=test_policy_frame,
        score_column="calibrated_pd",
        target_column="TARGET",
        threshold=final_pd_threshold,
        id_column=settings.id_column,
        num_cases=3,
    )
    shap_summary = compute_shap_contribution_summary(
        champion_model,
        artifacts.X_test,
        artifacts.feature_names,
        raw_feature_candidates=artifacts.selected_feature_columns,
        sample_size=min(int(shap_sample_size), int(artifacts.X_test.shape[0])),
        random_state=settings.random_state,
    )
    local_shap = compute_shap_local_explanations(
        champion_model,
        artifacts.X_test,
        artifacts.feature_names,
        selected_rows=selected_rows,
        raw_feature_candidates=artifacts.selected_feature_columns,
        top_n_features=6,
    )
    reason_codes = _build_reason_code_table(local_shap["local_case_explanations"])

    model_label = f"governed_xgboost_{feature_set_name}"
    resolved_output_dir = output_dir or (paths.data_processed / "ds_audit_refactor" / model_label)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir = resolved_output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    save_governed_preprocessing_artifacts(artifacts, output_dir=preprocessing_dir)

    dump_joblib(champion_model, resolved_output_dir / "champion_model.joblib")
    dump_joblib(baseline_model, resolved_output_dir / "baseline_model.joblib")
    dump_joblib(calibrator, resolved_output_dir / "platt_calibrator.joblib")
    training_result["cv_results"].to_csv(resolved_output_dir / "cv_results.csv", index=False)
    pd.DataFrame(
        [
            {"model": "logistic_baseline", **baseline_test_metrics},
            {"model": "xgboost_raw", **raw_test_metrics},
            {"model": "xgboost_calibrated", **calibrated_test_metrics},
        ]
    ).to_csv(resolved_output_dir / "metrics_comparison.csv", index=False)
    pd.DataFrame(
        {
            settings.id_column: artifacts.test_ids,
            "TARGET": artifacts.y_test,
            "raw_pd": test_raw_pd,
            "calibrated_pd": test_calibrated_pd,
            "score": test_score.to_numpy(),
            "risk_band": test_policy_frame["risk_band"],
            "final_decision": test_policy_frame["final_decision"],
        }
    ).to_csv(resolved_output_dir / "holdout_scores.csv", index=False)
    frozen_risk_band_table.to_csv(resolved_output_dir / "frozen_risk_band_table.csv", index=False)
    risk_band_summary.to_csv(resolved_output_dir / "holdout_risk_band_summary.csv", index=False)
    calibration_summary.to_csv(resolved_output_dir / "holdout_calibration_summary.csv", index=False)
    cutoff_sweep.to_csv(resolved_output_dir / "holdout_cutoff_sweep.csv", index=False)
    fairness_metric_summary.to_csv(resolved_output_dir / "holdout_fairness_metric_summary.csv", index=False)
    policy_group_summary.to_csv(resolved_output_dir / "holdout_policy_group_summary.csv", index=False)
    monitoring["fairness_drift"].to_csv(resolved_output_dir / "monitoring_fairness_drift.csv", index=False)
    pd.DataFrame([monitoring["summary"]]).to_csv(
        resolved_output_dir / "monitoring_summary.csv",
        index=False,
    )
    shap_summary["raw_feature_contributions"].to_csv(
        resolved_output_dir / "shap_raw_feature_contributions.csv",
        index=False,
    )
    shap_summary["proxy_family_contributions"].to_csv(
        resolved_output_dir / "shap_proxy_family_contributions.csv",
        index=False,
    )
    local_shap["local_case_explanations"].to_csv(
        resolved_output_dir / "shap_local_case_explanations.csv",
        index=False,
    )
    reason_codes.to_csv(resolved_output_dir / "adverse_action_reason_codes.csv", index=False)

    run_manifest = build_run_manifest(
        raw_frame=resolved_raw_frame,
        config_snapshot={
            "settings": asdict(settings),
            "scorecard_settings": asdict(scorecard_settings),
            "feature_set_name": feature_set_name,
            "feature_engineering_options": asdict(resolved_feature_engineering_options),
            "preprocessing_options": asdict(resolved_preprocessing_options),
            "split_options": asdict(resolved_split_options),
            "feature_review_options": asdict(resolved_feature_review_options),
            "training_options": asdict(resolved_training_options),
        },
        split_hashes={
            "development": artifacts.split_manifest["development_id_hash"],
            "calibration": artifacts.split_manifest["calibration_id_hash"],
            "test": artifacts.split_manifest["test_id_hash"],
        },
        model_manifest=training_result["training_manifest"],
        cwd=paths.root,
    )
    (resolved_output_dir / "run_manifest.json").write_text(
        json.dumps(to_builtin(run_manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "headline": "Governed application-only champion pipeline completed.",
        "feature_set_name": feature_set_name,
        "development_rows": artifacts.split_manifest["development_rows"],
        "calibration_rows": artifacts.split_manifest["calibration_rows"],
        "test_rows": artifacts.split_manifest["test_rows"],
        "baseline_test_metrics": baseline_test_metrics,
        "raw_test_metrics": raw_test_metrics,
        "calibrated_test_metrics": calibrated_test_metrics,
        "final_cutoff": int(final_cutoff),
        "final_review_cutoff": int(final_review_cutoff),
        "oot_status": "unavailable_without_application_timestamp",
        "training_manifest": training_result["training_manifest"],
        "monitoring_summary": monitoring["summary"],
    }
    (resolved_output_dir / "summary.json").write_text(
        json.dumps(to_builtin(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_dir": resolved_output_dir,
        "summary": summary,
        "metrics_comparison": pd.read_csv(resolved_output_dir / "metrics_comparison.csv"),
        "frozen_risk_band_table": frozen_risk_band_table,
        "fairness_metric_summary": fairness_metric_summary,
        "policy_group_summary": policy_group_summary,
        "monitoring_summary": monitoring["summary"],
    }
