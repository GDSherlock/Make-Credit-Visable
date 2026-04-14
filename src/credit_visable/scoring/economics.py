"""Data-driven unit economics and cutoff optimization helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def _to_mapping(value: Any) -> dict[str, Any]:
    """Normalize config objects into plain mappings."""

    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Coerce numeric input and fill missing values with a stable fallback."""

    return pd.to_numeric(series, errors="coerce").fillna(fill_value).astype(float)


def build_unit_economics_frame(
    frame: pd.DataFrame,
    unit_economics_config: Any,
    scenario_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build borrower-level economics terms from scorecard config."""

    config_mapping = _to_mapping(unit_economics_config)
    data_drivers = config_mapping.get("data_drivers", {})
    assumptions = {
        **config_mapping.get("assumptions", {}),
        **(scenario_overrides or {}),
    }
    cost_multiplier = float(assumptions.pop("cost_multiplier", 1.0))

    principal = _safe_numeric(frame[data_drivers["principal"]], fill_value=0.0).clip(lower=0.0)
    payment = _safe_numeric(frame[data_drivers["payment"]], fill_value=np.nan)
    annual_income = _safe_numeric(frame[data_drivers["annual_income"]], fill_value=np.nan)

    if data_drivers["goods_price"] in frame.columns:
        goods_price = _safe_numeric(frame[data_drivers["goods_price"]], fill_value=np.nan)
    else:
        goods_price = pd.Series(np.nan, index=frame.index, dtype=float)

    goods_price = goods_price.where(goods_price > 0.0, principal)
    monthly_income_proxy = (annual_income / 12.0).where(annual_income > 0.0, np.nan)
    term_proxy_months = (principal / payment).where(payment > 0.0, np.nan).fillna(24.0).clip(6.0, 48.0)
    payment_burden = (payment / monthly_income_proxy).where(
        (payment > 0.0) & (monthly_income_proxy > 0.0),
        np.nan,
    ).fillna(0.20).clip(0.05, 0.60)
    advance_ratio = (principal / goods_price).where(goods_price > 0.0, np.nan).fillna(1.0).clip(0.80, 1.50)

    base_lgd = float(assumptions["base_lgd"])
    burden_lgd_slope = float(assumptions["burden_lgd_slope"])
    advance_lgd_slope = float(assumptions["advance_lgd_slope"])
    lgd = (
        base_lgd
        + burden_lgd_slope * np.maximum(payment_burden - 0.20, 0.0)
        + advance_lgd_slope * np.maximum(advance_ratio - 1.0, 0.0)
    ).clip(0.65, 0.90)

    net_margin_rate_annual = float(assumptions["net_margin_rate_annual"])
    ead_rate = float(assumptions["ead_rate"])
    reject_good_capture = float(assumptions["reject_good_capture"])
    reject_bad_loss_avoidance = float(assumptions["reject_bad_loss_avoidance"])

    approve_good = principal * net_margin_rate_annual * (term_proxy_months / 12.0)
    approve_bad = -(principal * ead_rate * lgd)
    reject_good = -reject_good_capture * approve_good
    reject_bad = reject_bad_loss_avoidance * (principal * ead_rate * lgd)

    processing_cost = (30.0 + 0.0005 * principal) * cost_multiplier
    review_cost = (90.0 + 0.0010 * principal) * cost_multiplier

    return pd.DataFrame(
        {
            "principal": principal,
            "payment": payment.fillna(0.0),
            "annual_income": annual_income.fillna(0.0),
            "goods_price": goods_price.fillna(principal),
            "monthly_income_proxy": monthly_income_proxy.fillna(0.0),
            "term_proxy_months": term_proxy_months,
            "payment_burden": payment_burden,
            "advance_ratio": advance_ratio,
            "lgd": lgd,
            "approve_good": approve_good,
            "approve_bad": approve_bad,
            "reject_good": reject_good,
            "reject_bad": reject_bad,
            "processing_cost": processing_cost,
            "review_cost": review_cost,
        },
        index=frame.index,
    )


def compute_expected_value_frame(
    frame: pd.DataFrame,
    calibrated_pd_column: str,
    unit_economics_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute expected decision values from calibrated PD and economics."""

    calibrated_pd = pd.to_numeric(frame[calibrated_pd_column], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    ev_approve = (
        (1.0 - calibrated_pd) * unit_economics_frame["approve_good"]
        + calibrated_pd * unit_economics_frame["approve_bad"]
        - unit_economics_frame["processing_cost"]
    )
    ev_reject = (
        (1.0 - calibrated_pd) * unit_economics_frame["reject_good"]
        + calibrated_pd * unit_economics_frame["reject_bad"]
        - unit_economics_frame["processing_cost"]
    )
    ev_review = np.maximum(ev_approve, ev_reject) - unit_economics_frame["review_cost"]

    return pd.DataFrame(
        {
            "ev_approve": ev_approve,
            "ev_reject": ev_reject,
            "ev_review": ev_review,
        },
        index=frame.index,
    )


def build_unit_economics_summary(unit_economics_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize borrower-level economics into a reporting frame."""

    summary_rows = [
        {"metric": "population_size", "value": float(len(unit_economics_frame))},
        {"metric": "mean_principal", "value": float(unit_economics_frame["principal"].mean())},
        {"metric": "median_principal", "value": float(unit_economics_frame["principal"].median())},
        {"metric": "mean_term_proxy_months", "value": float(unit_economics_frame["term_proxy_months"].mean())},
        {"metric": "mean_payment_burden", "value": float(unit_economics_frame["payment_burden"].mean())},
        {"metric": "mean_advance_ratio", "value": float(unit_economics_frame["advance_ratio"].mean())},
        {"metric": "mean_lgd", "value": float(unit_economics_frame["lgd"].mean())},
        {"metric": "mean_approve_good", "value": float(unit_economics_frame["approve_good"].mean())},
        {"metric": "mean_approve_bad", "value": float(unit_economics_frame["approve_bad"].mean())},
        {"metric": "mean_processing_cost", "value": float(unit_economics_frame["processing_cost"].mean())},
        {"metric": "mean_review_cost", "value": float(unit_economics_frame["review_cost"].mean())},
    ]
    return pd.DataFrame(summary_rows)


def evaluate_cutoff_curve(
    frame: pd.DataFrame,
    score_column: str,
    calibrated_pd_column: str,
    unit_economics_frame: pd.DataFrame,
    cutoff_grid: list[int] | np.ndarray,
    review_buffer_points: int,
    target_column: str | None = None,
    guardrails: Any | None = None,
    scenario_name: str = "base",
) -> pd.DataFrame:
    """Evaluate a review-buffer cutoff policy over a grid of score cutoffs."""

    scores = pd.to_numeric(frame[score_column], errors="coerce")
    calibrated_pd = pd.to_numeric(frame[calibrated_pd_column], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    expected_values = compute_expected_value_frame(frame, calibrated_pd_column, unit_economics_frame)
    guardrail_mapping = _to_mapping(guardrails) if guardrails is not None else {}

    rows: list[dict[str, float | int | bool | str | None]] = []
    for cutoff in np.asarray(cutoff_grid, dtype=int):
        approve_min_score = float(cutoff + review_buffer_points)
        reject_below_score = float(cutoff - review_buffer_points)

        approve_mask = scores >= approve_min_score
        reject_mask = scores < reject_below_score
        review_mask = ~(approve_mask | reject_mask)

        total_expected_value = float(
            expected_values.loc[approve_mask, "ev_approve"].sum()
            + expected_values.loc[reject_mask, "ev_reject"].sum()
            + expected_values.loc[review_mask, "ev_review"].sum()
        )
        approval_rate = float(approve_mask.mean())
        review_rate = float(review_mask.mean())
        reject_rate = float(reject_mask.mean())
        approved_book_mean_pd = float(calibrated_pd.loc[approve_mask].mean()) if approve_mask.any() else np.nan

        actual_approved_bad_rate = None
        if target_column is not None and target_column in frame.columns:
            actual_approved_bad_rate = (
                float(pd.to_numeric(frame.loc[approve_mask, target_column], errors="coerce").mean())
                if approve_mask.any()
                else np.nan
            )

        passes_guardrails = bool(
            approve_mask.any()
            and approved_book_mean_pd <= float(guardrail_mapping.get("max_approved_book_bad_rate", np.inf))
            and review_rate <= float(guardrail_mapping.get("max_manual_review_share", np.inf))
            and approval_rate >= float(guardrail_mapping.get("min_approval_rate", 0.0))
        )

        rows.append(
            {
                "scenario_name": scenario_name,
                "score_cutoff": int(cutoff),
                "approve_min_score": approve_min_score,
                "reject_below_score": reject_below_score,
                "approval_rate": approval_rate,
                "review_rate": review_rate,
                "reject_rate": reject_rate,
                "approved_count": int(approve_mask.sum()),
                "review_count": int(review_mask.sum()),
                "rejected_count": int(reject_mask.sum()),
                "approved_book_mean_pd": approved_book_mean_pd,
                "actual_approved_bad_rate": actual_approved_bad_rate,
                "total_expected_value": total_expected_value,
                "expected_value_per_applicant": float(total_expected_value / len(frame)) if len(frame) else 0.0,
                "passes_guardrails": passes_guardrails,
            }
        )

    return pd.DataFrame(rows)


def select_optimal_cutoff(cutoff_curve: pd.DataFrame) -> dict[str, float | int | bool | str | None]:
    """Select the best cutoff row, preferring guardrail-compliant scenarios."""

    if cutoff_curve.empty:
        raise ValueError("cutoff_curve must not be empty.")

    eligible = cutoff_curve.loc[cutoff_curve["passes_guardrails"]].copy()
    if eligible.empty:
        eligible = cutoff_curve.copy()

    best_row = eligible.sort_values(
        by=["total_expected_value", "expected_value_per_applicant", "score_cutoff"],
        ascending=[False, False, False],
    ).iloc[0]
    return best_row.to_dict()


def run_cutoff_sensitivity_analysis(
    frame: pd.DataFrame,
    score_column: str,
    calibrated_pd_column: str,
    unit_economics_config: Any,
    cutoff_strategy_config: Any,
    sensitivity_config: Any,
    cutoff_grid: list[int] | np.ndarray,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run configured economics scenarios across a cutoff grid."""

    cutoff_strategy_mapping = _to_mapping(cutoff_strategy_config)
    review_buffer_points = int(cutoff_strategy_mapping.get("review_buffer_points", 10))
    guardrails = cutoff_strategy_mapping.get("guardrails", {})
    sensitivity_mapping = _to_mapping(sensitivity_config)

    summary_rows: list[dict[str, float | int | bool | str | None]] = []
    curve_frames: list[pd.DataFrame] = []

    for scenario_name, scenario_overrides in sensitivity_mapping.get("scenarios", {}).items():
        unit_economics_frame = build_unit_economics_frame(
            frame=frame,
            unit_economics_config=unit_economics_config,
            scenario_overrides=scenario_overrides,
        )
        scenario_curve = evaluate_cutoff_curve(
            frame=frame,
            score_column=score_column,
            calibrated_pd_column=calibrated_pd_column,
            unit_economics_frame=unit_economics_frame,
            cutoff_grid=cutoff_grid,
            review_buffer_points=review_buffer_points,
            target_column=target_column,
            guardrails=guardrails,
            scenario_name=str(scenario_name),
        )
        curve_frames.append(scenario_curve)

        optimal_row = select_optimal_cutoff(scenario_curve)
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "optimal_cutoff": int(optimal_row["score_cutoff"]),
                "approval_rate": float(optimal_row["approval_rate"]),
                "review_rate": float(optimal_row["review_rate"]),
                "approved_book_mean_pd": float(optimal_row["approved_book_mean_pd"]),
                "actual_approved_bad_rate": optimal_row.get("actual_approved_bad_rate"),
                "total_expected_value": float(optimal_row["total_expected_value"]),
                "expected_value_per_applicant": float(optimal_row["expected_value_per_applicant"]),
                "passes_guardrails": bool(optimal_row["passes_guardrails"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    combined_curve = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    return summary, combined_curve


def select_final_scenario_cutoff(
    sensitivity_summary: pd.DataFrame,
    percentile: float = 0.75,
) -> int:
    """Select the final cutoff from scenario-optimal cutoffs using a percentile rule."""

    if sensitivity_summary.empty:
        raise ValueError("sensitivity_summary must not be empty.")

    eligible = sensitivity_summary.loc[sensitivity_summary["passes_guardrails"]].copy()
    if eligible.empty:
        eligible = sensitivity_summary.copy()

    cutoff_values = np.sort(eligible["optimal_cutoff"].astype(float).to_numpy())
    index = int(np.ceil((len(cutoff_values) - 1) * percentile))
    return int(cutoff_values[index])
