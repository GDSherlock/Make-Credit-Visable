"""Grouped fairness and governance review helpers for Phase 5."""

from __future__ import annotations

from typing import Any

import numpy as np
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
            frame.groupby(column, dropna=False, observed=False)[target_column]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={column: "group", "mean": "target_rate"})
        )
        summary.insert(0, "protected_attribute", column)
        outputs.append(summary)

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return result


def derive_age_band_from_days_birth(
    days_birth: pd.Series,
    missing_label: str = "Missing",
) -> pd.Series:
    """Convert Home Credit days-birth values into fixed policy review bands."""

    age_years = -pd.to_numeric(days_birth, errors="coerce") / 365.25
    labels = ["[20,25)", "[25,35)", "[35,45)", "[45,55)", "[55,65)", "[65,70)"]
    age_band = pd.cut(
        age_years,
        bins=[20, 25, 35, 45, 55, 65, 70],
        labels=labels,
        right=False,
        include_lowest=True,
    )
    output = age_band.astype("object").fillna(missing_label)
    return pd.Series(output, index=days_birth.index, name="age_band")


def collapse_rare_categories(
    values: pd.Series,
    top_n: int = 8,
    other_label: str = "Other",
    missing_label: str = "Missing",
) -> pd.Series:
    """Keep the top-N categories and collapse the remainder for grouped review."""

    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    series = values.astype("object")
    filled = series.where(series.notna(), missing_label)
    top_categories = [
        category
        for category in filled.loc[filled != missing_label]
        .value_counts(dropna=False)
        .head(top_n)
        .index
        .tolist()
    ]
    collapsed = filled.where(filled.isin(top_categories + [missing_label]), other_label)
    return pd.Series(collapsed, index=values.index, name=values.name)


def build_grouped_operational_summary(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str,
    score_column: str,
    threshold: float = 0.5,
    protected_attribute: str | None = None,
) -> pd.DataFrame:
    """Compute grouped decision diagnostics for one protected or sensitive attribute."""

    required_columns = {group_column, target_column, score_column}
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(f"Frame missing required columns: {sorted(missing)}")

    working = frame[[group_column, target_column, score_column]].copy()
    working[group_column] = working[group_column].astype("object").fillna("Missing")
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce").fillna(0)
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce").fillna(0.0)
    working["approved_flag"] = (working[score_column] < threshold).astype(int)
    working["rejected_flag"] = 1 - working["approved_flag"]
    working["approved_bad_flag"] = np.where(
        working["approved_flag"] == 1,
        working[target_column],
        np.nan,
    )
    working["rejected_bad_flag"] = np.where(
        working["rejected_flag"] == 1,
        working[target_column],
        0.0,
    )

    total_count = len(working)
    total_bad = float(working[target_column].sum())
    summary = (
        working.groupby(group_column, dropna=False, observed=False)
        .agg(
            count=(target_column, "size"),
            bad_count=(target_column, "sum"),
            actual_default_rate=(target_column, "mean"),
            mean_predicted_pd=(score_column, "mean"),
            approval_rate=("approved_flag", "mean"),
            reject_rate=("rejected_flag", "mean"),
            approved_count=("approved_flag", "sum"),
            rejected_count=("rejected_flag", "sum"),
            approved_bad_rate=("approved_bad_flag", "mean"),
            rejected_bad_count=("rejected_bad_flag", "sum"),
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )
    summary["population_share"] = summary["count"] / total_count if total_count else 0.0
    summary["rejected_bad_capture_rate"] = np.where(
        total_bad > 0,
        summary["rejected_bad_count"] / total_bad,
        0.0,
    )
    best_approval_rate = summary["approval_rate"].max()
    summary["approval_rate_vs_best_group"] = np.where(
        best_approval_rate > 0,
        summary["approval_rate"] / best_approval_rate,
        np.nan,
    )
    summary.insert(0, "protected_attribute", protected_attribute or group_column)
    summary["threshold"] = float(threshold)
    summary = summary.drop(columns=["rejected_bad_count"])
    summary = summary.sort_values(["count", "group"], ascending=[False, True]).reset_index(
        drop=True
    )
    return summary[
        [
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
        ]
    ]


def _prepare_group_series(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    top_n_categories: int,
) -> tuple[str, pd.Series]:
    source_column = spec["source_column"]
    group_column = spec.get("group_column", spec["protected_attribute"])
    kind = spec.get("kind", "identity")

    if source_column not in frame.columns:
        raise KeyError(f"Source column not found for fairness group: {source_column}")

    if kind == "age_band":
        grouped = derive_age_band_from_days_birth(
            frame[source_column],
            missing_label=spec.get("missing_label", "Missing"),
        )
    elif kind == "top_categories":
        grouped = collapse_rare_categories(
            frame[source_column],
            top_n=int(spec.get("top_n", top_n_categories)),
            other_label=spec.get("other_label", "Other"),
            missing_label=spec.get("missing_label", "Missing"),
        )
    else:
        grouped = frame[source_column].astype("object").fillna(
            spec.get("missing_label", "Missing")
        )

    grouped.name = group_column
    return group_column, grouped


def build_group_fairness_summary(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
    group_specs: list[dict[str, Any]],
    threshold: float = 0.5,
    top_n_categories: int = 8,
) -> pd.DataFrame:
    """Apply multiple grouped fairness reviews and return one long summary table."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if score_column not in frame.columns:
        raise KeyError(f"Score column not found: {score_column}")

    working = frame.copy()
    outputs = []
    for spec in group_specs:
        protected_attribute = spec["protected_attribute"]
        group_column, grouped = _prepare_group_series(
            working,
            spec=spec,
            top_n_categories=top_n_categories,
        )
        working[group_column] = grouped
        summary = build_grouped_operational_summary(
            working,
            group_column=group_column,
            target_column=target_column,
            score_column=score_column,
            threshold=threshold,
            protected_attribute=protected_attribute,
        )
        summary.insert(1, "group_column", group_column)
        summary.insert(2, "source_column", spec["source_column"])
        outputs.append(summary)

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def _build_group_decision_detail(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    working = frame[[group_column, target_column, score_column]].copy()
    working[group_column] = working[group_column].astype("object").fillna("Missing")
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[target_column, score_column])
    working["approved_flag"] = (working[score_column] < threshold).astype(int)
    working["good_flag"] = (working[target_column] == 0).astype(int)
    working["bad_flag"] = (working[target_column] == 1).astype(int)

    grouped = (
        working.groupby(group_column, dropna=False, observed=False)
        .agg(
            count=(target_column, "size"),
            approval_rate=("approved_flag", "mean"),
            good_count=("good_flag", "sum"),
            bad_count=("bad_flag", "sum"),
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )

    group_rows = []
    for group_value, subset in working.groupby(group_column, dropna=False, observed=False):
        good_mask = subset[target_column] == 0
        bad_mask = subset[target_column] == 1
        true_positive_rate = (
            float(subset.loc[good_mask, "approved_flag"].mean()) if good_mask.any() else np.nan
        )
        false_positive_rate = (
            float(subset.loc[bad_mask, "approved_flag"].mean()) if bad_mask.any() else np.nan
        )
        group_rows.append(
            {
                "group": group_value,
                "true_positive_rate_for_goods": true_positive_rate,
                "false_positive_rate_for_bads": false_positive_rate,
            }
        )

    detail = grouped.merge(pd.DataFrame(group_rows), on="group", how="left")
    detail["approval_rate"] = detail["approval_rate"].astype(float)
    return detail


def _safe_gap(values: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return float("nan")
    return float(clean_values.max() - clean_values.min())


def _safe_ratio(values: pd.Series) -> float:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return float("nan")
    max_value = float(clean_values.max())
    min_value = float(clean_values.min())
    if max_value <= 0:
        return float("nan")
    return float(min_value / max_value)


def build_group_fairness_metric_summary(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
    group_specs: list[dict[str, Any]],
    threshold: float = 0.5,
    top_n_categories: int = 8,
) -> pd.DataFrame:
    """Summarize regulator-style fairness metrics for configured groups.

    The fairness lens here treats approval as the positive decision and
    ``TARGET == 0`` as the favorable ground-truth outcome.
    """

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if score_column not in frame.columns:
        raise KeyError(f"Score column not found: {score_column}")

    working = frame.copy()
    outputs: list[dict[str, Any]] = []

    for spec in group_specs:
        protected_attribute = spec["protected_attribute"]
        group_column, grouped = _prepare_group_series(
            working,
            spec=spec,
            top_n_categories=top_n_categories,
        )
        working[group_column] = grouped
        detail = _build_group_decision_detail(
            working,
            group_column=group_column,
            target_column=target_column,
            score_column=score_column,
            threshold=threshold,
        )
        if detail.empty:
            continue

        approval_rates = detail["approval_rate"]
        best_group_row = detail.loc[approval_rates.idxmax()]
        worst_group_row = detail.loc[approval_rates.idxmin()]
        equal_opportunity_diff = _safe_gap(detail["true_positive_rate_for_goods"])
        false_positive_rate_diff = _safe_gap(detail["false_positive_rate_for_bads"])

        outputs.append(
            {
                "protected_attribute": protected_attribute,
                "group_column": group_column,
                "source_column": spec["source_column"],
                "group_count": int(detail["group"].nunique(dropna=False)),
                "threshold": float(threshold),
                "best_approval_group": best_group_row["group"],
                "best_approval_rate": float(best_group_row["approval_rate"]),
                "worst_approval_group": worst_group_row["group"],
                "worst_approval_rate": float(worst_group_row["approval_rate"]),
                "demographic_parity_diff": _safe_gap(approval_rates),
                "demographic_parity_ratio": _safe_ratio(approval_rates),
                "approval_disparity_ratio": _safe_ratio(approval_rates),
                "equal_opportunity_diff": equal_opportunity_diff,
                "false_positive_rate_diff": false_positive_rate_diff,
                "equalized_odds_gap": float(
                    np.nanmax([equal_opportunity_diff, false_positive_rate_diff])
                ),
            }
        )

    return pd.DataFrame(outputs)
