"""IV / WOE diagnostics used by report-facing EDA workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


_MISSING_LABEL = "Missing"
_WOE_SMOOTHING = 0.5


def _coerce_binary_target(target: pd.Series) -> pd.Series:
    target_values = pd.to_numeric(target, errors="coerce")
    if target_values.isna().any():
        raise ValueError("Target column contains non-numeric or missing values.")

    unique_values = set(target_values.unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError("Target column must be binary with values 0 and 1.")

    return target_values.astype(int)


def _bin_numeric_feature(values: pd.Series, bins: int) -> tuple[pd.Series, str]:
    numeric_values = pd.to_numeric(values, errors="coerce")
    non_missing = numeric_values.dropna()

    if non_missing.nunique(dropna=True) <= 1:
        categories = numeric_values.astype("object").where(numeric_values.notna(), _MISSING_LABEL)
        categories.attrs["bin_labels"] = sorted(
            categories.astype(str).unique().tolist(),
            key=lambda value: (value == _MISSING_LABEL, value),
        )
        return categories.rename(values.name), "numeric_constant"

    requested_bins = max(2, min(int(bins), int(non_missing.nunique(dropna=True))))
    try:
        binned = pd.qcut(non_missing, q=requested_bins, duplicates="drop")
        bucketed = pd.Series(pd.NA, index=values.index, dtype="object")
        bucketed.loc[non_missing.index] = binned.astype(str)
        bucketed = bucketed.fillna(_MISSING_LABEL)
        bucketed.attrs["bin_labels"] = [
            *[str(category) for category in binned.cat.categories],
            *([_MISSING_LABEL] if bucketed.eq(_MISSING_LABEL).any() else []),
        ]
        return bucketed.rename(values.name), "numeric_quantile_bin"
    except ValueError:
        categories = numeric_values.astype("object").where(numeric_values.notna(), _MISSING_LABEL)
        categories.attrs["bin_labels"] = sorted(
            categories.astype(str).unique().tolist(),
            key=lambda value: (value == _MISSING_LABEL, value),
        )
        return categories.rename(values.name), "numeric_fallback_category"


def _bucket_feature(values: pd.Series, bins: int) -> tuple[pd.Series, str]:
    if pd.api.types.is_numeric_dtype(values):
        return _bin_numeric_feature(values, bins=bins)

    categories = values.astype("object").where(values.notna(), _MISSING_LABEL)
    categories.attrs["bin_labels"] = sorted(
        categories.astype(str).unique().tolist(),
        key=lambda value: (value == _MISSING_LABEL, value),
    )
    return categories.rename(values.name), "categorical"


def compute_woe_detail(
    frame: pd.DataFrame,
    target_column: str,
    feature_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute a WOE detail table for a single feature.

    The positive class is assumed to be ``1`` (bad / default), while
    the negative class is assumed to be ``0`` (good / non-default).
    """

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if feature_column not in frame.columns:
        raise KeyError(f"Feature column not found: {feature_column}")

    target_values = _coerce_binary_target(frame[target_column])
    bucketed_values, binning_strategy = _bucket_feature(frame[feature_column], bins=bins)

    working = pd.DataFrame(
        {
            "feature": feature_column,
            "bin": bucketed_values.astype(str),
            "target": target_values,
        }
    )
    ordered_bin_labels = bucketed_values.attrs.get("bin_labels")

    grouped = (
        working.groupby("bin", dropna=False, observed=False)["target"]
        .agg(total_count="size", bad_count="sum")
        .reset_index()
    )
    grouped["good_count"] = grouped["total_count"] - grouped["bad_count"]
    grouped["distribution_share"] = grouped["total_count"] / max(len(working), 1)
    grouped["event_rate"] = np.where(
        grouped["total_count"] > 0,
        grouped["bad_count"] / grouped["total_count"],
        0.0,
    )

    total_good = float((target_values == 0).sum())
    total_bad = float((target_values == 1).sum())
    bin_count = len(grouped)
    smoothing_total_good = total_good + _WOE_SMOOTHING * bin_count
    smoothing_total_bad = total_bad + _WOE_SMOOTHING * bin_count

    grouped["good_share"] = np.where(
        total_good > 0,
        grouped["good_count"] / total_good,
        0.0,
    )
    grouped["bad_share"] = np.where(
        total_bad > 0,
        grouped["bad_count"] / total_bad,
        0.0,
    )
    grouped["smoothed_good_share"] = (grouped["good_count"] + _WOE_SMOOTHING) / max(
        smoothing_total_good,
        1.0,
    )
    grouped["smoothed_bad_share"] = (grouped["bad_count"] + _WOE_SMOOTHING) / max(
        smoothing_total_bad,
        1.0,
    )
    grouped["woe"] = np.log(grouped["smoothed_good_share"] / grouped["smoothed_bad_share"])
    grouped["iv_component"] = (
        grouped["smoothed_good_share"] - grouped["smoothed_bad_share"]
    ) * grouped["woe"]

    if ordered_bin_labels:
        ordered_categories = pd.Categorical(
            grouped["bin"],
            categories=ordered_bin_labels,
            ordered=True,
        )
        order_frame = grouped.assign(_bin_order_key=ordered_categories).sort_values(
            "_bin_order_key"
        )
        order_frame = order_frame.drop(columns="_bin_order_key").reset_index(drop=True)
    else:
        order_frame = grouped.sort_values(
            by=["bin"],
            key=lambda series: series.map(lambda value: (value == _MISSING_LABEL, str(value))),
        ).reset_index(drop=True)
    order_frame.insert(0, "bin_order", np.arange(1, len(order_frame) + 1))
    order_frame.insert(1, "feature_type", binning_strategy)
    order_frame.insert(2, "feature", feature_column)
    order_frame["status"] = "ok"

    return order_frame[
        [
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
        ]
    ]


def compute_woe_details(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute WOE detail rows for multiple features."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    selected_features = feature_columns or [
        column for column in frame.columns if column != target_column
    ]
    detail_frames = [
        compute_woe_detail(
            frame=frame,
            target_column=target_column,
            feature_column=feature_column,
            bins=bins,
        )
        for feature_column in selected_features
    ]
    return pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()


def compute_iv_summary(
    frame: pd.DataFrame,
    target_column: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute a feature-level IV summary table."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    feature_columns = [column for column in frame.columns if column != target_column]
    detail = compute_woe_details(
        frame=frame,
        target_column=target_column,
        feature_columns=feature_columns,
        bins=bins,
    )
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_type",
                "iv",
                "bin_count",
                "missing_bin_present",
                "bins_requested",
                "status",
            ]
        )

    summary = (
        detail.groupby(["feature", "feature_type", "status"], dropna=False)
        .agg(
            iv=("iv_component", "sum"),
            bin_count=("bin", "nunique"),
            missing_bin_present=("bin", lambda values: _MISSING_LABEL in set(values)),
        )
        .reset_index()
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    summary["bins_requested"] = int(bins)
    return summary[
        [
            "feature",
            "feature_type",
            "iv",
            "bin_count",
            "missing_bin_present",
            "bins_requested",
            "status",
        ]
    ]


def fit_woe_placeholder(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    """Return metadata for a future WOE transformer implementation."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")

    selected_features = feature_columns or [
        column for column in frame.columns if column != target_column
    ]

    return {
        "target_column": target_column,
        "features": selected_features,
        "fitted": False,
        "notes": "TODO: replace this placeholder with a reusable WOE transformer.",
    }
