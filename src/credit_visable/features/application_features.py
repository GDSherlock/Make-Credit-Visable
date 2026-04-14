"""Application-table feature engineering helpers for the governed pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DAYS_EMPLOYED_SENTINEL = 365243


@dataclass(slots=True)
class ApplicationFeatureEngineeringOptions:
    """Configuration for application-stage feature engineering."""

    days_employed_sentinel: int = DEFAULT_DAYS_EMPLOYED_SENTINEL
    missing_indicator_threshold: float = 0.20
    max_auto_missing_indicators: int = 12
    explicit_missing_indicator_columns: tuple[str, ...] = (
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "OWN_CAR_AGE",
        "AMT_GOODS_PRICE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "COMMONAREA_AVG",
        "COMMONAREA_MEDI",
        "COMMONAREA_MODE",
    )
    ratio_features: tuple[str, ...] = (
        "INCOME_CREDIT_RATIO",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "PAYMENT_BURDEN_RATIO",
        "CHILDREN_FAMILY_RATIO",
    )
    epsilon: float = 1e-6


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_value: float | None = np.nan,
) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    ratio = numerator_numeric / denominator_numeric.replace(0.0, np.nan)
    if fill_value is not None:
        ratio = ratio.fillna(fill_value)
    return ratio.astype(float)


def _resolve_missing_indicator_columns(
    frame: pd.DataFrame,
    options: ApplicationFeatureEngineeringOptions,
) -> list[str]:
    missing_share = frame.isna().mean().sort_values(ascending=False)
    auto_columns = [
        str(column_name)
        for column_name, share in missing_share.items()
        if float(share) >= float(options.missing_indicator_threshold)
    ][: int(options.max_auto_missing_indicators)]

    ordered_columns: list[str] = []
    for column_name in [*options.explicit_missing_indicator_columns, *auto_columns]:
        if column_name in frame.columns and column_name not in ordered_columns:
            ordered_columns.append(column_name)
    return ordered_columns


def engineer_application_features(
    frame: pd.DataFrame,
    options: ApplicationFeatureEngineeringOptions | None = None,
) -> pd.DataFrame:
    """Create a governed application-stage modeling frame."""

    resolved_options = options or ApplicationFeatureEngineeringOptions()
    engineered = frame.copy()

    if "DAYS_EMPLOYED" in engineered.columns:
        days_employed_clean = pd.to_numeric(
            engineered["DAYS_EMPLOYED"], errors="coerce"
        ).replace(float(resolved_options.days_employed_sentinel), np.nan)
        engineered["DAYS_EMPLOYED_CLEAN"] = days_employed_clean
        engineered["YEARS_EMPLOYED"] = (-days_employed_clean / 365.25).clip(lower=0.0)

    if "AMT_INCOME_TOTAL" in engineered.columns and "AMT_CREDIT" in engineered.columns:
        engineered["INCOME_CREDIT_RATIO"] = _safe_divide(
            engineered["AMT_INCOME_TOTAL"],
            engineered["AMT_CREDIT"],
        )
        engineered["CREDIT_INCOME_RATIO"] = _safe_divide(
            engineered["AMT_CREDIT"],
            engineered["AMT_INCOME_TOTAL"],
        )

    if "AMT_ANNUITY" in engineered.columns and "AMT_INCOME_TOTAL" in engineered.columns:
        engineered["ANNUITY_INCOME_RATIO"] = _safe_divide(
            engineered["AMT_ANNUITY"],
            engineered["AMT_INCOME_TOTAL"],
        )
        monthly_income = pd.to_numeric(
            engineered["AMT_INCOME_TOTAL"], errors="coerce"
        ) / 12.0
        engineered["PAYMENT_BURDEN_RATIO"] = _safe_divide(
            engineered["AMT_ANNUITY"],
            monthly_income,
        )

    if "AMT_CREDIT" in engineered.columns and "AMT_GOODS_PRICE" in engineered.columns:
        goods_price = pd.to_numeric(engineered["AMT_GOODS_PRICE"], errors="coerce").where(
            pd.to_numeric(engineered["AMT_GOODS_PRICE"], errors="coerce") > 0.0,
            pd.to_numeric(engineered["AMT_CREDIT"], errors="coerce"),
        )
        engineered["CREDIT_GOODS_RATIO"] = _safe_divide(
            engineered["AMT_CREDIT"],
            goods_price,
        )

    if "CNT_CHILDREN" in engineered.columns and "CNT_FAM_MEMBERS" in engineered.columns:
        engineered["CHILDREN_FAMILY_RATIO"] = _safe_divide(
            engineered["CNT_CHILDREN"],
            engineered["CNT_FAM_MEMBERS"],
        )

    for column_name in _resolve_missing_indicator_columns(engineered, resolved_options):
        engineered[f"MISSING_FLAG_{column_name}"] = engineered[column_name].isna().astype(int)

    return engineered


def build_application_feature_summary(
    raw_frame: pd.DataFrame,
    engineered_frame: pd.DataFrame,
    options: ApplicationFeatureEngineeringOptions | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable summary of the engineered application features."""

    resolved_options = options or ApplicationFeatureEngineeringOptions()
    new_columns = [
        column_name
        for column_name in engineered_frame.columns
        if column_name not in raw_frame.columns
    ]
    missing_indicator_columns = [
        column_name for column_name in new_columns if column_name.startswith("MISSING_FLAG_")
    ]
    ratio_columns = [
        column_name for column_name in new_columns if column_name in set(resolved_options.ratio_features)
    ]

    return {
        "days_employed_sentinel": int(resolved_options.days_employed_sentinel),
        "new_column_count": len(new_columns),
        "new_columns": new_columns,
        "ratio_columns": ratio_columns,
        "missing_indicator_columns": missing_indicator_columns,
        "missing_indicator_threshold": float(resolved_options.missing_indicator_threshold),
        "explicit_missing_indicator_columns": list(resolved_options.explicit_missing_indicator_columns),
    }
