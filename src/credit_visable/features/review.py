"""Feature review and pruning helpers for the governed application pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from credit_visable.features.iv_woe import compute_iv_summary


@dataclass(slots=True)
class FeatureReviewOptions:
    """Configuration for IV/correlation/VIF-driven feature review."""

    correlation_threshold: float = 0.85
    vif_threshold: float = 8.0
    max_numeric_features_for_vif: int = 20
    pca_max_components: int = 8
    review_sample_size: int = 10000
    random_state: int = 42


def _sample_frame(
    frame: pd.DataFrame,
    review_sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= review_sample_size:
        return frame.copy()
    return frame.sample(n=review_sample_size, random_state=random_state).copy()


def _numeric_review_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(numeric.median())


def _compute_vif_from_correlation(numeric_frame: pd.DataFrame) -> pd.DataFrame:
    if numeric_frame.empty:
        return pd.DataFrame(columns=["feature", "vif"])

    correlation = numeric_frame.corr().fillna(0.0)
    identity = np.eye(correlation.shape[0])
    regularized = correlation.to_numpy() + (identity * 1e-6)
    precision = np.linalg.pinv(regularized)
    vif_values = np.clip(np.diag(precision), a_min=1.0, a_max=None)
    return pd.DataFrame(
        {
            "feature": numeric_frame.columns.tolist(),
            "vif": vif_values.astype(float),
        }
    ).sort_values("vif", ascending=False, ignore_index=True)


def _build_high_correlation_pairs(
    numeric_frame: pd.DataFrame,
    iv_lookup: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    correlation = numeric_frame.corr().abs()
    rows: list[dict[str, Any]] = []
    columns = correlation.columns.tolist()

    for left_index, left_feature in enumerate(columns):
        for right_feature in columns[left_index + 1 :]:
            corr_value = float(correlation.loc[left_feature, right_feature])
            if corr_value < threshold:
                continue
            left_iv = float(iv_lookup.get(left_feature, np.nan))
            right_iv = float(iv_lookup.get(right_feature, np.nan))
            if np.isnan(left_iv) and np.isnan(right_iv):
                drop_feature = max(left_feature, right_feature)
            elif np.isnan(left_iv):
                drop_feature = left_feature
            elif np.isnan(right_iv):
                drop_feature = right_feature
            else:
                drop_feature = right_feature if left_iv >= right_iv else left_feature
            keep_feature = left_feature if drop_feature == right_feature else right_feature
            rows.append(
                {
                    "left_feature": left_feature,
                    "right_feature": right_feature,
                    "abs_correlation": corr_value,
                    "left_iv": left_iv,
                    "right_iv": right_iv,
                    "keep_feature": keep_feature,
                    "drop_feature": drop_feature,
                    "review_reason": "high_correlation",
                }
            )

    return pd.DataFrame(rows).sort_values(
        "abs_correlation",
        ascending=False,
        ignore_index=True,
    ) if rows else pd.DataFrame(
        columns=[
            "left_feature",
            "right_feature",
            "abs_correlation",
            "left_iv",
            "right_iv",
            "keep_feature",
            "drop_feature",
            "review_reason",
        ]
    )


def _build_pca_diagnostics(
    numeric_frame: pd.DataFrame,
    options: FeatureReviewOptions,
) -> pd.DataFrame:
    if numeric_frame.shape[1] < 2:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_variance_ratio"])

    component_count = min(
        int(options.pca_max_components),
        int(numeric_frame.shape[1]),
    )
    standardized = (numeric_frame - numeric_frame.mean()) / numeric_frame.std(ddof=0).replace(0.0, 1.0)
    standardized = standardized.fillna(0.0)
    pca = PCA(n_components=component_count, random_state=options.random_state)
    pca.fit(standardized)

    explained_variance_ratio = pca.explained_variance_ratio_.astype(float)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained_variance_ratio) + 1),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": np.cumsum(explained_variance_ratio),
        }
    )


def prune_training_features(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    options: FeatureReviewOptions | None = None,
) -> dict[str, Any]:
    """Prune features using IV, correlation, and VIF diagnostics."""

    resolved_options = options or FeatureReviewOptions()
    review_columns = [column_name for column_name in feature_columns if column_name in frame.columns]
    if not review_columns:
        return {
            "selected_feature_columns": [],
            "dropped_feature_columns": [],
            "iv_summary": pd.DataFrame(),
            "high_correlation_pairs": pd.DataFrame(),
            "vif_summary": pd.DataFrame(),
            "pca_summary": pd.DataFrame(),
        }

    sampled_frame = _sample_frame(
        frame[[target_column, *review_columns]],
        review_sample_size=resolved_options.review_sample_size,
        random_state=resolved_options.random_state,
    )
    iv_summary = compute_iv_summary(sampled_frame, target_column=target_column, bins=10)
    iv_lookup = (
        iv_summary.set_index("feature")["iv"]
        if not iv_summary.empty and "feature" in iv_summary.columns
        else pd.Series(dtype=float)
    )

    numeric_columns = sampled_frame.drop(columns=[target_column]).select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [column_name for column_name in numeric_columns if column_name in review_columns]
    numeric_review = _numeric_review_frame(sampled_frame, numeric_columns)

    high_correlation_pairs = _build_high_correlation_pairs(
        numeric_review,
        iv_lookup=iv_lookup,
        threshold=float(resolved_options.correlation_threshold),
    )
    correlation_drops = set(high_correlation_pairs["drop_feature"].tolist())

    surviving_numeric = [
        column_name for column_name in numeric_columns if column_name not in correlation_drops
    ]
    surviving_numeric = sorted(
        surviving_numeric,
        key=lambda column_name: (
            -float(iv_lookup.get(column_name, 0.0)),
            column_name,
        ),
    )[: int(resolved_options.max_numeric_features_for_vif)]
    vif_summary = _compute_vif_from_correlation(
        _numeric_review_frame(sampled_frame, surviving_numeric)
    )
    vif_drops = set(
        vif_summary.loc[vif_summary["vif"] > float(resolved_options.vif_threshold), "feature"].tolist()
    )

    dropped_feature_columns = sorted(correlation_drops | vif_drops)
    selected_feature_columns = [
        column_name for column_name in review_columns if column_name not in set(dropped_feature_columns)
    ]
    pca_summary = _build_pca_diagnostics(
        _numeric_review_frame(sampled_frame, surviving_numeric),
        options=resolved_options,
    )

    return {
        "selected_feature_columns": selected_feature_columns,
        "dropped_feature_columns": dropped_feature_columns,
        "iv_summary": iv_summary,
        "high_correlation_pairs": high_correlation_pairs,
        "vif_summary": vif_summary,
        "pca_summary": pca_summary,
    }
