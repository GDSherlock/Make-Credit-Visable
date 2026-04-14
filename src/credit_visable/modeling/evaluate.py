"""Evaluation utilities for binary credit risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def _build_calibration_payload(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]:
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_score": np.asarray(y_score, dtype=float),
        }
    ).sort_values("y_score")

    if frame.empty:
        return {
            "bin_index": [],
            "count": [],
            "predicted_mean": [],
            "observed_rate": [],
            "score_min": [],
            "score_max": [],
        }

    quantile_count = max(1, min(int(n_bins), len(frame)))
    ranked = frame["y_score"].rank(method="first")
    frame["calibration_bin"] = pd.qcut(
        ranked,
        q=quantile_count,
        labels=False,
        duplicates="drop",
    )

    calibration = (
        frame.groupby("calibration_bin", observed=False)
        .agg(
            count=("y_true", "size"),
            predicted_mean=("y_score", "mean"),
            observed_rate=("y_true", "mean"),
            score_min=("y_score", "min"),
            score_max=("y_score", "max"),
        )
        .reset_index()
        .rename(columns={"calibration_bin": "bin_index"})
    )
    calibration["bin_index"] = calibration["bin_index"].astype(int) + 1

    return {
        "bin_index": calibration["bin_index"].astype(int).tolist(),
        "count": calibration["count"].astype(int).tolist(),
        "predicted_mean": calibration["predicted_mean"].astype(float).tolist(),
        "observed_rate": calibration["observed_rate"].astype(float).tolist(),
        "score_min": calibration["score_min"].astype(float).tolist(),
        "score_max": calibration["score_max"].astype(float).tolist(),
    }


def _build_gain_lift_payload(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_points: int = 101,
) -> dict[str, dict[str, list[float]]]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_score_array = np.asarray(y_score, dtype=float)
    if y_true_array.size == 0:
        empty = {
            "population_share": [],
            "captured_bad_share": [],
            "lift": [],
        }
        return {"gain": empty, "lift": empty}

    sort_order = np.argsort(-y_score_array, kind="mergesort")
    sorted_target = y_true_array[sort_order]
    cumulative_bad = np.cumsum(sorted_target)
    total_bad = float(sorted_target.sum())

    population_grid = np.linspace(0.0, 1.0, num=max(2, int(n_points)))
    captured_bad_share: list[float] = []
    lift_values: list[float] = []
    effective_population_share: list[float] = []

    for population_share in population_grid:
        count = int(np.ceil(population_share * len(sorted_target)))
        if count <= 0:
            effective_population_share.append(0.0)
            captured_bad_share.append(0.0)
            lift_values.append(0.0)
            continue

        actual_population_share = float(count / len(sorted_target))
        captured_share = (
            float(cumulative_bad[count - 1] / total_bad) if total_bad > 0 else 0.0
        )
        lift = captured_share / actual_population_share if actual_population_share > 0 else 0.0

        effective_population_share.append(actual_population_share)
        captured_bad_share.append(captured_share)
        lift_values.append(lift)

    return {
        "gain": {
            "population_share": effective_population_share,
            "captured_bad_share": captured_bad_share,
        },
        "lift": {
            "population_share": effective_population_share,
            "captured_bad_share": captured_bad_share,
            "lift": lift_values,
        },
    }


def build_binary_diagnostic_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, dict[str, float | list[float]]]:
    """Build ROC, precision-recall, and KS diagnostics for plotting."""

    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true_array, y_score_array)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_true_array,
        y_score_array,
    )

    serializable_roc_thresholds = np.where(
        np.isfinite(roc_thresholds),
        roc_thresholds,
        1.0,
    )
    ks_values = roc_tpr - roc_fpr
    ks_index = int(np.argmax(ks_values))

    calibration = _build_calibration_payload(y_true_array, y_score_array)
    gain_lift = _build_gain_lift_payload(y_true_array, y_score_array)

    return {
        "roc": {
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
            "thresholds": serializable_roc_thresholds.tolist(),
        },
        "precision_recall": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "ks": {
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
            "thresholds": serializable_roc_thresholds.tolist(),
            "values": ks_values.tolist(),
            "statistic": float(ks_values[ks_index]),
            "threshold": float(serializable_roc_thresholds[ks_index]),
        },
        "calibration": calibration,
        "gain": gain_lift["gain"],
        "lift": gain_lift["lift"],
    }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Compute a compact set of starter classification metrics."""

    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)
    y_pred = (y_score_array >= threshold).astype(int)
    diagnostic_curves = build_binary_diagnostic_curves(y_true_array, y_score_array)
    calibration_table = diagnostic_curves["calibration"]
    calibration_gap_values = [
        abs(float(predicted_mean) - float(observed_rate))
        for predicted_mean, observed_rate in zip(
            calibration_table["predicted_mean"],
            calibration_table["observed_rate"],
        )
    ]

    calibration_slope = np.nan
    calibration_intercept = np.nan
    clipped_scores = np.clip(y_score_array, 1e-6, 1.0 - 1e-6)
    if np.unique(y_true_array).size >= 2:
        slope_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        slope_model.fit(np.log(clipped_scores / (1.0 - clipped_scores)).reshape(-1, 1), y_true_array)
        calibration_slope = float(slope_model.coef_[0][0])
        calibration_intercept = float(slope_model.intercept_[0])

    return {
        "roc_auc": float(roc_auc_score(y_true_array, y_score_array)),
        "gini": float((2.0 * roc_auc_score(y_true_array, y_score_array)) - 1.0),
        "average_precision": float(
            average_precision_score(y_true_array, y_score_array)
        ),
        "ks_statistic": float(diagnostic_curves["ks"]["statistic"]),
        "ks_threshold": float(diagnostic_curves["ks"]["threshold"]),
        "brier_score": float(brier_score_loss(y_true_array, y_score_array)),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "max_abs_decile_gap": float(max(calibration_gap_values, default=0.0)),
        "positive_rate_baseline": float(y_true_array.mean()),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred).tolist(),
    }
