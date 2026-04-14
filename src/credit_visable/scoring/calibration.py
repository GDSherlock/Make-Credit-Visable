"""Probability calibration helpers for production scorecards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


DEFAULT_EPSILON = 1e-6


def clip_probabilities(values: np.ndarray | pd.Series, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for stable calibration math."""

    return np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)


def _logit(values: np.ndarray | pd.Series, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    clipped = clip_probabilities(values, epsilon=epsilon)
    return np.log(clipped / (1.0 - clipped))


def fit_platt_calibrator(
    raw_probabilities: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
) -> LogisticRegression:
    """Fit a one-dimensional Platt calibrator on raw model probabilities."""

    y_array = np.asarray(y_true, dtype=int)
    x_array = _logit(raw_probabilities).reshape(-1, 1)

    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("raw_probabilities and y_true must have the same length.")
    if np.unique(y_array).size < 2:
        raise ValueError("Platt calibration requires both classes to be present.")

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(x_array, y_array)
    return calibrator


def apply_platt_calibrator(
    raw_probabilities: np.ndarray | pd.Series,
    calibrator: LogisticRegression,
) -> np.ndarray:
    """Apply a fitted Platt calibrator to raw model probabilities."""

    x_array = _logit(raw_probabilities).reshape(-1, 1)
    return clip_probabilities(calibrator.predict_proba(x_array)[:, 1])


def build_calibration_table(
    y_true: np.ndarray | pd.Series,
    calibrated_probabilities: np.ndarray | pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Build a decile-style calibration table from predicted probabilities."""

    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_score": clip_probabilities(calibrated_probabilities),
        }
    ).sort_values("y_score")

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "bin_index",
                "count",
                "predicted_mean",
                "observed_rate",
                "score_min",
                "score_max",
                "absolute_gap",
            ]
        )

    quantile_count = max(1, min(int(bins), len(frame)))
    ranked = frame["y_score"].rank(method="first")
    frame["calibration_bin"] = pd.qcut(
        ranked,
        q=quantile_count,
        labels=False,
        duplicates="drop",
    )

    calibration_table = (
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
    calibration_table["bin_index"] = calibration_table["bin_index"].astype(int) + 1
    calibration_table["absolute_gap"] = (
        calibration_table["predicted_mean"] - calibration_table["observed_rate"]
    ).abs()
    return calibration_table


def compute_population_stability_index(
    expected_scores: np.ndarray | pd.Series,
    actual_scores: np.ndarray | pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI using quantile bins defined on the expected population."""

    expected_array = np.asarray(expected_scores, dtype=float)
    actual_array = np.asarray(actual_scores, dtype=float)

    if expected_array.size == 0 or actual_array.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, num=max(2, int(bins) + 1))
    edges = np.unique(np.quantile(expected_array, quantiles))
    if edges.size <= 1:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected_array, bins=edges)
    actual_counts, _ = np.histogram(actual_array, bins=edges)

    expected_share = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_share = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def compute_calibration_monitoring_metrics(
    y_true: np.ndarray | pd.Series,
    calibrated_probabilities: np.ndarray | pd.Series,
    reference_scores: np.ndarray | pd.Series | None = None,
    comparison_scores: np.ndarray | pd.Series | None = None,
    bins: int = 10,
) -> dict[str, float | int | None]:
    """Compute core production calibration monitoring metrics."""

    y_array = np.asarray(y_true, dtype=int)
    calibrated_array = clip_probabilities(calibrated_probabilities)

    metrics: dict[str, float | int | None] = {
        "brier_score": float(brier_score_loss(y_array, calibrated_array)),
        "calibration_slope": None,
        "calibration_intercept": None,
        "max_abs_decile_gap": 0.0,
        "score_psi": None,
        "calibration_bins": int(bins),
    }

    if np.unique(y_array).size >= 2:
        slope_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        slope_model.fit(_logit(calibrated_array).reshape(-1, 1), y_array)
        metrics["calibration_slope"] = float(slope_model.coef_[0][0])
        metrics["calibration_intercept"] = float(slope_model.intercept_[0])

    calibration_table = build_calibration_table(y_array, calibrated_array, bins=bins)
    if not calibration_table.empty:
        metrics["max_abs_decile_gap"] = float(calibration_table["absolute_gap"].max())

    if reference_scores is not None and comparison_scores is not None:
        metrics["score_psi"] = compute_population_stability_index(
            expected_scores=reference_scores,
            actual_scores=comparison_scores,
            bins=bins,
        )

    return metrics
