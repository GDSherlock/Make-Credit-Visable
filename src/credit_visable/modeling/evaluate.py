"""Evaluation utilities for binary credit risk models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


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

    return {
        "roc_auc": float(roc_auc_score(y_true_array, y_score_array)),
        "average_precision": float(
            average_precision_score(y_true_array, y_score_array)
        ),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred).tolist(),
    }
