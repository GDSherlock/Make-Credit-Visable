"""Evaluation utilities for binary credit risk models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
