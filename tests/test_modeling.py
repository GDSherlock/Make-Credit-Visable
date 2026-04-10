"""Tests for Phase 3 baseline modeling helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.modeling import (
    build_binary_diagnostic_curves,
    evaluate_binary_classifier,
    get_tree_backend_availability,
    train_logistic_baseline,
    train_tree_model,
    train_tree_model_placeholder,
)
from credit_visable.modeling import train_tree_models as tree_model_helpers


def test_train_logistic_baseline_supports_sparse_phase2_matrices() -> None:
    X, y = make_classification(
        n_samples=80,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.7, 0.3],
        random_state=42,
    )

    X_sparse = sparse.csr_matrix(X)
    y_series = pd.Series(y)

    model = train_logistic_baseline(X_sparse, y_series, random_state=42)
    predicted_pd = model.predict_proba(X_sparse)[:, 1]

    assert model.solver == "saga"
    assert predicted_pd.shape == (80,)
    assert np.all(predicted_pd >= 0.0)
    assert np.all(predicted_pd <= 1.0)


def test_evaluate_binary_classifier_preserves_public_metric_fields() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.10, 0.40, 0.35, 0.80])

    metrics = evaluate_binary_classifier(y_true, y_score, threshold=0.5)

    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "positive_rate_baseline",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "confusion_matrix",
    }
    assert metrics["threshold"] == 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_build_binary_diagnostic_curves_returns_plot_ready_payloads() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.05, 0.25, 0.45, 0.70, 0.30, 0.90])

    curves = build_binary_diagnostic_curves(y_true, y_score)

    assert set(curves) == {"roc", "precision_recall", "ks", "calibration", "gain", "lift"}
    assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"])
    assert len(curves["roc"]["thresholds"]) == len(curves["roc"]["fpr"])
    assert len(curves["precision_recall"]["precision"]) == len(
        curves["precision_recall"]["recall"]
    )
    assert len(curves["precision_recall"]["thresholds"]) == (
        len(curves["precision_recall"]["precision"]) - 1
    )
    assert len(curves["ks"]["values"]) == len(curves["ks"]["thresholds"])
    assert 0.0 <= curves["ks"]["statistic"] <= 1.0
    assert 0.0 <= curves["ks"]["threshold"] <= 1.0
    assert len(curves["calibration"]["bin_index"]) == len(curves["calibration"]["predicted_mean"])
    assert len(curves["gain"]["population_share"]) == len(curves["gain"]["captured_bad_share"])
    assert len(curves["lift"]["population_share"]) == len(curves["lift"]["lift"])


def test_get_tree_backend_availability_prefers_lightgbm_when_installed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tree_model_helpers,
        "_is_backend_installed",
        lambda backend_name: backend_name == "lightgbm",
    )

    availability = get_tree_backend_availability()

    assert availability["available_backends"] == ["lightgbm"]
    assert availability["preferred_backend"] == "lightgbm"
    assert availability["backends"]["lightgbm"]["installed"] is True
    assert availability["backends"]["xgboost"]["installed"] is False


def test_get_tree_backend_availability_falls_back_to_xgboost(monkeypatch) -> None:
    monkeypatch.setattr(
        tree_model_helpers,
        "_is_backend_installed",
        lambda backend_name: backend_name == "xgboost",
    )

    availability = get_tree_backend_availability()

    assert availability["available_backends"] == ["xgboost"]
    assert availability["preferred_backend"] == "xgboost"
    assert availability["backends"]["lightgbm"]["installed"] is False
    assert availability["backends"]["xgboost"]["installed"] is True


def test_get_tree_backend_availability_reports_missing_optional_dependencies(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tree_model_helpers,
        "_is_backend_installed",
        lambda backend_name: False,
    )

    availability = get_tree_backend_availability()

    assert availability["available_backends"] == []
    assert availability["preferred_backend"] is None
    assert "lightgbm" in availability["install_hint"]
    assert "xgboost" in availability["install_hint"]


def test_train_tree_model_supports_sparse_phase2_matrices(monkeypatch) -> None:
    fitted = {}

    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            fitted["kwargs"] = kwargs

        def fit(self, X, y):
            fitted["used_sparse_matrix"] = sparse.issparse(X)
            fitted["y_length"] = len(y)
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(X.shape[0], 0.6), np.full(X.shape[0], 0.4)])

    monkeypatch.setattr(
        tree_model_helpers,
        "_is_backend_installed",
        lambda backend_name: backend_name == "xgboost",
    )
    monkeypatch.setattr(
        tree_model_helpers,
        "_load_tree_estimator_class",
        lambda backend_name: FakeXGBClassifier,
    )

    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_redundant=1,
        weights=[0.8, 0.2],
        random_state=42,
    )

    model = train_tree_model(
        sparse.csr_matrix(X),
        pd.Series(y),
        random_state=13,
    )

    assert isinstance(model, FakeXGBClassifier)
    assert fitted["used_sparse_matrix"] is True
    assert fitted["y_length"] == 120
    assert fitted["kwargs"]["objective"] == "binary:logistic"
    assert fitted["kwargs"]["eval_metric"] == "auc"
    assert fitted["kwargs"]["tree_method"] == "hist"
    assert fitted["kwargs"]["random_state"] == 13
    assert fitted["kwargs"]["scale_pos_weight"] > 1.0


def test_train_tree_model_placeholder_reports_backend_readiness(monkeypatch) -> None:
    monkeypatch.setattr(
        tree_model_helpers,
        "_is_backend_installed",
        lambda backend_name: backend_name == "xgboost",
    )

    summary = train_tree_model_placeholder(model_name="xgboost", max_depth=4)

    assert summary["ready_to_train"] is True
    assert summary["preferred_backend"] == "xgboost"
    assert summary["model_kwargs"] == {"max_depth": 4}
    assert "train_tree_model" in summary["notes"]
