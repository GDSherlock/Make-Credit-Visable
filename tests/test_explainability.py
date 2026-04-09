"""Tests for Phase 5 explainability helpers."""

from __future__ import annotations

import importlib
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

from credit_visable.explainability import (
    build_transformed_feature_mapping,
    compute_xgboost_contribution_summary,
    get_explainability_runtime_status,
)
from credit_visable.explainability import shap_analysis


def test_get_explainability_runtime_status_handles_broken_shap_import(
    monkeypatch,
) -> None:
    original_import_module = importlib.import_module

    monkeypatch.setattr(
        shap_analysis,
        "find_spec",
        lambda module_name: object() if module_name == "shap" else None,
    )

    def fake_import_module(module_name: str):
        if module_name == "shap":
            raise ImportError("broken shap runtime")
        return original_import_module(module_name)

    monkeypatch.setattr(shap_analysis.importlib, "import_module", fake_import_module)

    runtime_status = get_explainability_runtime_status()

    assert runtime_status["shap_module_found"] is True
    assert runtime_status["shap_import_ok"] is False
    assert "broken shap runtime" in runtime_status["shap_import_error"]


def test_build_transformed_feature_mapping_recovers_raw_features_and_proxy_families() -> None:
    feature_names = [
        "numeric__EXT_SOURCE_1",
        "numeric__AMT_CREDIT",
        "categorical__NAME_FAMILY_STATUS_Married",
        "categorical__OCCUPATION_TYPE_Laborers",
    ]
    raw_feature_candidates = [
        "EXT_SOURCE_1",
        "AMT_CREDIT",
        "NAME_FAMILY_STATUS",
        "OCCUPATION_TYPE",
    ]

    mapping = build_transformed_feature_mapping(
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )

    assert mapping["raw_feature_name"].tolist() == [
        "EXT_SOURCE_1",
        "AMT_CREDIT",
        "NAME_FAMILY_STATUS",
        "OCCUPATION_TYPE",
    ]
    assert mapping["encoded_value"].tolist() == [None, None, "Married", "Laborers"]
    assert mapping["proxy_family"].tolist() == [
        "ext_source",
        "traditional_non_proxy",
        "traditional_non_proxy",
        "organization_occupation",
    ]


def test_compute_xgboost_contribution_summary_supports_sparse_inputs() -> None:
    xgboost = importlib.import_module("xgboost")

    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_sparse = sparse.csr_matrix(X)
    feature_names = [
        "numeric__EXT_SOURCE_1",
        "numeric__AMT_CREDIT",
        "categorical__NAME_FAMILY_STATUS_Married",
        "categorical__OCCUPATION_TYPE_Laborers",
    ]
    raw_feature_candidates = [
        "EXT_SOURCE_1",
        "AMT_CREDIT",
        "NAME_FAMILY_STATUS",
        "OCCUPATION_TYPE",
    ]
    model = xgboost.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )
    model.fit(X_sparse, pd.Series(y))

    summary = compute_xgboost_contribution_summary(
        model=model,
        X_matrix=X_sparse,
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
        sample_size=40,
        random_state=42,
    )

    assert summary["feature_contribution_values"].shape == (40, 4)
    assert list(summary["global_feature_contributions"].columns[:4]) == [
        "global_rank",
        "transformed_feature_name",
        "transformer_name",
        "raw_feature_name",
    ]
    assert len(summary["raw_feature_contributions"]) == 4
    assert "ext_source" in summary["proxy_family_contributions"]["proxy_family"].tolist()
    assert np.all(
        summary["global_feature_contributions"]["mean_abs_contribution"].to_numpy() >= 0.0
    )
