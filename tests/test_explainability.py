"""Tests for Phase 5 explainability helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.explainability import (
    build_shap_plot_explanation,
    build_transformed_feature_mapping,
    compute_lime_local_explanations,
    compute_shap_contribution_summary,
    compute_shap_local_explanations,
    compute_xgboost_contribution_summary,
    get_explainability_runtime_status,
)
from credit_visable.explainability import shap_analysis


class _FakeExplanation:
    def __init__(self, values, base_values, data, feature_names):
        self.values = np.asarray(values, dtype=float)
        self.base_values = np.asarray(base_values, dtype=float)
        self.data = np.asarray(data, dtype=float)
        self.feature_names = list(feature_names)


class _FakeTreeExplainer:
    def __init__(self, model):
        self.expected_value = -0.25

    def shap_values(self, X_matrix):
        array = np.asarray(X_matrix.toarray() if sparse.issparse(X_matrix) else X_matrix)
        feature_count = array.shape[1]
        scale = np.arange(1, feature_count + 1, dtype=float)
        return array * scale


class _FakeLimeExplanation:
    def __init__(self, feature_count: int):
        self.local_exp = {1: [(feature_index, 0.2 / (feature_index + 1)) for feature_index in range(feature_count)]}
        self.intercept = np.array([0.1, 0.2], dtype=float)
        self.local_pred = np.array([0.73], dtype=float)


class _FakeDictLimeExplanation:
    def __init__(self, feature_count: int):
        self.local_exp = {1: [(feature_index, 0.2 / (feature_index + 1)) for feature_index in range(feature_count)]}
        self.intercept = {1: 0.2}
        self.local_pred = {1: 0.73}


class _FakeLimeTabularExplainer:
    def __init__(
        self,
        training_data,
        mode,
        feature_names,
        class_names,
        discretize_continuous,
        random_state,
    ):
        self.training_data = np.asarray(training_data, dtype=float)
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)

    def explain_instance(self, data_row, predict_fn, num_features, top_labels=None, labels=None):
        _ = predict_fn(np.asarray([data_row], dtype=float))
        return _FakeLimeExplanation(min(num_features, len(self.feature_names)))


def _patch_optional_modules(monkeypatch) -> None:
    original_import_module = importlib.import_module

    fake_shap = SimpleNamespace(
        TreeExplainer=_FakeTreeExplainer,
        Explanation=_FakeExplanation,
    )
    fake_lime_tabular = SimpleNamespace(LimeTabularExplainer=_FakeLimeTabularExplainer)

    def fake_import_module(module_name: str):
        if module_name == "shap":
            return fake_shap
        if module_name == "lime.lime_tabular":
            return fake_lime_tabular
        return original_import_module(module_name)

    monkeypatch.setattr(shap_analysis.importlib, "import_module", fake_import_module)


def test_get_explainability_runtime_status_handles_broken_optional_imports(
    monkeypatch,
) -> None:
    original_import_module = importlib.import_module

    monkeypatch.setattr(
        shap_analysis,
        "find_spec",
        lambda module_name: object() if module_name in {"shap", "lime"} else None,
    )

    def fake_import_module(module_name: str):
        if module_name == "shap":
            raise ImportError("broken shap runtime")
        if module_name == "lime.lime_tabular":
            raise ImportError("broken lime runtime")
        return original_import_module(module_name)

    monkeypatch.setattr(shap_analysis.importlib, "import_module", fake_import_module)

    runtime_status = get_explainability_runtime_status()

    assert runtime_status["shap_module_found"] is True
    assert runtime_status["shap_import_ok"] is False
    assert "broken shap runtime" in runtime_status["shap_import_error"]
    assert runtime_status["lime_module_found"] is True
    assert runtime_status["lime_import_ok"] is False
    assert "broken lime runtime" in runtime_status["lime_import_error"]


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


def test_compute_shap_contribution_summary_returns_expected_schema(monkeypatch) -> None:
    _patch_optional_modules(monkeypatch)

    X_matrix = np.array(
        [
            [0.2, 1.0, 0.0, 1.0],
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.2, 0.0, 1.0],
        ],
        dtype=float,
    )
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

    summary = compute_shap_contribution_summary(
        model=object(),
        X_matrix=X_matrix,
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
        sample_indices=[0, 2],
    )

    assert summary["explainability_method"] == "shap_tree_explainer"
    assert summary["sample_indices"].tolist() == [0, 2]
    assert summary["feature_contribution_values"].shape == (2, 4)
    assert isinstance(summary["shap_explanation"], _FakeExplanation)
    assert set(summary["proxy_family_contributions"]["proxy_family"]) >= {
        "ext_source",
        "organization_occupation",
    }


def test_compute_shap_local_explanations_returns_selected_cases(monkeypatch) -> None:
    _patch_optional_modules(monkeypatch)

    X_matrix = np.array(
        [
            [0.2, 1.0, 0.0, 1.0],
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.2, 0.0, 1.0],
        ],
        dtype=float,
    )
    selected_rows = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100003],
            "case_role": ["high_risk_bad", "low_risk_good"],
            "row_position": [0, 2],
            "candidate_predicted_pd": [0.82, 0.09],
        }
    )

    bundle = compute_shap_local_explanations(
        model=object(),
        X_matrix=X_matrix,
        feature_names=[
            "numeric__EXT_SOURCE_1",
            "numeric__AMT_CREDIT",
            "categorical__NAME_FAMILY_STATUS_Married",
            "categorical__OCCUPATION_TYPE_Laborers",
        ],
        raw_feature_candidates=[
            "EXT_SOURCE_1",
            "AMT_CREDIT",
            "NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE",
        ],
        selected_rows=selected_rows,
        top_n_features=2,
    )

    local_frame = bundle["local_case_explanations"]
    assert set(local_frame["SK_ID_CURR"]) == {100001, 100003}
    assert local_frame.groupby("case_role")["feature_rank"].max().tolist() == [2, 2]
    assert "feature_value" in local_frame.columns
    assert isinstance(bundle["shap_explanation"], _FakeExplanation)


def test_compute_lime_local_explanations_returns_stable_schema(monkeypatch) -> None:
    _patch_optional_modules(monkeypatch)

    class DummyModel:
        def predict_proba(self, X):
            array = np.asarray(X, dtype=float)
            positive = np.clip(array[:, 0] * 0.2 + 0.5, 0.01, 0.99)
            negative = 1.0 - positive
            return np.column_stack([negative, positive])

    selected_rows = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100003],
            "case_role": ["high_risk_bad", "low_risk_good"],
            "row_position": [0, 2],
            "candidate_predicted_pd": [0.82, 0.09],
        }
    )

    lime_frame = compute_lime_local_explanations(
        model=DummyModel(),
        X_train_matrix=np.array(
            [
                [0.1, 1.1, 0.0, 1.0],
                [0.2, 0.8, 1.0, 0.0],
                [0.4, 0.2, 0.0, 1.0],
                [0.5, 0.1, 1.0, 0.0],
            ],
            dtype=float,
        ),
        X_explain_matrix=np.array(
            [
                [0.2, 1.0, 0.0, 1.0],
                [0.3, 0.5, 1.0, 0.0],
                [0.4, 0.2, 0.0, 1.0],
            ],
            dtype=float,
        ),
        feature_names=[
            "numeric__EXT_SOURCE_1",
            "numeric__AMT_CREDIT",
            "categorical__NAME_FAMILY_STATUS_Married",
            "categorical__OCCUPATION_TYPE_Laborers",
        ],
        raw_feature_candidates=[
            "EXT_SOURCE_1",
            "AMT_CREDIT",
            "NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE",
        ],
        selected_rows=selected_rows,
        num_features=3,
        random_state=42,
    )

    assert set(lime_frame["SK_ID_CURR"]) == {100001, 100003}
    assert set(["local_weight", "abs_local_weight", "predicted_probability"]).issubset(
        lime_frame.columns
    )
    assert lime_frame.groupby("case_role")["feature_rank"].max().tolist() == [3, 3]


def test_compute_lime_local_explanations_supports_dict_intercept(monkeypatch) -> None:
    original_import_module = importlib.import_module

    class FakeDictLimeTabularExplainer(_FakeLimeTabularExplainer):
        def explain_instance(self, data_row, predict_fn, num_features, top_labels=None, labels=None):
            _ = predict_fn(np.asarray([data_row], dtype=float))
            return _FakeDictLimeExplanation(min(num_features, len(self.feature_names)))

    def fake_import_module(module_name: str):
        if module_name == "lime.lime_tabular":
            return SimpleNamespace(LimeTabularExplainer=FakeDictLimeTabularExplainer)
        return original_import_module(module_name)

    monkeypatch.setattr(shap_analysis.importlib, "import_module", fake_import_module)

    class DummyModel:
        def predict_proba(self, X):
            array = np.asarray(X, dtype=float)
            positive = np.clip(array[:, 0] * 0.2 + 0.5, 0.01, 0.99)
            negative = 1.0 - positive
            return np.column_stack([negative, positive])

    selected_rows = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "case_role": ["high_risk_bad"],
            "row_position": [0],
        }
    )

    lime_frame = compute_lime_local_explanations(
        model=DummyModel(),
        X_train_matrix=np.array([[0.1, 1.1], [0.2, 0.8]], dtype=float),
        X_explain_matrix=np.array([[0.2, 1.0]], dtype=float),
        feature_names=["numeric__EXT_SOURCE_1", "numeric__AMT_CREDIT"],
        raw_feature_candidates=["EXT_SOURCE_1", "AMT_CREDIT"],
        selected_rows=selected_rows,
        num_features=2,
        random_state=42,
    )

    assert lime_frame["intercept"].tolist() == [0.2, 0.2]
    assert lime_frame["local_prediction"].tolist() == [0.73, 0.73]


def test_build_shap_plot_explanation_uses_feature_names(monkeypatch) -> None:
    _patch_optional_modules(monkeypatch)

    explanation = build_shap_plot_explanation(
        feature_contribution_values=np.array([[0.1, -0.2]]),
        base_values=np.array([0.5]),
        feature_data=np.array([[1.0, 2.0]]),
        feature_names=["numeric__A", "numeric__B"],
    )

    assert explanation.feature_names == ["numeric__A", "numeric__B"]
    assert explanation.values.shape == (1, 2)
