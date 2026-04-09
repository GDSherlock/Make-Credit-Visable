"""Explainability helpers for Phase 5 notebook workflows."""

from __future__ import annotations

import importlib
from importlib.util import find_spec
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance

from credit_visable.features.feature_sets import is_proxy_feature


_SOCIAL_CIRCLE_PATTERN = re.compile(r"^(OBS|DEF)_\d+_CNT_SOCIAL_CIRCLE$")
_DOCUMENT_PATTERN = re.compile(r"^FLAG_DOCUMENT_\d+$")


def get_explainability_runtime_status() -> dict[str, Any]:
    """Describe whether optional explainability backends can actually run."""

    shap_module_found = find_spec("shap") is not None
    shap_import_ok = False
    shap_import_error = None

    if shap_module_found:
        try:
            importlib.import_module("shap")
            shap_import_ok = True
        except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
            shap_import_error = f"{type(exc).__name__}: {exc}"

    lime_module_found = find_spec("lime") is not None
    lime_import_ok = False
    lime_import_error = None

    if lime_module_found:
        try:
            importlib.import_module("lime.lime_tabular")
            lime_import_ok = True
        except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
            lime_import_error = f"{type(exc).__name__}: {exc}"

    xgboost_contrib_ok = False
    try:
        xgboost_module = importlib.import_module("xgboost")
        xgboost_contrib_ok = hasattr(xgboost_module, "DMatrix")
    except Exception:
        xgboost_contrib_ok = False

    return {
        "shap_module_found": shap_module_found,
        "shap_import_ok": shap_import_ok,
        "shap_import_error": shap_import_error,
        "lime_module_found": lime_module_found,
        "lime_import_ok": lime_import_ok,
        "lime_import_error": lime_import_error,
        "xgboost_contrib_ok": xgboost_contrib_ok,
    }


def resolve_proxy_family(raw_feature_name: str) -> str:
    """Bucket raw features into governance-sensitive proxy families."""

    if raw_feature_name.startswith("EXT_SOURCE_"):
        return "ext_source"
    if _SOCIAL_CIRCLE_PATTERN.fullmatch(raw_feature_name):
        return "social_circle"
    if raw_feature_name in {
        "FLAG_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_EMP_PHONE",
        "FLAG_EMAIL",
        "DAYS_LAST_PHONE_CHANGE",
    }:
        return "contactability"
    if "REGION" in raw_feature_name or "_CITY_" in raw_feature_name:
        return "region_city"
    if raw_feature_name in {"ORGANIZATION_TYPE", "OCCUPATION_TYPE"}:
        return "organization_occupation"
    if _DOCUMENT_PATTERN.fullmatch(raw_feature_name):
        return "document_flags"
    if is_proxy_feature(raw_feature_name):
        return "other_proxy"
    return "traditional_non_proxy"


def _resolve_raw_feature_name(
    transformed_feature_name: str,
    raw_feature_candidates: list[str],
) -> tuple[str, str, str | None]:
    if "__" in transformed_feature_name:
        transformer_name, encoded_name = transformed_feature_name.split("__", 1)
    else:
        transformer_name, encoded_name = "unknown", transformed_feature_name

    if transformer_name != "categorical":
        return transformer_name, encoded_name, None

    for candidate in raw_feature_candidates:
        if encoded_name == candidate:
            return transformer_name, candidate, None
        if encoded_name.startswith(f"{candidate}_"):
            return transformer_name, candidate, encoded_name[len(candidate) + 1 :]

    if "_" in encoded_name:
        fallback_raw, encoded_value = encoded_name.rsplit("_", 1)
        return transformer_name, fallback_raw, encoded_value

    return transformer_name, encoded_name, None


def build_transformed_feature_mapping(
    feature_names: list[str] | pd.Series,
    raw_feature_candidates: list[str] | None = None,
) -> pd.DataFrame:
    """Map transformed sklearn feature names back to raw Home Credit columns."""

    resolved_feature_names = (
        feature_names.astype(str).tolist()
        if isinstance(feature_names, pd.Series)
        else [str(feature_name) for feature_name in feature_names]
    )
    candidate_list = sorted(
        {str(candidate) for candidate in (raw_feature_candidates or [])},
        key=len,
        reverse=True,
    )

    rows = []
    for transformed_feature_name in resolved_feature_names:
        transformer_name, raw_feature_name, encoded_value = _resolve_raw_feature_name(
            transformed_feature_name,
            candidate_list,
        )
        rows.append(
            {
                "transformed_feature_name": transformed_feature_name,
                "transformer_name": transformer_name,
                "raw_feature_name": raw_feature_name,
                "encoded_value": encoded_value,
                "is_proxy_feature": bool(is_proxy_feature(raw_feature_name)),
                "proxy_family": resolve_proxy_family(raw_feature_name),
            }
        )

    return pd.DataFrame(rows)


def summarize_contribution_values(
    contribution_values: np.ndarray,
    feature_names: list[str] | pd.Series,
    raw_feature_candidates: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build transformed, raw, and proxy-family contribution summaries."""

    contribution_array = np.asarray(contribution_values, dtype=float)
    if contribution_array.ndim != 2:
        raise ValueError("contribution_values must be a 2D array.")

    mapping = build_transformed_feature_mapping(
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    if contribution_array.shape[1] != len(mapping):
        raise ValueError(
            "Contribution column count does not match the number of feature names."
        )

    global_feature_contributions = mapping.copy()
    global_feature_contributions["mean_abs_contribution"] = np.abs(
        contribution_array
    ).mean(axis=0)
    global_feature_contributions["mean_contribution"] = contribution_array.mean(axis=0)
    global_feature_contributions["non_zero_share"] = (
        np.abs(contribution_array) > 1e-12
    ).mean(axis=0)
    global_feature_contributions = global_feature_contributions.sort_values(
        "mean_abs_contribution",
        ascending=False,
    ).reset_index(drop=True)
    global_feature_contributions.insert(
        0,
        "global_rank",
        np.arange(1, len(global_feature_contributions) + 1),
    )

    raw_feature_contributions = (
        global_feature_contributions.groupby(
            ["raw_feature_name", "is_proxy_feature", "proxy_family"],
            dropna=False,
        )
        .agg(
            transformed_feature_count=("transformed_feature_name", "count"),
            mean_abs_contribution=("mean_abs_contribution", "sum"),
            mean_contribution=("mean_contribution", "sum"),
            max_non_zero_share=("non_zero_share", "max"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    raw_feature_contributions.insert(
        0,
        "raw_feature_rank",
        np.arange(1, len(raw_feature_contributions) + 1),
    )

    proxy_family_contributions = (
        raw_feature_contributions.groupby("proxy_family", dropna=False)
        .agg(
            raw_feature_count=("raw_feature_name", "count"),
            mean_abs_contribution=("mean_abs_contribution", "sum"),
            mean_contribution=("mean_contribution", "sum"),
            proxy_feature_count=("is_proxy_feature", "sum"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    proxy_family_contributions.insert(
        0,
        "proxy_family_rank",
        np.arange(1, len(proxy_family_contributions) + 1),
    )

    return {
        "feature_mapping": mapping,
        "global_feature_contributions": global_feature_contributions,
        "raw_feature_contributions": raw_feature_contributions,
        "proxy_family_contributions": proxy_family_contributions,
    }


def _subset_matrix_rows(
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    row_indices: np.ndarray,
) -> pd.DataFrame | np.ndarray | sparse.spmatrix:
    if isinstance(X_matrix, pd.DataFrame):
        return X_matrix.iloc[row_indices]
    return X_matrix[row_indices]


def _sample_matrix_rows(
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    sample_size: int | None,
    random_state: int,
) -> tuple[pd.DataFrame | np.ndarray | sparse.spmatrix, np.ndarray]:
    row_count = X_matrix.shape[0]
    sample_count = row_count if sample_size is None else min(int(sample_size), row_count)
    sample_indices = np.arange(row_count)

    if sample_count < row_count:
        sample_indices = np.random.default_rng(random_state).choice(
            row_count,
            size=sample_count,
            replace=False,
        )
        sample_indices = np.sort(sample_indices)

    return _subset_matrix_rows(X_matrix, sample_indices), sample_indices


def _resolve_sample_rows(
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    sample_size: int | None = None,
    random_state: int = 42,
    sample_indices: np.ndarray | list[int] | None = None,
) -> tuple[pd.DataFrame | np.ndarray | sparse.spmatrix, np.ndarray]:
    if sample_indices is not None:
        resolved_indices = np.asarray(sample_indices, dtype=int)
        return _subset_matrix_rows(X_matrix, resolved_indices), resolved_indices
    return _sample_matrix_rows(
        X_matrix=X_matrix,
        sample_size=sample_size,
        random_state=random_state,
    )


def _matrix_to_numpy(
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
) -> np.ndarray:
    if sparse.issparse(X_matrix):
        return X_matrix.toarray()
    if isinstance(X_matrix, pd.DataFrame):
        return X_matrix.to_numpy()
    return np.asarray(X_matrix)


def _is_xgboost_model(model: Any) -> bool:
    module_name = getattr(model.__class__, "__module__", "")
    return module_name.startswith("xgboost")


def _normalize_base_values(base_values: Any, row_count: int) -> np.ndarray:
    base_array = np.asarray(base_values, dtype=float)
    if base_array.ndim == 0:
        return np.full(row_count, float(base_array))
    if base_array.size == row_count:
        return base_array.reshape(row_count).astype(float)
    return np.full(row_count, float(base_array.reshape(-1)[-1]))


def _resolve_binary_explanation(
    contribution_values: Any,
    base_values: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(contribution_values, (list, tuple)):
        feature_contributions = np.asarray(contribution_values[-1], dtype=float)
        return feature_contributions, _normalize_base_values(
            np.asarray(base_values).reshape(-1)[-1] if np.asarray(base_values).ndim > 0 else base_values,
            feature_contributions.shape[0],
        )

    contribution_array = np.asarray(contribution_values, dtype=float)
    if contribution_array.ndim == 3:
        feature_contributions = contribution_array[:, :, -1]
    else:
        feature_contributions = contribution_array

    return feature_contributions, _normalize_base_values(
        base_values,
        feature_contributions.shape[0],
    )


def build_shap_plot_explanation(
    feature_contribution_values: np.ndarray,
    base_values: np.ndarray,
    feature_data: np.ndarray,
    feature_names: list[str] | pd.Series,
) -> Any:
    """Build a SHAP Explanation object suitable for plotting APIs."""

    shap_module = importlib.import_module("shap")
    resolved_feature_names = (
        feature_names.astype(str).tolist()
        if isinstance(feature_names, pd.Series)
        else [str(feature_name) for feature_name in feature_names]
    )
    return shap_module.Explanation(
        values=np.asarray(feature_contribution_values, dtype=float),
        base_values=np.asarray(base_values, dtype=float),
        data=np.asarray(feature_data, dtype=float),
        feature_names=resolved_feature_names,
    )


def compute_xgboost_contribution_summary(
    model: Any,
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | pd.Series,
    raw_feature_candidates: list[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
    sample_indices: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    """Compute TreeSHAP-style contribution summaries via XGBoost pred_contribs."""

    if not _is_xgboost_model(model):
        raise TypeError("compute_xgboost_contribution_summary requires an XGBoost model.")

    xgboost_module = importlib.import_module("xgboost")
    X_sample, resolved_indices = _resolve_sample_rows(
        X_matrix=X_matrix,
        sample_size=sample_size,
        random_state=random_state,
        sample_indices=sample_indices,
    )
    X_sample_array = _matrix_to_numpy(X_sample)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgboost_module.DMatrix(X_sample, feature_names=list(feature_names))
    contribution_matrix = booster.predict(dmatrix, pred_contribs=True)
    base_values = contribution_matrix[:, -1]
    feature_contributions = contribution_matrix[:, :-1]

    summary = summarize_contribution_values(
        contribution_values=feature_contributions,
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    try:
        shap_explanation = build_shap_plot_explanation(
            feature_contribution_values=feature_contributions,
            base_values=base_values,
            feature_data=X_sample_array,
            feature_names=feature_names,
        )
    except Exception:  # pragma: no cover - diagnostic fallback only
        shap_explanation = None
    summary.update(
        {
            "sample_indices": resolved_indices,
            "base_values": base_values,
            "feature_contribution_values": feature_contributions,
            "feature_data": X_sample_array,
            "shap_explanation": shap_explanation,
            "explainability_method": "xgboost_pred_contribs",
        }
    )
    return summary


def compute_shap_contribution_summary(
    model: Any,
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | pd.Series,
    raw_feature_candidates: list[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
    sample_indices: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    """Compute global SHAP contribution summaries via shap.TreeExplainer."""

    shap_module = importlib.import_module("shap")
    X_sample, resolved_indices = _resolve_sample_rows(
        X_matrix=X_matrix,
        sample_size=sample_size,
        random_state=random_state,
        sample_indices=sample_indices,
    )
    X_sample_array = _matrix_to_numpy(X_sample)
    explainer = shap_module.TreeExplainer(model)
    contribution_values = explainer.shap_values(X_sample)
    feature_contributions, base_values = _resolve_binary_explanation(
        contribution_values=contribution_values,
        base_values=getattr(explainer, "expected_value", 0.0),
    )

    summary = summarize_contribution_values(
        contribution_values=feature_contributions,
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    summary.update(
        {
            "sample_indices": resolved_indices,
            "base_values": base_values,
            "feature_contribution_values": feature_contributions,
            "feature_data": X_sample_array,
            "shap_explanation": build_shap_plot_explanation(
                feature_contribution_values=feature_contributions,
                base_values=base_values,
                feature_data=X_sample_array,
                feature_names=feature_names,
            ),
            "explainability_method": "shap_tree_explainer",
        }
    )
    return summary


def compute_permutation_importance_summary(
    model: Any,
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y_true: pd.Series | np.ndarray,
    feature_names: list[str] | pd.Series,
    raw_feature_candidates: list[str] | None = None,
    sample_size: int | None = None,
    scoring: str = "roc_auc",
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fallback global explainability summary when local SHAP-like values are unavailable."""

    X_sample, sample_indices = _sample_matrix_rows(
        X_matrix=X_matrix,
        sample_size=sample_size,
        random_state=random_state,
    )
    y_array = np.asarray(y_true)
    y_sample = y_array[sample_indices]
    result = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    contribution_values = np.tile(result.importances_mean, (len(sample_indices), 1))
    summary = summarize_contribution_values(
        contribution_values=contribution_values,
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    std_map = pd.DataFrame(
        {
            "transformed_feature_name": list(feature_names),
            "importance_std": result.importances_std,
        }
    )
    summary["global_feature_contributions"] = summary["global_feature_contributions"].merge(
        std_map,
        on="transformed_feature_name",
        how="left",
    )
    summary["raw_feature_contributions"] = (
        summary["global_feature_contributions"]
        .groupby(["raw_feature_name", "is_proxy_feature", "proxy_family"], dropna=False)
        .agg(
            transformed_feature_count=("transformed_feature_name", "count"),
            mean_abs_contribution=("mean_abs_contribution", "sum"),
            mean_contribution=("mean_contribution", "sum"),
            importance_std=("importance_std", "max"),
            max_non_zero_share=("non_zero_share", "max"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    summary["raw_feature_contributions"].insert(
        0,
        "raw_feature_rank",
        np.arange(1, len(summary["raw_feature_contributions"]) + 1),
    )
    summary["proxy_family_contributions"] = (
        summary["raw_feature_contributions"]
        .groupby("proxy_family", dropna=False)
        .agg(
            raw_feature_count=("raw_feature_name", "count"),
            mean_abs_contribution=("mean_abs_contribution", "sum"),
            mean_contribution=("mean_contribution", "sum"),
            importance_std=("importance_std", "max"),
            proxy_feature_count=("is_proxy_feature", "sum"),
        )
        .reset_index()
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
    summary["proxy_family_contributions"].insert(
        0,
        "proxy_family_rank",
        np.arange(1, len(summary["proxy_family_contributions"]) + 1),
    )
    summary.update(
        {
            "sample_indices": sample_indices,
            "base_values": np.zeros(len(sample_indices), dtype=float),
            "feature_contribution_values": contribution_values,
            "feature_data": _matrix_to_numpy(X_sample),
            "explainability_method": "permutation_importance",
        }
    )
    return summary


def select_local_explanation_rows(
    validation_frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    threshold: float = 0.5,
    id_column: str | None = None,
    num_cases: int = 3,
) -> pd.DataFrame:
    """Pick representative validation rows for local explanation review."""

    required_columns = {score_column, target_column}
    missing = required_columns - set(validation_frame.columns)
    if missing:
        raise KeyError(f"Validation frame missing required columns: {sorted(missing)}")

    candidate_frame = validation_frame.reset_index(drop=True).copy()
    candidate_frame["row_position"] = np.arange(len(candidate_frame))
    selected_rows: list[pd.Series] = []
    used_positions: set[int] = set()

    selection_specs = [
        (
            "high_risk_bad",
            candidate_frame[candidate_frame[target_column] == 1].sort_values(
                score_column,
                ascending=False,
            ),
        ),
        (
            "borderline_case",
            candidate_frame.assign(
                distance_to_threshold=(candidate_frame[score_column] - threshold).abs()
            ).sort_values(["distance_to_threshold", score_column], ascending=[True, False]),
        ),
        (
            "low_risk_good",
            candidate_frame[candidate_frame[target_column] == 0].sort_values(
                score_column,
                ascending=True,
            ),
        ),
    ]

    for case_role, candidates in selection_specs:
        for _, row in candidates.iterrows():
            row_position = int(row["row_position"])
            if row_position in used_positions:
                continue
            payload = row.copy()
            payload["case_role"] = case_role
            selected_rows.append(payload)
            used_positions.add(row_position)
            break
        if len(selected_rows) >= num_cases:
            break

    if len(selected_rows) < num_cases:
        remaining_candidates = candidate_frame.sort_values(score_column, ascending=False)
        for _, row in remaining_candidates.iterrows():
            row_position = int(row["row_position"])
            if row_position in used_positions:
                continue
            payload = row.copy()
            payload["case_role"] = f"additional_case_{len(selected_rows) + 1}"
            selected_rows.append(payload)
            used_positions.add(row_position)
            if len(selected_rows) >= num_cases:
                break

    selection = pd.DataFrame(selected_rows)
    if selection.empty:
        columns = ["row_position", "case_role", score_column, target_column]
        if id_column is not None:
            columns.insert(0, id_column)
        return pd.DataFrame(columns=columns)

    preferred_columns = ["case_role", "row_position"]
    if id_column is not None and id_column in selection.columns:
        preferred_columns.insert(0, id_column)
    preferred_columns.extend([score_column, target_column])
    remaining_columns = [
        column for column in selection.columns if column not in preferred_columns
    ]
    return selection.loc[:, preferred_columns + remaining_columns].reset_index(drop=True)


def _build_local_case_frame(
    selected_rows: pd.DataFrame,
    feature_mapping: pd.DataFrame,
    feature_contributions: np.ndarray,
    feature_values: np.ndarray,
    base_values: np.ndarray,
    top_n_features: int,
    value_column_name: str,
) -> pd.DataFrame:
    rows = []
    for local_index, (_, selection_row) in enumerate(selected_rows.iterrows()):
        ranked_indices = np.argsort(np.abs(feature_contributions[local_index]))[::-1][
            :top_n_features
        ]
        predicted_margin = float(
            base_values[local_index] + feature_contributions[local_index].sum()
        )
        for feature_rank, feature_index in enumerate(ranked_indices, start=1):
            feature_row = feature_mapping.iloc[int(feature_index)]
            rows.append(
                {
                    "case_role": selection_row["case_role"],
                    "row_position": int(selection_row["row_position"]),
                    "feature_rank": feature_rank,
                    "transformed_feature_name": feature_row["transformed_feature_name"],
                    "raw_feature_name": feature_row["raw_feature_name"],
                    "encoded_value": feature_row["encoded_value"],
                    "proxy_family": feature_row["proxy_family"],
                    "is_proxy_feature": bool(feature_row["is_proxy_feature"]),
                    value_column_name: float(feature_contributions[local_index, feature_index]),
                    f"abs_{value_column_name}": float(
                        abs(feature_contributions[local_index, feature_index])
                    ),
                    "feature_value": float(feature_values[local_index, feature_index]),
                    "base_value": float(base_values[local_index]),
                    "predicted_margin": predicted_margin,
                }
            )

    local_frame = pd.DataFrame(rows)
    passthrough_columns = [
        column for column in selected_rows.columns if column not in {"case_role", "row_position"}
    ]
    if passthrough_columns and not local_frame.empty:
        local_frame = local_frame.merge(
            selected_rows[["case_role", "row_position", *passthrough_columns]],
            on=["case_role", "row_position"],
            how="left",
        )
    return local_frame


def compute_xgboost_local_explanations(
    model: Any,
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | pd.Series,
    selected_rows: pd.DataFrame,
    raw_feature_candidates: list[str] | None = None,
    top_n_features: int = 10,
) -> pd.DataFrame:
    """Build a long-form local contribution table for selected validation rows."""

    if selected_rows.empty:
        return pd.DataFrame()

    xgboost_module = importlib.import_module("xgboost")
    row_positions = selected_rows["row_position"].astype(int).to_numpy()
    X_selected = _subset_matrix_rows(X_matrix, row_positions)
    X_selected_array = _matrix_to_numpy(X_selected)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgboost_module.DMatrix(
        X_selected,
        feature_names=list(feature_names),
    )
    contribution_matrix = booster.predict(dmatrix, pred_contribs=True)
    base_values = contribution_matrix[:, -1]
    feature_contributions = contribution_matrix[:, :-1]
    feature_mapping = build_transformed_feature_mapping(
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )

    return _build_local_case_frame(
        selected_rows=selected_rows,
        feature_mapping=feature_mapping,
        feature_contributions=feature_contributions,
        feature_values=X_selected_array,
        base_values=base_values,
        top_n_features=top_n_features,
        value_column_name="contribution",
    )


def compute_shap_local_explanations(
    model: Any,
    X_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | pd.Series,
    selected_rows: pd.DataFrame,
    raw_feature_candidates: list[str] | None = None,
    top_n_features: int = 10,
) -> dict[str, Any]:
    """Build SHAP local explanations for selected validation rows."""

    if selected_rows.empty:
        return {
            "local_case_explanations": pd.DataFrame(),
            "selected_rows": selected_rows.copy(),
            "base_values": np.asarray([], dtype=float),
            "feature_contribution_values": np.empty((0, 0), dtype=float),
            "feature_data": np.empty((0, 0), dtype=float),
            "shap_explanation": None,
        }

    shap_module = importlib.import_module("shap")
    row_positions = selected_rows["row_position"].astype(int).to_numpy()
    X_selected = _subset_matrix_rows(X_matrix, row_positions)
    X_selected_array = _matrix_to_numpy(X_selected)
    explainer = shap_module.TreeExplainer(model)
    contribution_values = explainer.shap_values(X_selected)
    feature_contributions, base_values = _resolve_binary_explanation(
        contribution_values=contribution_values,
        base_values=getattr(explainer, "expected_value", 0.0),
    )
    feature_mapping = build_transformed_feature_mapping(
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    local_frame = _build_local_case_frame(
        selected_rows=selected_rows,
        feature_mapping=feature_mapping,
        feature_contributions=feature_contributions,
        feature_values=X_selected_array,
        base_values=base_values,
        top_n_features=top_n_features,
        value_column_name="contribution",
    )

    return {
        "local_case_explanations": local_frame,
        "selected_rows": selected_rows.reset_index(drop=True).copy(),
        "base_values": base_values,
        "feature_contribution_values": feature_contributions,
        "feature_data": X_selected_array,
        "shap_explanation": build_shap_plot_explanation(
            feature_contribution_values=feature_contributions,
            base_values=base_values,
            feature_data=X_selected_array,
            feature_names=feature_names,
        ),
    }


def _resolve_label_specific_value(
    value: Any,
    target_label: int,
) -> float | None:
    """Extract a scalar from LIME outputs that may be dict-, list-, or scalar-shaped."""

    if value is None:
        return None

    if isinstance(value, dict):
        for candidate_key in (target_label, str(target_label)):
            if candidate_key in value:
                return float(value[candidate_key])
        if value:
            return float(next(iter(value.values())))
        return None

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    if array.size > target_label:
        return float(array[target_label])
    return float(array[-1])


def compute_lime_local_explanations(
    model: Any,
    X_train_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    X_explain_matrix: pd.DataFrame | np.ndarray | sparse.spmatrix,
    feature_names: list[str] | pd.Series,
    selected_rows: pd.DataFrame,
    raw_feature_candidates: list[str] | None = None,
    training_sample_size: int | None = 5000,
    num_features: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build LIME local explanations for selected validation rows."""

    if selected_rows.empty:
        return pd.DataFrame()

    lime_tabular = importlib.import_module("lime.lime_tabular")
    training_sample, _ = _sample_matrix_rows(
        X_matrix=X_train_matrix,
        sample_size=training_sample_size,
        random_state=random_state,
    )
    training_array = _matrix_to_numpy(training_sample)
    row_positions = selected_rows["row_position"].astype(int).to_numpy()
    explain_array = _matrix_to_numpy(_subset_matrix_rows(X_explain_matrix, row_positions))
    feature_mapping = build_transformed_feature_mapping(
        feature_names=feature_names,
        raw_feature_candidates=raw_feature_candidates,
    )
    resolved_feature_names = (
        feature_names.astype(str).tolist()
        if isinstance(feature_names, pd.Series)
        else [str(feature_name) for feature_name in feature_names]
    )

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_array,
        mode="classification",
        feature_names=resolved_feature_names,
        class_names=["good", "bad"],
        discretize_continuous=False,
        random_state=random_state,
    )

    prediction_matrix = np.asarray(model.predict_proba(explain_array))
    rows = []
    for local_index, (_, selection_row) in enumerate(selected_rows.iterrows()):
        prediction_row = prediction_matrix[local_index]
        target_label = int(min(1, prediction_row.shape[0] - 1))
        explanation = explainer.explain_instance(
            data_row=explain_array[local_index],
            predict_fn=model.predict_proba,
            num_features=min(num_features, len(resolved_feature_names)),
            labels=(target_label,),
        )
        ranked_pairs_source = explanation.local_exp.get(target_label, [])
        if not ranked_pairs_source and explanation.local_exp:
            ranked_pairs_source = next(iter(explanation.local_exp.values()))
        ranked_pairs = sorted(
            ranked_pairs_source,
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:num_features]
        intercept = _resolve_label_specific_value(explanation.intercept, target_label)
        local_prediction = _resolve_label_specific_value(explanation.local_pred, 0)

        for feature_rank, (feature_index, local_weight) in enumerate(ranked_pairs, start=1):
            feature_row = feature_mapping.iloc[int(feature_index)]
            rows.append(
                {
                    "case_role": selection_row["case_role"],
                    "row_position": int(selection_row["row_position"]),
                    "feature_rank": feature_rank,
                    "transformed_feature_name": feature_row["transformed_feature_name"],
                    "raw_feature_name": feature_row["raw_feature_name"],
                    "encoded_value": feature_row["encoded_value"],
                    "proxy_family": feature_row["proxy_family"],
                    "is_proxy_feature": bool(feature_row["is_proxy_feature"]),
                    "feature_value": float(explain_array[local_index, feature_index]),
                    "local_weight": float(local_weight),
                    "abs_local_weight": float(abs(local_weight)),
                    "class_index": target_label,
                    "predicted_probability": float(prediction_row[target_label]),
                    "intercept": intercept,
                    "local_prediction": local_prediction,
                }
            )

    lime_frame = pd.DataFrame(rows)
    passthrough_columns = [
        column for column in selected_rows.columns if column not in {"case_role", "row_position"}
    ]
    if passthrough_columns and not lime_frame.empty:
        lime_frame = lime_frame.merge(
            selected_rows[["case_role", "row_position", *passthrough_columns]],
            on=["case_role", "row_position"],
            how="left",
        )
    return lime_frame


def run_shap_placeholder(
    model: Any,
    X_sample: Any,
    max_rows: int = 1000,
) -> dict[str, Any]:
    """Keep the original placeholder contract while exposing runtime details."""

    runtime_status = get_explainability_runtime_status()
    return {
        "package_available": runtime_status["shap_module_found"],
        "max_rows": max_rows,
        "ready": runtime_status["shap_import_ok"],
        "runtime_status": runtime_status,
        "notes": (
            "Use the Phase 5 helpers for working explainability flows. "
            "The placeholder remains for backward compatibility."
        ),
    }
