"""Advanced tree-based model training with optional dependencies."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


_TREE_BACKEND_SPECS: dict[str, dict[str, str]] = {
    "lightgbm": {
        "module_name": "lightgbm",
        "class_name": "LGBMClassifier",
        "install_hint": "pip install lightgbm",
    },
    "xgboost": {
        "module_name": "xgboost",
        "class_name": "XGBClassifier",
        "install_hint": "pip install xgboost",
    },
}
_TREE_BACKEND_PRIORITY = ("lightgbm", "xgboost")


@dataclass(slots=True)
class GovernedTreeTrainingOptions:
    """Configuration for the governed XGBoost training path."""

    backend_name: str = "xgboost"
    cv_folds: int = 3
    early_stopping_rounds: int = 50
    internal_validation_size: float = 0.15
    random_state: int = 42
    scoring: str = "roc_auc"
    param_grid: tuple[dict[str, Any], ...] = (
        {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2.0,
            "reg_lambda": 1.5,
        },
        {
            "n_estimators": 650,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 3.0,
            "reg_lambda": 2.0,
        },
    )


def _is_backend_installed(backend_name: str) -> bool:
    """Return whether the optional backend dependency can be imported."""

    module_name = _TREE_BACKEND_SPECS[backend_name]["module_name"]
    return importlib.util.find_spec(module_name) is not None


def _load_tree_estimator_class(backend_name: str) -> type[Any]:
    """Import the classifier class for the selected backend lazily."""

    spec = _TREE_BACKEND_SPECS[backend_name]
    module = importlib.import_module(spec["module_name"])
    return getattr(module, spec["class_name"])


def _resolve_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Estimate the positive-class weight from the observed label balance."""

    y_array = np.asarray(y_train)
    positive_count = int(np.sum(y_array == 1))
    negative_count = int(np.sum(y_array == 0))

    if positive_count == 0 or negative_count == 0:
        return 1.0

    return float(negative_count / positive_count)


def _resolve_tree_backend(model_name: str | None) -> str:
    """Choose an installed backend or raise a clear dependency error."""

    availability = get_tree_backend_availability()

    if model_name is not None:
        normalized_name = model_name.lower()
        if normalized_name not in _TREE_BACKEND_SPECS:
            supported = ", ".join(_TREE_BACKEND_PRIORITY)
            raise ValueError(
                f"Unsupported tree backend '{model_name}'. Supported backends: {supported}."
            )
        if not availability["backends"][normalized_name]["installed"]:
            install_hint = availability["backends"][normalized_name]["install_hint"]
            raise ImportError(
                f"Tree backend '{normalized_name}' is not installed. {install_hint}"
            )
        return normalized_name

    preferred_backend = availability["preferred_backend"]
    if preferred_backend is None:
        raise ImportError(
            "No supported tree-model backend is installed. "
            f"{availability['install_hint']}"
        )

    return preferred_backend


def get_tree_backend_availability() -> dict[str, Any]:
    """Describe which optional tree-model backends are available right now."""

    backend_status = {
        backend_name: {
            "installed": _is_backend_installed(backend_name),
            "install_hint": _TREE_BACKEND_SPECS[backend_name]["install_hint"],
        }
        for backend_name in _TREE_BACKEND_PRIORITY
    }
    available_backends = [
        backend_name
        for backend_name in _TREE_BACKEND_PRIORITY
        if backend_status[backend_name]["installed"]
    ]

    return {
        "backends": backend_status,
        "available_backends": available_backends,
        "preferred_backend": available_backends[0] if available_backends else None,
        "install_hint": "Install one of the optional backends: "
        + " or ".join(
            _TREE_BACKEND_SPECS[backend_name]["install_hint"]
            for backend_name in _TREE_BACKEND_PRIORITY
        ),
    }


def train_tree_model(
    X_train: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y_train: pd.Series | np.ndarray,
    model_name: str | None = None,
    random_state: int = 42,
    **model_kwargs: Any,
) -> Any:
    """Fit an optional-dependency tree model for Phase 4 comparisons."""

    backend_name = _resolve_tree_backend(model_name)
    scale_pos_weight = _resolve_scale_pos_weight(y_train)

    if backend_name == "lightgbm":
        resolved_model_kwargs = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
    else:
        resolved_model_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }

    resolved_model_kwargs.update(model_kwargs)

    estimator_class = _load_tree_estimator_class(backend_name)
    model = estimator_class(**resolved_model_kwargs)
    model.fit(X_train, y_train)
    return model


def _build_tree_model_kwargs(
    backend_name: str,
    y_train: pd.Series | np.ndarray,
    random_state: int,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    scale_pos_weight = _resolve_scale_pos_weight(y_train)

    if backend_name == "lightgbm":
        resolved_model_kwargs = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
    else:
        resolved_model_kwargs = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }

    resolved_model_kwargs.update(model_kwargs)
    return resolved_model_kwargs


def _instantiate_tree_model(
    backend_name: str,
    y_train: pd.Series | np.ndarray,
    *,
    random_state: int,
    model_kwargs: dict[str, Any],
) -> Any:
    estimator_class = _load_tree_estimator_class(backend_name)
    return estimator_class(
        **_build_tree_model_kwargs(
            backend_name,
            y_train=y_train,
            random_state=random_state,
            model_kwargs=model_kwargs,
        )
    )


def _fit_with_optional_early_stopping(
    model: Any,
    X_train: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y_train: pd.Series | np.ndarray,
    X_valid: pd.DataFrame | np.ndarray | sparse.spmatrix | None = None,
    y_valid: pd.Series | np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> Any:
    fit_kwargs: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        if early_stopping_rounds is not None:
            try:
                model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            except (AttributeError, ValueError):
                fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_governed_tree_model(
    X_dev: pd.DataFrame | np.ndarray | sparse.spmatrix,
    y_dev: pd.Series | np.ndarray,
    *,
    options: GovernedTreeTrainingOptions | None = None,
) -> dict[str, Any]:
    """Train a governed tree model using CV inside the development sample only."""

    resolved_options = options or GovernedTreeTrainingOptions()
    backend_name = _resolve_tree_backend(resolved_options.backend_name)
    if backend_name != "xgboost":
        raise NotImplementedError(
            "The governed training flow currently supports XGBoost only."
        )

    y_array = np.asarray(y_dev)
    cv = StratifiedKFold(
        n_splits=int(resolved_options.cv_folds),
        shuffle=True,
        random_state=resolved_options.random_state,
    )
    cv_rows: list[dict[str, Any]] = []

    for param_index, candidate_params in enumerate(resolved_options.param_grid, start=1):
        fold_metrics: list[dict[str, float]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(cv.split(np.zeros_like(y_array), y_array), start=1):
            X_fold_train = X_dev[train_idx]
            X_fold_valid = X_dev[valid_idx]
            y_fold_train = y_array[train_idx]
            y_fold_valid = y_array[valid_idx]

            fold_model = _instantiate_tree_model(
                backend_name,
                y_fold_train,
                random_state=resolved_options.random_state,
                model_kwargs=dict(candidate_params),
            )
            fold_model = _fit_with_optional_early_stopping(
                fold_model,
                X_fold_train,
                y_fold_train,
                X_valid=X_fold_valid,
                y_valid=y_fold_valid,
                early_stopping_rounds=resolved_options.early_stopping_rounds,
            )
            valid_score = np.asarray(fold_model.predict_proba(X_fold_valid))[:, 1]
            fold_metrics.append(
                {
                    "fold_index": fold_index,
                    "roc_auc": float(roc_auc_score(y_fold_valid, valid_score)),
                    "average_precision": float(average_precision_score(y_fold_valid, valid_score)),
                    "brier_score": float(brier_score_loss(y_fold_valid, valid_score)),
                }
            )

        fold_frame = pd.DataFrame(fold_metrics)
        cv_rows.append(
            {
                "param_index": param_index,
                "params": dict(candidate_params),
                "mean_roc_auc": float(fold_frame["roc_auc"].mean()),
                "mean_average_precision": float(fold_frame["average_precision"].mean()),
                "mean_brier_score": float(fold_frame["brier_score"].mean()),
                "std_roc_auc": float(fold_frame["roc_auc"].std(ddof=0)),
                "std_average_precision": float(fold_frame["average_precision"].std(ddof=0)),
                "std_brier_score": float(fold_frame["brier_score"].std(ddof=0)),
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values(
        by=["mean_roc_auc", "mean_average_precision", "mean_brier_score"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    best_params = dict(cv_results.iloc[0]["params"]) if not cv_results.empty else {}

    train_idx, early_stop_idx = train_test_split(
        np.arange(len(y_array)),
        test_size=resolved_options.internal_validation_size,
        stratify=y_array,
        random_state=resolved_options.random_state,
    )
    final_model = _instantiate_tree_model(
        backend_name,
        y_array[train_idx],
        random_state=resolved_options.random_state,
        model_kwargs=best_params,
    )
    final_model = _fit_with_optional_early_stopping(
        final_model,
        X_dev[train_idx],
        y_array[train_idx],
        X_valid=X_dev[early_stop_idx],
        y_valid=y_array[early_stop_idx],
        early_stopping_rounds=resolved_options.early_stopping_rounds,
    )
    dev_score = np.asarray(final_model.predict_proba(X_dev))[:, 1]

    return {
        "model": final_model,
        "backend_name": backend_name,
        "best_params": best_params,
        "cv_results": cv_results,
        "development_metrics": {
            "roc_auc": float(roc_auc_score(y_array, dev_score)),
            "average_precision": float(average_precision_score(y_array, dev_score)),
            "brier_score": float(brier_score_loss(y_array, dev_score)),
        },
        "development_score": pd.Series(dev_score),
        "training_manifest": {
            "backend_name": backend_name,
            "cv_folds": int(resolved_options.cv_folds),
            "early_stopping_rounds": int(resolved_options.early_stopping_rounds),
            "internal_validation_size": float(resolved_options.internal_validation_size),
            "best_params": best_params,
        },
    }


def train_tree_model_placeholder(
    model_name: str = "lightgbm",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Return a compatibility summary instead of silently training."""

    availability = get_tree_backend_availability()
    normalized_name = model_name.lower()
    requested_backend_supported = normalized_name in _TREE_BACKEND_SPECS
    requested_backend_installed = (
        availability["backends"][normalized_name]["installed"]
        if requested_backend_supported
        else False
    )
    selected_backend = (
        normalized_name
        if requested_backend_installed
        else availability["preferred_backend"]
    )

    install_hint = (
        availability["backends"][normalized_name]["install_hint"]
        if requested_backend_supported
        else availability["install_hint"]
    )

    return {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "ready_to_train": selected_backend is not None,
        "requested_backend_ready": requested_backend_installed,
        "selected_backend": selected_backend,
        "preferred_backend": availability["preferred_backend"],
        "available_backends": availability["available_backends"],
        "backends": availability["backends"],
        "install_hint": install_hint,
        "notes": (
            "Use train_tree_model(...) for concrete Phase 4 training. "
            "This placeholder remains as a compatibility shim for the old scaffold."
        ),
    }


def main() -> None:
    """Print the currently available optional tree-model backend."""

    availability = get_tree_backend_availability()
    preferred_backend = availability["preferred_backend"]
    if preferred_backend is None:
        print(availability["install_hint"])
    else:
        print(f"Preferred Phase 4 tree-model backend: {preferred_backend}")


if __name__ == "__main__":
    main()
