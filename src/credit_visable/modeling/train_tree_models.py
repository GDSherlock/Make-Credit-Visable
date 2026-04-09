"""Advanced tree-based model training with optional dependencies."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


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
