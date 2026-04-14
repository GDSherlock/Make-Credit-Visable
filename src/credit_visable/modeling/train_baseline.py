"""Baseline model training utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


def train_logistic_baseline(
    X_train: pd.DataFrame | sparse.spmatrix,
    y_train: pd.Series,
    random_state: int = 42,
    **model_kwargs: Any,
) -> LogisticRegression:
    """Fit a simple logistic regression baseline.

    This is intentionally minimal and assumes preprocessing has already happened.
    It supports the sparse CSR matrices produced by the Phase 2 preprocessing
    workflow as well as dense pandas inputs.
    """

    resolved_model_kwargs = {"solver": "saga", "n_jobs": -1, "tol": 1e-3}
    resolved_model_kwargs.update(model_kwargs)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
        **resolved_model_kwargs,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Future script entrypoint placeholder."""

    print("TODO: wire baseline model training to prepared feature tables.")


if __name__ == "__main__":
    main()
