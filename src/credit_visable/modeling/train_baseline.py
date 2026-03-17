"""Baseline model training utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **model_kwargs: Any,
) -> LogisticRegression:
    """Fit a simple logistic regression baseline.

    This is intentionally minimal and assumes preprocessing has already happened.
    TODO: Wrap this into a reusable training pipeline with validation splits.
    """

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        **model_kwargs,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Future script entrypoint placeholder."""

    print("TODO: wire baseline model training to prepared feature tables.")


if __name__ == "__main__":
    main()
