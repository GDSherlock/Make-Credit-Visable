"""Preprocessing utilities used by baseline models and notebooks."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def split_feature_types(
    frame: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
) -> dict[str, list[str]]:
    """Split a DataFrame into numeric and categorical feature lists."""

    excluded = {column for column in [target_column, id_column] if column}
    feature_frame = frame.drop(columns=list(excluded), errors="ignore")

    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }


def build_basic_preprocessor(feature_groups: dict[str, list[str]]) -> ColumnTransformer:
    """Create a minimal sklearn preprocessor for tabular experiments.

    TODO: Extend with outlier handling, rare-category logic, and optional scaling.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_groups.get("numeric", [])),
            ("categorical", categorical_pipeline, feature_groups.get("categorical", [])),
        ],
        remainder="drop",
    )
