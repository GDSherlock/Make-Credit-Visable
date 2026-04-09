"""Preprocessing utilities used by baseline models and notebooks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_visable.utils.paths import get_paths


@dataclass(slots=True)
class PreprocessingOptions:
    """Stable defaults for the Phase 2 preprocessing workflow."""

    validation_size: float = 0.2
    rare_category_min_frequency: float | int = 0.01
    scale_numeric: bool = False
    clip_quantiles: tuple[float, float] | None = None
    random_state: int = 42


@dataclass(slots=True)
class PreparedPreprocessingArtifacts:
    """Reusable outputs produced by the Phase 2 preprocessing pipeline."""

    options: PreprocessingOptions
    feature_groups: dict[str, list[str]]
    preprocessor: ColumnTransformer
    X_train: sparse.csr_matrix
    X_valid: sparse.csr_matrix
    y_train: pd.Series
    y_valid: pd.Series
    train_ids: pd.Series | None
    valid_ids: pd.Series | None
    feature_names: list[str]
    target_column: str
    id_column: str | None


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to fitted quantile bounds when requested."""

    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QuantileClipper":
        frame = _coerce_frame(X)
        self.columns_ = frame.columns.tolist()
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _coerce_frame(X, columns=getattr(self, "columns_", None))
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)


def _coerce_frame(
    values: pd.DataFrame | pd.Series | object,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Preserve DataFrame column names when sklearn sends ndarray-like values."""

    if isinstance(values, pd.DataFrame):
        return values.copy()
    if isinstance(values, pd.Series):
        return values.to_frame()
    return pd.DataFrame(values, columns=columns)


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


def _build_numeric_pipeline(options: PreprocessingOptions) -> Pipeline:
    steps: list[tuple[str, object]] = []

    if options.clip_quantiles is not None:
        lower_quantile, upper_quantile = options.clip_quantiles
        steps.append(
            ("clipper", QuantileClipper(lower_quantile=lower_quantile, upper_quantile=upper_quantile))
        )

    steps.append(("imputer", SimpleImputer(strategy="median")))

    if options.scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_basic_preprocessor(
    feature_groups: dict[str, list[str]],
    options: PreprocessingOptions | None = None,
) -> ColumnTransformer:
    """Create the default preprocessing pipeline used in Phase 2."""

    resolved_options = options or PreprocessingOptions()

    numeric_pipeline = _build_numeric_pipeline(resolved_options)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=resolved_options.rare_category_min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_groups.get("numeric", [])),
            ("categorical", categorical_pipeline, feature_groups.get("categorical", [])),
        ],
        remainder="drop",
    )


def prepare_preprocessing_artifacts(
    frame: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
    options: PreprocessingOptions | None = None,
) -> PreparedPreprocessingArtifacts:
    """Split a raw frame, fit the default preprocessor, and return train/valid artifacts."""

    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the input frame.")

    resolved_options = options or PreprocessingOptions()
    if not 0 < resolved_options.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    train_index, valid_index = train_test_split(
        frame.index.to_numpy(),
        test_size=resolved_options.validation_size,
        stratify=frame[target_column],
        random_state=resolved_options.random_state,
    )

    train_frame = frame.loc[train_index].copy()
    valid_frame = frame.loc[valid_index].copy()

    feature_groups = split_feature_types(
        train_frame,
        target_column=target_column,
        id_column=id_column,
    )
    preprocessor = build_basic_preprocessor(feature_groups, options=resolved_options)

    X_train_frame = train_frame.drop(columns=[target_column, id_column], errors="ignore")
    X_valid_frame = valid_frame.drop(columns=[target_column, id_column], errors="ignore")

    X_train = sparse.csr_matrix(preprocessor.fit_transform(X_train_frame))
    X_valid = sparse.csr_matrix(preprocessor.transform(X_valid_frame))

    feature_names = preprocessor.get_feature_names_out().tolist()

    train_ids = None
    valid_ids = None
    if id_column is not None and id_column in frame.columns:
        train_ids = train_frame[id_column].reset_index(drop=True)
        valid_ids = valid_frame[id_column].reset_index(drop=True)

    return PreparedPreprocessingArtifacts(
        options=resolved_options,
        feature_groups=feature_groups,
        preprocessor=preprocessor,
        X_train=X_train,
        X_valid=X_valid,
        y_train=train_frame[target_column].reset_index(drop=True),
        y_valid=valid_frame[target_column].reset_index(drop=True),
        train_ids=train_ids,
        valid_ids=valid_ids,
        feature_names=feature_names,
        target_column=target_column,
        id_column=id_column,
    )


def _build_meta_frame(
    ids: pd.Series | None,
    targets: pd.Series,
    id_column: str | None,
    target_column: str,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {target_column: targets.reset_index(drop=True)}

    if ids is not None and id_column is not None:
        rows = {
            id_column: ids.reset_index(drop=True),
            target_column: targets.reset_index(drop=True),
        }

    return pd.DataFrame(rows)


def _matrix_density(matrix: sparse.csr_matrix) -> float:
    total_cells = matrix.shape[0] * matrix.shape[1]
    if total_cells == 0:
        return 0.0
    return float(matrix.nnz / total_cells)


def save_preprocessing_artifacts(
    artifacts: PreparedPreprocessingArtifacts,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Persist Phase 2 preprocessing outputs under data/processed/preprocessing."""

    destination = (
        Path(output_dir) if output_dir is not None else get_paths().data_processed / "preprocessing"
    )
    destination.mkdir(parents=True, exist_ok=True)

    file_map = {
        "X_train": destination / "X_train.npz",
        "X_valid": destination / "X_valid.npz",
        "train_meta": destination / "train_meta.csv",
        "valid_meta": destination / "valid_meta.csv",
        "feature_names": destination / "feature_names.csv",
        "manifest": destination / "manifest.json",
    }

    sparse.save_npz(file_map["X_train"], artifacts.X_train)
    sparse.save_npz(file_map["X_valid"], artifacts.X_valid)

    train_meta = _build_meta_frame(
        ids=artifacts.train_ids,
        targets=artifacts.y_train,
        id_column=artifacts.id_column,
        target_column=artifacts.target_column,
    )
    valid_meta = _build_meta_frame(
        ids=artifacts.valid_ids,
        targets=artifacts.y_valid,
        id_column=artifacts.id_column,
        target_column=artifacts.target_column,
    )

    train_meta.to_csv(file_map["train_meta"], index=False)
    valid_meta.to_csv(file_map["valid_meta"], index=False)
    pd.DataFrame({"feature_name": artifacts.feature_names}).to_csv(
        file_map["feature_names"],
        index=False,
    )

    manifest = {
        "target_column": artifacts.target_column,
        "id_column": artifacts.id_column,
        "feature_groups": artifacts.feature_groups,
        "options": asdict(artifacts.options),
        "train_shape": list(artifacts.X_train.shape),
        "valid_shape": list(artifacts.X_valid.shape),
        "train_density": _matrix_density(artifacts.X_train),
        "valid_density": _matrix_density(artifacts.X_valid),
        "output_files": {
            name: str(path.resolve())
            for name, path in file_map.items()
        },
    }
    file_map["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return file_map
