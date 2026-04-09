"""Tests for the Phase 2 preprocessing workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.features.preprocess import (
    PreprocessingOptions,
    build_basic_preprocessor,
    prepare_preprocessing_artifacts,
    save_preprocessing_artifacts,
    split_feature_types,
)


def _sample_frame() -> pd.DataFrame:
    row_count = 50
    target = ([0] * 40) + ([1] * 10)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1000, 1000 + row_count),
            "TARGET": target,
            "income": [50_000 + (idx * 100) for idx in range(row_count)],
            "credit": [100_000 + (idx * 250) for idx in range(row_count)],
            "gender": ["F", "M"] * (row_count // 2),
            "owns_car": ([True, False, False, True, False] * 10)[:row_count],
        }
    )


def test_split_feature_types_excludes_id_and_target() -> None:
    feature_groups = split_feature_types(
        _sample_frame(),
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    assert feature_groups == {
        "numeric": ["income", "credit"],
        "categorical": ["gender", "owns_car"],
    }


def test_build_basic_preprocessor_groups_rare_categories_with_default_threshold() -> None:
    frame = pd.DataFrame(
        {
            "numeric_feature": list(range(200)),
            "segment": ["common"] * 199 + ["rare"],
        }
    )
    feature_groups = split_feature_types(frame)

    preprocessor = build_basic_preprocessor(feature_groups)
    preprocessor.fit(frame)
    feature_names = preprocessor.get_feature_names_out().tolist()

    assert "categorical__segment_common" in feature_names
    assert "categorical__segment_infrequent_sklearn" in feature_names
    assert "categorical__segment_rare" not in feature_names


def test_prepare_preprocessing_artifacts_is_repeatable_and_stratified() -> None:
    frame = _sample_frame()
    options = PreprocessingOptions(validation_size=0.2, random_state=42)

    first = prepare_preprocessing_artifacts(
        frame,
        target_column="TARGET",
        id_column="SK_ID_CURR",
        options=options,
    )
    second = prepare_preprocessing_artifacts(
        frame,
        target_column="TARGET",
        id_column="SK_ID_CURR",
        options=options,
    )

    assert first.train_ids is not None
    assert first.valid_ids is not None
    assert first.train_ids.tolist() == second.train_ids.tolist()
    assert first.valid_ids.tolist() == second.valid_ids.tolist()
    assert first.y_train.tolist() == second.y_train.tolist()
    assert first.y_valid.tolist() == second.y_valid.tolist()
    assert first.feature_names == second.feature_names
    assert first.X_train.shape[0] == 40
    assert first.X_valid.shape[0] == 10
    assert first.y_train.mean() == 0.2
    assert first.y_valid.mean() == 0.2


def test_save_preprocessing_artifacts_writes_expected_files(tmp_path: Path) -> None:
    artifacts = prepare_preprocessing_artifacts(
        _sample_frame(),
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    saved = save_preprocessing_artifacts(artifacts, output_dir=tmp_path)

    expected_keys = {
        "X_train",
        "X_valid",
        "train_meta",
        "valid_meta",
        "feature_names",
        "manifest",
    }
    assert set(saved) == expected_keys
    assert all(path.exists() for path in saved.values())

    X_train = sparse.load_npz(saved["X_train"])
    X_valid = sparse.load_npz(saved["X_valid"])
    feature_names = pd.read_csv(saved["feature_names"])
    train_meta = pd.read_csv(saved["train_meta"])

    assert X_train.shape == artifacts.X_train.shape
    assert X_valid.shape == artifacts.X_valid.shape
    assert feature_names["feature_name"].tolist() == artifacts.feature_names
    assert train_meta.columns.tolist() == ["SK_ID_CURR", "TARGET"]

    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    assert manifest["train_shape"] == list(artifacts.X_train.shape)
    assert manifest["valid_shape"] == list(artifacts.X_valid.shape)
    assert manifest["options"]["rare_category_min_frequency"] == 0.01
