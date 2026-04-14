"""Tests for the Phase 2 preprocessing workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.features.preprocess import (
    GovernedSplitOptions,
    PreprocessingOptions,
    build_basic_preprocessor,
    prepare_governed_preprocessing_artifacts,
    prepare_preprocessing_artifacts,
    save_governed_preprocessing_artifacts,
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
            "annuity": [8_000 + (idx * 10) for idx in range(row_count)],
            "goods_price": [95_000 + (idx * 200) for idx in range(row_count)],
            "days_employed": [365243 if idx % 10 == 0 else -(100 + idx) for idx in range(row_count)],
            "days_birth": [-(9000 + idx * 5) for idx in range(row_count)],
            "gender": ["F", "M"] * (row_count // 2),
            "family_status": ["Married", "Single"] * (row_count // 2),
            "owns_car": ([True, False, False, True, False] * 10)[:row_count],
            "ext_source_1": [np.nan if idx % 4 == 0 else 0.2 + (idx * 0.01) for idx in range(row_count)],
        }
    )


def test_split_feature_types_excludes_id_and_target() -> None:
    feature_groups = split_feature_types(
        _sample_frame(),
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    assert feature_groups["numeric"][:2] == ["income", "credit"]
    assert "gender" in feature_groups["categorical"]
    assert "owns_car" in feature_groups["categorical"]


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


def test_governed_preprocessing_artifacts_create_isolated_splits() -> None:
    frame = _sample_frame().rename(
        columns={
            "income": "AMT_INCOME_TOTAL",
            "credit": "AMT_CREDIT",
            "annuity": "AMT_ANNUITY",
            "goods_price": "AMT_GOODS_PRICE",
            "days_employed": "DAYS_EMPLOYED",
            "days_birth": "DAYS_BIRTH",
            "gender": "CODE_GENDER",
            "family_status": "NAME_FAMILY_STATUS",
            "ext_source_1": "EXT_SOURCE_1",
        }
    )
    artifacts = prepare_governed_preprocessing_artifacts(
        frame,
        feature_set_name="traditional_plus_proxy",
        target_column="TARGET",
        id_column="SK_ID_CURR",
        split_options=GovernedSplitOptions(calibration_size=0.2, test_size=0.2, random_state=42),
    )

    assert artifacts.X_dev.shape[0] == 30
    assert artifacts.X_calibration.shape[0] == 10
    assert artifacts.X_test.shape[0] == 10
    assert set(artifacts.dev_ids.tolist()).isdisjoint(artifacts.calibration_ids.tolist())
    assert set(artifacts.dev_ids.tolist()).isdisjoint(artifacts.test_ids.tolist())
    assert "CODE_GENDER" not in artifacts.selected_feature_columns
    assert "NAME_FAMILY_STATUS" not in artifacts.selected_feature_columns
    assert any(column_name.startswith("MISSING_FLAG_") for column_name in artifacts.selected_feature_columns)
    assert "DAYS_EMPLOYED" not in artifacts.selected_feature_columns


def test_save_governed_preprocessing_artifacts_writes_split_outputs(tmp_path: Path) -> None:
    frame = _sample_frame().rename(
        columns={
            "income": "AMT_INCOME_TOTAL",
            "credit": "AMT_CREDIT",
            "annuity": "AMT_ANNUITY",
            "goods_price": "AMT_GOODS_PRICE",
            "days_employed": "DAYS_EMPLOYED",
            "days_birth": "DAYS_BIRTH",
            "gender": "CODE_GENDER",
            "family_status": "NAME_FAMILY_STATUS",
            "ext_source_1": "EXT_SOURCE_1",
        }
    )
    artifacts = prepare_governed_preprocessing_artifacts(
        frame,
        feature_set_name="traditional_plus_proxy",
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    saved = save_governed_preprocessing_artifacts(artifacts, output_dir=tmp_path)

    assert saved["X_dev"].exists()
    assert saved["X_calibration"].exists()
    assert saved["X_test"].exists()
    assert saved["split_manifest"].exists()
    manifest = json.loads(saved["split_manifest"].read_text(encoding="utf-8"))
    assert manifest["split_strategy"] == "dev_calibration_test"
    assert manifest["selected_feature_columns"] == artifacts.selected_feature_columns
