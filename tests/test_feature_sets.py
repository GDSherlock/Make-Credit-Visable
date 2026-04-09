"""Tests for feature-set selection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.features.feature_sets import (
    FEATURE_SET_TRADITIONAL_CORE,
    FEATURE_SET_TRADITIONAL_PLUS_PROXY,
    build_feature_set_manifest,
    is_proxy_feature,
    resolve_feature_set_columns,
    select_feature_set_frame,
)


def _sample_application_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "TARGET": [0, 1],
            "AMT_INCOME_TOTAL": [120000.0, 80000.0],
            "AMT_CREDIT": [300000.0, 180000.0],
            "EXT_SOURCE_1": [0.5, 0.2],
            "OBS_30_CNT_SOCIAL_CIRCLE": [0.0, 2.0],
            "FLAG_PHONE": [1, 0],
            "REGION_POPULATION_RELATIVE": [0.01, 0.02],
            "REG_CITY_NOT_WORK_CITY": [0, 1],
            "ORGANIZATION_TYPE": ["Business", "XNA"],
            "FLAG_DOCUMENT_3": [1, 0],
            "NAME_INCOME_TYPE": ["Working", "Pensioner"],
        }
    )


def test_is_proxy_feature_covers_explicit_proxy_families() -> None:
    assert is_proxy_feature("EXT_SOURCE_2") is True
    assert is_proxy_feature("OBS_60_CNT_SOCIAL_CIRCLE") is True
    assert is_proxy_feature("FLAG_PHONE") is True
    assert is_proxy_feature("REG_REGION_NOT_LIVE_REGION") is True
    assert is_proxy_feature("LIVE_CITY_NOT_WORK_CITY") is True
    assert is_proxy_feature("ORGANIZATION_TYPE") is True
    assert is_proxy_feature("FLAG_DOCUMENT_21") is True
    assert is_proxy_feature("AMT_INCOME_TOTAL") is False


def test_resolve_feature_set_columns_excludes_proxy_fields_from_traditional_core() -> None:
    frame = _sample_application_frame()

    traditional_columns = resolve_feature_set_columns(
        frame.columns.tolist(),
        feature_set_name=FEATURE_SET_TRADITIONAL_CORE,
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )
    proxy_columns = [
        column_name
        for column_name in resolve_feature_set_columns(
            frame.columns.tolist(),
            feature_set_name=FEATURE_SET_TRADITIONAL_PLUS_PROXY,
            target_column="TARGET",
            id_column="SK_ID_CURR",
        )
        if column_name not in traditional_columns
    ]

    assert traditional_columns == [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "NAME_INCOME_TYPE",
    ]
    assert proxy_columns == [
        "EXT_SOURCE_1",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "FLAG_PHONE",
        "REGION_POPULATION_RELATIVE",
        "REG_CITY_NOT_WORK_CITY",
        "ORGANIZATION_TYPE",
        "FLAG_DOCUMENT_3",
    ]


def test_select_feature_set_frame_preserves_id_target_and_selected_feature_order() -> None:
    frame = _sample_application_frame()

    selected = select_feature_set_frame(
        frame,
        feature_set_name=FEATURE_SET_TRADITIONAL_CORE,
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    assert selected.columns.tolist() == [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "NAME_INCOME_TYPE",
    ]


def test_build_feature_set_manifest_reports_disjoint_traditional_and_proxy_columns() -> None:
    frame = _sample_application_frame()

    manifest = build_feature_set_manifest(
        frame.columns.tolist(),
        feature_set_name=FEATURE_SET_TRADITIONAL_CORE,
        target_column="TARGET",
        id_column="SK_ID_CURR",
    )

    assert manifest["feature_set_name"] == FEATURE_SET_TRADITIONAL_CORE
    assert manifest["selected_feature_columns"] == [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "NAME_INCOME_TYPE",
    ]
    assert manifest["traditional_feature_count"] == 3
    assert manifest["proxy_feature_count"] == 7
    assert set(manifest["traditional_feature_columns"]).isdisjoint(
        manifest["proxy_feature_columns"]
    )
    assert "Proxy refers only to internal proxy variables" in manifest["proxy_definition"]["note"]
