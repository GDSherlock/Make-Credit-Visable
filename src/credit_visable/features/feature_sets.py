"""Feature-set definitions for traditional versus proxy comparisons."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd


FEATURE_SET_TRADITIONAL_CORE = "traditional_core"
FEATURE_SET_TRADITIONAL_PLUS_PROXY = "traditional_plus_proxy"
FEATURE_SET_NAMES = (
    FEATURE_SET_TRADITIONAL_CORE,
    FEATURE_SET_TRADITIONAL_PLUS_PROXY,
)

_PROXY_EXACT_COLUMNS = {
    "FLAG_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_EMP_PHONE",
    "FLAG_EMAIL",
    "DAYS_LAST_PHONE_CHANGE",
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
}
_PROXY_PATTERNS = (
    re.compile(r"^EXT_SOURCE_\d+$"),
    re.compile(r"^(OBS|DEF)_\d+_CNT_SOCIAL_CIRCLE$"),
    re.compile(r"^FLAG_DOCUMENT_\d+$"),
)


def list_supported_feature_sets() -> tuple[str, ...]:
    """Return the supported feature-set names in stable order."""

    return FEATURE_SET_NAMES


def validate_feature_set_name(feature_set_name: str) -> str:
    """Validate that the requested feature set is supported."""

    if feature_set_name not in FEATURE_SET_NAMES:
        supported = ", ".join(FEATURE_SET_NAMES)
        raise ValueError(
            f"Unsupported feature set '{feature_set_name}'. Supported values: {supported}."
        )
    return feature_set_name


def is_proxy_feature(column_name: str) -> bool:
    """Return whether a raw Home Credit application column is treated as proxy data."""

    if column_name in _PROXY_EXACT_COLUMNS:
        return True
    if "REGION" in column_name or "_CITY_" in column_name:
        return True
    return any(pattern.fullmatch(column_name) for pattern in _PROXY_PATTERNS)


def _excluded_columns(
    columns: Sequence[str],
    target_column: str | None = None,
    id_column: str | None = None,
) -> list[str]:
    return [
        column_name
        for column_name in columns
        if column_name in {target_column, id_column}
    ]


def resolve_feature_set_columns(
    columns: Sequence[str],
    feature_set_name: str,
    target_column: str | None = None,
    id_column: str | None = None,
) -> list[str]:
    """Return ordered feature columns for the requested feature set."""

    resolved_feature_set = validate_feature_set_name(feature_set_name)
    excluded = set(_excluded_columns(columns, target_column=target_column, id_column=id_column))
    feature_columns = [column_name for column_name in columns if column_name not in excluded]

    if resolved_feature_set == FEATURE_SET_TRADITIONAL_PLUS_PROXY:
        return feature_columns

    return [column_name for column_name in feature_columns if not is_proxy_feature(column_name)]


def select_feature_set_frame(
    frame: pd.DataFrame,
    feature_set_name: str,
    target_column: str | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy of the frame limited to the selected feature set plus id/target."""

    selected_columns: list[str] = []
    if id_column is not None and id_column in frame.columns:
        selected_columns.append(id_column)
    if target_column is not None and target_column in frame.columns:
        selected_columns.append(target_column)

    selected_columns.extend(
        resolve_feature_set_columns(
            frame.columns.tolist(),
            feature_set_name=feature_set_name,
            target_column=target_column,
            id_column=id_column,
        )
    )
    return frame.loc[:, selected_columns].copy()


def build_feature_set_manifest(
    columns: Sequence[str],
    feature_set_name: str,
    target_column: str | None = None,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable manifest describing the selected feature set."""

    resolved_feature_set = validate_feature_set_name(feature_set_name)
    excluded_columns = _excluded_columns(
        columns,
        target_column=target_column,
        id_column=id_column,
    )
    available_feature_columns = [
        column_name for column_name in columns if column_name not in set(excluded_columns)
    ]
    proxy_feature_columns = [
        column_name for column_name in available_feature_columns if is_proxy_feature(column_name)
    ]
    traditional_feature_columns = [
        column_name for column_name in available_feature_columns if not is_proxy_feature(column_name)
    ]
    selected_feature_columns = resolve_feature_set_columns(
        columns,
        feature_set_name=resolved_feature_set,
        target_column=target_column,
        id_column=id_column,
    )

    return {
        "feature_set_name": resolved_feature_set,
        "target_column": target_column,
        "id_column": id_column,
        "available_feature_count": len(available_feature_columns),
        "selected_feature_count": len(selected_feature_columns),
        "traditional_feature_count": len(traditional_feature_columns),
        "proxy_feature_count": len(proxy_feature_columns),
        "available_feature_columns": available_feature_columns,
        "selected_feature_columns": selected_feature_columns,
        "traditional_feature_columns": traditional_feature_columns,
        "proxy_feature_columns": proxy_feature_columns,
        "excluded_columns": excluded_columns,
        "proxy_definition": {
            "exact_columns": sorted(_PROXY_EXACT_COLUMNS),
            "pattern_rules": [
                "^EXT_SOURCE_\\d+$",
                "^(OBS|DEF)_\\d+_CNT_SOCIAL_CIRCLE$",
                "^FLAG_DOCUMENT_\\d+$",
            ],
            "substring_rules": [
                "contains REGION",
                "contains _CITY_",
            ],
            "note": (
                "Proxy refers only to internal proxy variables already present in "
                "Home Credit application data. It does not refer to external "
                "alternative data sources."
            ),
        },
    }
