"""Feature preparation and diagnostic helpers."""

from credit_visable.features.feature_sets import (
    FEATURE_SET_NAMES,
    FEATURE_SET_TRADITIONAL_CORE,
    FEATURE_SET_TRADITIONAL_PLUS_PROXY,
    build_feature_set_manifest,
    is_proxy_feature,
    list_supported_feature_sets,
    resolve_feature_set_columns,
    select_feature_set_frame,
    validate_feature_set_name,
)
from credit_visable.features.iv_woe import (
    compute_iv_summary,
    compute_woe_detail,
    compute_woe_details,
    fit_woe_placeholder,
)
from credit_visable.features.preprocess import (
    PreparedPreprocessingArtifacts,
    PreprocessingOptions,
    build_basic_preprocessor,
    build_feature_catalog,
    build_preprocessing_decision_manifest,
    prepare_preprocessing_artifacts,
    save_preprocessing_artifacts,
    split_feature_types,
)

__all__ = [
    "FEATURE_SET_NAMES",
    "FEATURE_SET_TRADITIONAL_CORE",
    "FEATURE_SET_TRADITIONAL_PLUS_PROXY",
    "PreparedPreprocessingArtifacts",
    "PreprocessingOptions",
    "build_basic_preprocessor",
    "build_feature_catalog",
    "build_feature_set_manifest",
    "compute_iv_summary",
    "compute_woe_detail",
    "compute_woe_details",
    "build_preprocessing_decision_manifest",
    "fit_woe_placeholder",
    "is_proxy_feature",
    "list_supported_feature_sets",
    "prepare_preprocessing_artifacts",
    "resolve_feature_set_columns",
    "save_preprocessing_artifacts",
    "select_feature_set_frame",
    "split_feature_types",
    "validate_feature_set_name",
]
