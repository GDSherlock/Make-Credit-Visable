"""Smoke tests for package imports."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_package_imports() -> None:
    _ensure_src_on_path()

    import credit_visable
    from credit_visable.data import load_table, summarize_table_availability
    from credit_visable.explainability import (
        build_transformed_feature_mapping,
        build_shap_plot_explanation,
        compute_lime_local_explanations,
        compute_permutation_importance_summary,
        compute_shap_contribution_summary,
        compute_shap_local_explanations,
        compute_xgboost_contribution_summary,
        compute_xgboost_local_explanations,
        get_explainability_runtime_status,
        run_shap_placeholder,
        select_local_explanation_rows,
        summarize_contribution_values,
    )
    from credit_visable.features import (
        FEATURE_SET_NAMES,
        PreparedPreprocessingArtifacts,
        PreprocessingOptions,
        build_basic_preprocessor,
        build_feature_catalog,
        build_feature_set_manifest,
        build_preprocessing_decision_manifest,
        compute_iv_summary,
        compute_woe_detail,
        is_proxy_feature,
        prepare_preprocessing_artifacts,
        resolve_feature_set_columns,
        save_preprocessing_artifacts,
        select_feature_set_frame,
    )
    from credit_visable.governance import (
        build_group_fairness_metric_summary,
        build_group_fairness_summary,
        build_grouped_operational_summary,
        collapse_rare_categories,
        derive_age_band_from_days_birth,
        fairness_report_placeholder,
    )
    from credit_visable.modeling import (
        build_binary_diagnostic_curves,
        evaluate_binary_classifier,
        get_tree_backend_availability,
        train_logistic_baseline,
        train_tree_model,
        train_tree_model_placeholder,
    )
    from credit_visable.scoring import (
        build_profit_assumption_config,
        build_scorecard_placeholder,
        compute_threshold_profit_curve,
        select_optimal_profit_threshold,
    )
    from credit_visable.utils import (
        REPORT_COLOR_PALETTE,
        REPORT_SANS_SERIF_STACK,
        add_conclusion_annotation,
        apply_report_style,
        build_report_summary_fields,
        format_percent_axis,
        get_paths,
        to_builtin,
    )

    assert credit_visable.__version__ == "0.1.0"
    assert callable(load_table)
    assert callable(summarize_table_availability)
    assert PreprocessingOptions.__name__ == "PreprocessingOptions"
    assert PreparedPreprocessingArtifacts.__name__ == "PreparedPreprocessingArtifacts"
    assert FEATURE_SET_NAMES == ("traditional_core", "traditional_plus_proxy")
    assert callable(build_basic_preprocessor)
    assert callable(build_feature_catalog)
    assert callable(build_feature_set_manifest)
    assert callable(build_preprocessing_decision_manifest)
    assert callable(compute_iv_summary)
    assert callable(compute_woe_detail)
    assert callable(is_proxy_feature)
    assert callable(prepare_preprocessing_artifacts)
    assert callable(resolve_feature_set_columns)
    assert callable(save_preprocessing_artifacts)
    assert callable(select_feature_set_frame)
    assert callable(train_logistic_baseline)
    assert callable(get_tree_backend_availability)
    assert callable(train_tree_model)
    assert callable(train_tree_model_placeholder)
    assert callable(build_binary_diagnostic_curves)
    assert callable(evaluate_binary_classifier)
    assert callable(run_shap_placeholder)
    assert callable(build_transformed_feature_mapping)
    assert callable(build_shap_plot_explanation)
    assert callable(compute_lime_local_explanations)
    assert callable(compute_permutation_importance_summary)
    assert callable(compute_shap_contribution_summary)
    assert callable(compute_shap_local_explanations)
    assert callable(compute_xgboost_contribution_summary)
    assert callable(compute_xgboost_local_explanations)
    assert callable(get_explainability_runtime_status)
    assert callable(select_local_explanation_rows)
    assert callable(summarize_contribution_values)
    assert callable(fairness_report_placeholder)
    assert callable(build_group_fairness_metric_summary)
    assert callable(build_group_fairness_summary)
    assert callable(build_grouped_operational_summary)
    assert callable(collapse_rare_categories)
    assert callable(derive_age_band_from_days_birth)
    assert callable(build_profit_assumption_config)
    assert callable(build_scorecard_placeholder)
    assert callable(compute_threshold_profit_curve)
    assert callable(select_optimal_profit_threshold)
    assert callable(add_conclusion_annotation)
    assert callable(apply_report_style)
    assert callable(build_report_summary_fields)
    assert callable(format_percent_axis)
    assert REPORT_SANS_SERIF_STACK[0] == "DejaVu Sans"
    assert REPORT_COLOR_PALETTE["good"] == "#4C78A8"
    assert callable(get_paths)
    assert callable(to_builtin)
