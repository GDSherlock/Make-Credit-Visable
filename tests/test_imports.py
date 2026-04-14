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
    from credit_visable.config import ScorecardSettings, load_scorecard_settings
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
        ApplicationFeatureEngineeringOptions,
        FEATURE_SET_NAMES,
        FeatureReviewOptions,
        GovernedPreprocessingArtifacts,
        GovernedSplitOptions,
        PreparedPreprocessingArtifacts,
        PreprocessingOptions,
        build_application_feature_summary,
        build_basic_preprocessor,
        build_feature_catalog,
        build_feature_set_manifest,
        build_preprocessing_decision_manifest,
        compute_iv_summary,
        compute_woe_detail,
        engineer_application_features,
        is_protected_feature,
        is_proxy_feature,
        is_training_restricted_feature,
        prepare_governed_preprocessing_artifacts,
        prepare_preprocessing_artifacts,
        prune_training_features,
        resolve_feature_set_columns,
        save_governed_preprocessing_artifacts,
        save_preprocessing_artifacts,
        select_feature_set_frame,
    )
    from credit_visable.governance import (
        build_group_fairness_metric_summary,
        build_group_fairness_summary,
        build_grouped_operational_summary,
        build_monitoring_baseline,
        collapse_rare_categories,
        derive_age_band_from_days_birth,
        fairness_report_placeholder,
    )
    from credit_visable.modeling import (
        GovernedTreeTrainingOptions,
        build_binary_diagnostic_curves,
        evaluate_binary_classifier,
        get_tree_backend_availability,
        train_governed_tree_model,
        train_logistic_baseline,
        train_tree_model,
        train_tree_model_placeholder,
    )
    from credit_visable.reporting import (
        generate_phase5_analysis_docx,
        generate_phase6_analysis_docx,
        generate_repaired_figures,
    )
    from credit_visable.scoring import (
        apply_cutoff_policy,
        apply_platt_calibrator,
        assign_frozen_risk_bands,
        assign_risk_band_from_pd,
        assign_risk_band_from_score,
        build_calibration_table,
        build_profit_assumption_config,
        build_risk_band_table,
        build_score_cutoff_grid,
        build_scorecard_metadata,
        build_scorecard_placeholder,
        build_unit_economics_frame,
        build_unit_economics_summary,
        compute_calibration_monitoring_metrics,
        compute_expected_value_frame,
        compute_population_stability_index,
        compute_threshold_profit_curve,
        evaluate_cutoff_curve,
        fit_platt_calibrator,
        freeze_risk_band_thresholds,
        pd_to_score,
        run_governed_application_pipeline,
        run_cutoff_sensitivity_analysis,
        score_to_pd,
        select_final_scenario_cutoff,
        select_optimal_cutoff,
        select_optimal_profit_threshold,
    )
    from credit_visable.utils import (
        REPORT_COLOR_PALETTE,
        REPORT_SANS_SERIF_STACK,
        add_conclusion_annotation,
        annotate_bar_values,
        apply_report_style,
        build_figure_quality_fields,
        build_run_manifest,
        build_report_summary_fields,
        compute_frame_fingerprint,
        compute_series_hash,
        format_percent_axis,
        get_paths,
        move_legend_outside,
        place_legend_inside,
        resolve_git_commit,
        resolve_report_figure_dir,
        to_builtin,
        wrap_tick_labels,
    )

    assert credit_visable.__version__ == "0.1.0"
    assert ScorecardSettings.__name__ == "ScorecardSettings"
    assert ApplicationFeatureEngineeringOptions.__name__ == "ApplicationFeatureEngineeringOptions"
    assert FeatureReviewOptions.__name__ == "FeatureReviewOptions"
    assert GovernedSplitOptions.__name__ == "GovernedSplitOptions"
    assert GovernedPreprocessingArtifacts.__name__ == "GovernedPreprocessingArtifacts"
    assert callable(load_table)
    assert callable(summarize_table_availability)
    assert callable(load_scorecard_settings)
    assert PreprocessingOptions.__name__ == "PreprocessingOptions"
    assert PreparedPreprocessingArtifacts.__name__ == "PreparedPreprocessingArtifacts"
    assert FEATURE_SET_NAMES == ("traditional_core", "traditional_plus_proxy")
    assert callable(build_application_feature_summary)
    assert callable(build_basic_preprocessor)
    assert callable(build_feature_catalog)
    assert callable(build_feature_set_manifest)
    assert callable(build_preprocessing_decision_manifest)
    assert callable(compute_iv_summary)
    assert callable(compute_woe_detail)
    assert callable(engineer_application_features)
    assert callable(is_protected_feature)
    assert callable(is_proxy_feature)
    assert callable(is_training_restricted_feature)
    assert callable(prepare_governed_preprocessing_artifacts)
    assert callable(prepare_preprocessing_artifacts)
    assert callable(prune_training_features)
    assert callable(resolve_feature_set_columns)
    assert callable(save_governed_preprocessing_artifacts)
    assert callable(save_preprocessing_artifacts)
    assert callable(select_feature_set_frame)
    assert GovernedTreeTrainingOptions.__name__ == "GovernedTreeTrainingOptions"
    assert callable(train_logistic_baseline)
    assert callable(get_tree_backend_availability)
    assert callable(train_governed_tree_model)
    assert callable(train_tree_model)
    assert callable(train_tree_model_placeholder)
    assert callable(generate_phase5_analysis_docx)
    assert callable(generate_phase6_analysis_docx)
    assert callable(generate_repaired_figures)
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
    assert callable(build_monitoring_baseline)
    assert callable(collapse_rare_categories)
    assert callable(derive_age_band_from_days_birth)
    assert callable(build_profit_assumption_config)
    assert callable(build_scorecard_metadata)
    assert callable(build_scorecard_placeholder)
    assert callable(build_score_cutoff_grid)
    assert callable(build_risk_band_table)
    assert callable(pd_to_score)
    assert callable(score_to_pd)
    assert callable(assign_frozen_risk_bands)
    assert callable(assign_risk_band_from_pd)
    assert callable(assign_risk_band_from_score)
    assert callable(apply_cutoff_policy)
    assert callable(fit_platt_calibrator)
    assert callable(apply_platt_calibrator)
    assert callable(build_calibration_table)
    assert callable(compute_calibration_monitoring_metrics)
    assert callable(compute_population_stability_index)
    assert callable(build_unit_economics_frame)
    assert callable(build_unit_economics_summary)
    assert callable(compute_expected_value_frame)
    assert callable(evaluate_cutoff_curve)
    assert callable(freeze_risk_band_thresholds)
    assert callable(select_optimal_cutoff)
    assert callable(run_governed_application_pipeline)
    assert callable(run_cutoff_sensitivity_analysis)
    assert callable(select_final_scenario_cutoff)
    assert callable(compute_threshold_profit_curve)
    assert callable(select_optimal_profit_threshold)
    assert callable(add_conclusion_annotation)
    assert callable(annotate_bar_values)
    assert callable(apply_report_style)
    assert callable(build_figure_quality_fields)
    assert callable(build_run_manifest)
    assert callable(build_report_summary_fields)
    assert callable(compute_frame_fingerprint)
    assert callable(compute_series_hash)
    assert callable(format_percent_axis)
    assert REPORT_SANS_SERIF_STACK[0] == "DejaVu Sans"
    assert REPORT_COLOR_PALETTE["good"] == "#4C78A8"
    assert callable(get_paths)
    assert callable(move_legend_outside)
    assert callable(place_legend_inside)
    assert callable(resolve_git_commit)
    assert callable(resolve_report_figure_dir)
    assert callable(to_builtin)
    assert callable(wrap_tick_labels)
