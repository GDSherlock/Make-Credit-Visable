"""Explainability helpers."""

from credit_visable.explainability.shap_analysis import (
    build_transformed_feature_mapping,
    build_shap_plot_explanation,
    compute_lime_local_explanations,
    compute_permutation_importance_summary,
    compute_shap_contribution_summary,
    compute_shap_local_explanations,
    compute_xgboost_contribution_summary,
    compute_xgboost_local_explanations,
    get_explainability_runtime_status,
    resolve_proxy_family,
    run_shap_placeholder,
    select_local_explanation_rows,
    summarize_contribution_values,
)

__all__ = [
    "build_transformed_feature_mapping",
    "build_shap_plot_explanation",
    "compute_lime_local_explanations",
    "compute_permutation_importance_summary",
    "compute_shap_contribution_summary",
    "compute_shap_local_explanations",
    "compute_xgboost_contribution_summary",
    "compute_xgboost_local_explanations",
    "get_explainability_runtime_status",
    "resolve_proxy_family",
    "run_shap_placeholder",
    "select_local_explanation_rows",
    "summarize_contribution_values",
]
