"""Smoke tests for repaired figure generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.reporting import generate_repaired_figures


def test_generate_repaired_figures_populates_v2_outputs() -> None:
    result = generate_repaired_figures()

    figures_v2 = ROOT / "reports" / "figures.2"
    assert figures_v2.exists()
    assert result["figures_v2"]
    assert result["audited_figures"]
    assert (figures_v2 / "phase1_main_target_distribution.png").exists()
    assert (figures_v2 / "phase3_logistic_feature_set_roc_comparison.png").exists()
    assert (figures_v2 / "phase4_four_model_roc_comparison.png").exists()
    assert (figures_v2 / "phase5_fairness_metric_gaps.png").exists()
    assert (figures_v2 / "phase6_xgboost_traditional_plus_proxy_calibration_curve.png").exists()
    assert (figures_v2 / "contact_sheets" / "phase6_overview.png").exists()


def test_phase_summaries_include_v2_manifests() -> None:
    summary_paths = [
        ROOT / "data" / "processed" / "eda" / "eda_summary.json",
        ROOT / "data" / "processed" / "preprocessing" / "processing_methods_summary.json",
        ROOT / "data" / "processed" / "modeling_baseline" / "summary.json",
        ROOT / "data" / "processed" / "modeling_advanced" / "summary.json",
        ROOT / "data" / "processed" / "xai_fairness" / "summary.json",
        ROOT / "data" / "processed" / "scorecard_cutoff" / "xgboost_traditional_plus_proxy" / "summary.json",
    ]

    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "figure_manifest_v2" in payload
        assert "figure_quality_status" in payload
        assert "figure_label_audit" in payload
        assert isinstance(payload["figure_label_audit"], dict)
        for repaired_path in payload["figure_manifest_v2"].values():
            assert "/reports/figures.2/" in repaired_path


def test_multiseries_figures_have_visible_identifiers() -> None:
    generate_repaired_figures()

    checks = {
        ROOT / "data" / "processed" / "modeling_baseline" / "summary.json": {
            "phase3_logistic_feature_set_roc_comparison": "direct_label",
            "phase3_logistic_feature_set_pr_comparison": "direct_label",
            "phase3_logistic_feature_set_ks_comparison": "direct_label",
            "phase3_logistic_feature_set_calibration_comparison": "direct_label",
            "phase3_logistic_feature_set_gain_comparison": "direct_label",
            "phase3_logistic_feature_set_lift_comparison": "direct_label",
        },
        ROOT / "data" / "processed" / "modeling_advanced" / "summary.json": {
            "phase4_four_model_roc_comparison": "direct_label",
            "phase4_four_model_pr_comparison": "direct_label",
            "phase4_four_model_ks_comparison": "direct_label",
            "phase4_four_model_calibration_comparison": "direct_label",
            "phase4_four_model_gain_comparison": "direct_label",
            "phase4_four_model_lift_comparison": "direct_label",
        },
        ROOT / "data" / "processed" / "xai_fairness" / "summary.json": {
            "phase5_fairness_metric_gaps": "legend_inside",
        },
        ROOT / "data" / "processed" / "scorecard_cutoff" / "xgboost_traditional_plus_proxy" / "summary.json": {
            "phase6_xgboost_traditional_plus_proxy_calibration_curve": "legend_inside",
            "phase6_xgboost_traditional_plus_proxy_decile_reliability": "legend_inside",
            "phase6_xgboost_traditional_plus_proxy_final_policy_cutoff_curve": "legend_inside",
            "phase6_xgboost_traditional_plus_proxy_age_band_final_policy": "legend_inside",
            "phase6_xgboost_traditional_plus_proxy_family_status_final_policy": "legend_inside",
            "phase6_xgboost_traditional_plus_proxy_region_rating_final_policy": "legend_inside",
        },
    }

    for summary_path, expected_modes in checks.items():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = payload["figure_label_audit"]
        for figure_key, expected_mode in expected_modes.items():
            assert figure_key in payload["figure_manifest_v2"]
            assert audit[figure_key]["identifier_present"] is True
            assert audit[figure_key]["identifier_mode"] == expected_mode


def test_phase4_comparison_figures_record_overlap_flags() -> None:
    generate_repaired_figures()

    payload = json.loads((ROOT / "data" / "processed" / "modeling_advanced" / "summary.json").read_text(encoding="utf-8"))
    audit = payload["figure_label_audit"]
    phase4_keys = [
        "phase4_four_model_roc_comparison",
        "phase4_four_model_pr_comparison",
        "phase4_four_model_ks_comparison",
        "phase4_four_model_calibration_comparison",
        "phase4_four_model_gain_comparison",
        "phase4_four_model_lift_comparison",
    ]

    for figure_key in phase4_keys:
        assert audit[figure_key]["identifier_present"] is True
        assert audit[figure_key]["identifier_mode"] == "direct_label"
        assert isinstance(audit[figure_key]["overlap_warning"], bool)
