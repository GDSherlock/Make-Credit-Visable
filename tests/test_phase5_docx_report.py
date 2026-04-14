"""Tests for the Phase 5 DOCX report generator."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_generate_phase5_docx_report(tmp_path: Path) -> None:
    _ensure_src_on_path()

    from docx import Document

    from credit_visable.reporting.phase5_docx_report import (
        APPENDIX_FIGURES,
        MAIN_FIGURES,
        generate_phase5_analysis_docx,
    )
    from credit_visable.utils import get_paths

    paths = get_paths()

    required_sources = [
        paths.data_processed / "xai_fairness" / "summary.json",
        paths.data_processed / "xai_fairness" / "candidate_model_selection.json",
        paths.data_processed / "xai_fairness" / "group_fairness_summary.csv",
        paths.data_processed / "xai_fairness" / "fairness_metric_summary.csv",
        paths.data_processed / "xai_fairness" / "proxy_uplift_summary.csv",
        paths.data_processed / "xai_fairness" / "top_shap_interactions.csv",
        paths.data_processed / "xai_fairness" / "candidate_partial_dependence.csv",
        paths.data_processed
        / "xai_fairness"
        / "traditional_plus_proxy"
        / "global_feature_contributions.csv",
        paths.data_processed
        / "xai_fairness"
        / "traditional_plus_proxy"
        / "local_case_explanations.csv",
        paths.data_processed
        / "xai_fairness"
        / "traditional_plus_proxy"
        / "lime_local_case_explanations.csv",
    ]
    required_sources.extend(paths.reports_figures_v2 / spec.filename for spec in MAIN_FIGURES + APPENDIX_FIGURES)

    missing = [str(path) for path in required_sources if not path.exists()]
    assert not missing, f"Required Phase 5 report sources are missing: {missing}"

    output_path = tmp_path / "phase5_report.docx"
    result = generate_phase5_analysis_docx(output_path=output_path)

    assert result == output_path
    assert result.exists()
    assert result.stat().st_size > 0

    document = Document(result)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    for heading in [
        "Executive Summary",
        "Candidate Model and Comparison Bridge",
        "Global Explainability and Proxy Uplift",
        "Grouped Fairness and Governance Review",
        "Interaction and Response Diagnostics",
        "Conclusion",
        "Limitations",
        "Appendix: Local Explanation Example",
    ]:
        assert heading in text

    for expected_snippet in [
        "xgboost_traditional_plus_proxy",
        "xgboost_traditional_core",
        "ext_source",
        "age_band",
        "0.7597",
        "0.6888",
        "0.901",
        "64.57 pp",
        "93.15%",
        "45.07%",
        "0.1346",
        "94.99%",
    ]:
        assert expected_snippet in text

    assert len(document.inline_shapes) >= 12
