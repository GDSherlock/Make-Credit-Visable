"""Tests for the Phase 6 DOCX report generator."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_generate_phase6_docx_report(tmp_path: Path) -> None:
    _ensure_src_on_path()

    from docx import Document

    from credit_visable.reporting.phase6_docx_report import (
        APPENDIX_FIGURES,
        MAIN_FIGURES,
        DEFAULT_PHASE6_LABEL,
        generate_phase6_analysis_docx,
    )
    from credit_visable.utils import get_paths

    paths = get_paths()

    required_sources = [
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "summary.json",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "final_policy_summary.json",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "score_transform_meta.json",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "risk_band_summary.csv",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "calibration_summary.csv",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "final_policy_group_summary.csv",
        paths.data_processed / "scorecard_cutoff" / DEFAULT_PHASE6_LABEL / "decision_migration_matrix.csv",
        paths.data_processed / "xai_fairness" / "proxy_uplift_summary.csv",
        paths.data_processed / "xai_fairness" / "fairness_metric_summary.csv",
    ]
    required_sources.extend(paths.reports_figures_v2 / spec.filename for spec in MAIN_FIGURES + APPENDIX_FIGURES)

    missing = [str(path) for path in required_sources if not path.exists()]
    assert not missing, f"Required Phase 5/6 report sources are missing: {missing}"

    output_path = tmp_path / "phase6_report.docx"
    result = generate_phase6_analysis_docx(output_path=output_path)

    assert result == output_path
    assert result.exists()
    assert result.stat().st_size > 0

    document = Document(result)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    for heading in [
        "Executive Summary",
        "Phase 5 Governance Bridge",
        "Calibration and Score Translation",
        "Risk Bands and Portfolio Stratification",
        "Final Policy Cutoff Analysis",
        "Group Sensitivity and Migration",
        "Conclusion",
        "Limitations",
    ]:
        assert heading in text

    for expected_snippet in [
        "38.98%",
        "8.07%",
        "0.0676",
        "0.87 pp",
        "505",
        "485",
        "92.05% / 3.93% / 4.02%",
        "6.18%",
        "133,810.65",
        "75.0%",
        "ext_source",
    ]:
        assert expected_snippet in text

    assert len(document.inline_shapes) >= 10
