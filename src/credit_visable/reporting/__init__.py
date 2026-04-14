"""Report-oriented helpers."""

from credit_visable.reporting.figure_repair import generate_repaired_figures
from credit_visable.reporting.phase1_phase2_docx_report import (
    generate_phase1_phase2_analysis_docx,
)
from credit_visable.reporting.phase5_docx_report import generate_phase5_analysis_docx
from credit_visable.reporting.phase6_docx_report import generate_phase6_analysis_docx

__all__ = [
    "generate_phase1_phase2_analysis_docx",
    "generate_phase5_analysis_docx",
    "generate_phase6_analysis_docx",
    "generate_repaired_figures",
]
