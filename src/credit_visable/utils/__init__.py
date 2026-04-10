"""Shared utility helpers."""

from credit_visable.utils.paths import ProjectPaths, get_paths, get_project_root
from credit_visable.utils.plotting import (
    REPORT_COLOR_PALETTE,
    REPORT_LINESTYLES,
    REPORT_SANS_SERIF_STACK,
    add_conclusion_annotation,
    apply_report_style,
    format_percent_axis,
)
from credit_visable.utils.reporting import (
    build_figure_manifest,
    build_report_summary_fields,
    to_builtin,
)

__all__ = [
    "ProjectPaths",
    "REPORT_COLOR_PALETTE",
    "REPORT_LINESTYLES",
    "REPORT_SANS_SERIF_STACK",
    "add_conclusion_annotation",
    "apply_report_style",
    "build_figure_manifest",
    "build_report_summary_fields",
    "format_percent_axis",
    "get_paths",
    "get_project_root",
    "to_builtin",
]
