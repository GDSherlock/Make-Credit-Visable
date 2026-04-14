"""Shared utility helpers."""

from credit_visable.utils.paths import (
    ProjectPaths,
    get_paths,
    get_project_root,
    resolve_report_figure_dir,
)
from credit_visable.utils.plotting import (
    REPORT_COLOR_PALETTE,
    REPORT_LINESTYLES,
    REPORT_SANS_SERIF_STACK,
    annotate_bar_values,
    add_conclusion_annotation,
    apply_report_style,
    format_percent_axis,
    move_legend_outside,
    place_legend_inside,
    wrap_tick_labels,
)
from credit_visable.utils.reporting import (
    build_figure_quality_fields,
    build_figure_manifest,
    build_report_summary_fields,
    to_builtin,
)
from credit_visable.utils.reproducibility import (
    build_run_manifest,
    compute_frame_fingerprint,
    compute_series_hash,
    resolve_git_commit,
)

__all__ = [
    "ProjectPaths",
    "REPORT_COLOR_PALETTE",
    "REPORT_LINESTYLES",
    "REPORT_SANS_SERIF_STACK",
    "annotate_bar_values",
    "add_conclusion_annotation",
    "apply_report_style",
    "build_figure_quality_fields",
    "build_figure_manifest",
    "build_report_summary_fields",
    "build_run_manifest",
    "compute_frame_fingerprint",
    "compute_series_hash",
    "format_percent_axis",
    "get_paths",
    "get_project_root",
    "move_legend_outside",
    "place_legend_inside",
    "resolve_git_commit",
    "resolve_report_figure_dir",
    "to_builtin",
    "wrap_tick_labels",
]
