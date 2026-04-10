"""Shared plotting helpers for report-facing notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


REPORT_SANS_SERIF_STACK = [
    "DejaVu Sans",
    "Arial",
    "Liberation Sans",
    "sans-serif",
]
REPORT_COLOR_PALETTE = {
    "good": "#4C78A8",
    "bad": "#E45756",
    "neutral": "#9D9D9D",
    "accent": "#72B7B2",
    "highlight": "#F58518",
}
REPORT_LINESTYLES = {
    "best": "-",
    "comparison": "-.",
    "baseline": "--",
}


def apply_report_style(**overrides: Any) -> None:
    """Apply a deterministic matplotlib style for exported report figures."""

    rc_updates: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": REPORT_SANS_SERIF_STACK,
        "axes.unicode_minus": False,
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
    rc_updates.update(overrides)
    plt.rcParams.update(rc_updates)


def format_percent_axis(
    ax: Axes,
    axis: str = "y",
    decimals: int = 0,
) -> Axes:
    """Apply percentage formatting to one axis."""

    formatter = PercentFormatter(xmax=1.0, decimals=decimals)
    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(formatter)
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(formatter)
    return ax


def add_conclusion_annotation(
    ax: Axes,
    text: str,
    x: float = 0.02,
    y: float = 0.98,
    color: str | None = None,
) -> Axes:
    """Add a concise report-style conclusion annotation inside the axes."""

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=color or REPORT_COLOR_PALETTE["neutral"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": color or REPORT_COLOR_PALETTE["neutral"],
            "alpha": 0.9,
        },
    )
    return ax
