"""Shared plotting helpers for report-facing notebooks."""

from __future__ import annotations

import textwrap
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


def wrap_tick_labels(
    ax: Axes,
    axis: str = "y",
    width: int = 18,
) -> Axes:
    """Wrap long tick labels to keep report figures readable."""

    wrapper = lambda label: "\n".join(textwrap.wrap(str(label), width=width)) or str(label)

    if axis in {"x", "both"}:
        ax.set_xticklabels([wrapper(label.get_text()) for label in ax.get_xticklabels()])
    if axis in {"y", "both"}:
        ax.set_yticklabels([wrapper(label.get_text()) for label in ax.get_yticklabels()])
    return ax


def move_legend_outside(
    ax: Axes,
    location: str = "center left",
    anchor: tuple[float, float] = (1.02, 0.5),
) -> Axes:
    """Move a legend outside the plotting area when one exists."""

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
        ax.legend(loc=location, bbox_to_anchor=anchor, frameon=True)
    return ax


def place_legend_inside(
    ax: Axes,
    *,
    location: str = "best",
    title: str | None = None,
    ncol: int = 1,
) -> Axes:
    """Place a legend inside the axes so exported figures keep visible labels."""

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    legend = ax.legend(loc=location, title=title, ncol=ncol, frameon=True, fancybox=True)
    if legend is not None:
        legend.get_frame().set_alpha(0.92)
    return ax


def annotate_bar_values(
    ax: Axes,
    *,
    orientation: str = "vertical",
    value_format: str = "{:.1f}",
    padding: float = 0.01,
    max_bars: int = 20,
) -> Axes:
    """Annotate bar charts with formatted values."""

    patches = list(ax.patches)
    if len(patches) > max_bars:
        return ax

    for patch in patches:
        if orientation == "horizontal":
            width = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2.0
            ax.text(
                width + padding,
                y,
                value_format.format(width),
                va="center",
                ha="left",
                fontsize=8,
            )
        else:
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2.0
            ax.text(
                x,
                height + padding,
                value_format.format(height),
                va="bottom",
                ha="center",
                fontsize=8,
            )
    return ax
