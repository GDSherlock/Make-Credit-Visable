"""Shared plotting helpers for report-facing notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


REPORT_SANS_SERIF_STACK = [
    "DejaVu Sans",
    "Arial",
    "Liberation Sans",
    "sans-serif",
]


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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
    rc_updates.update(overrides)
    plt.rcParams.update(rc_updates)
