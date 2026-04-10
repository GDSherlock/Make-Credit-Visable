"""Small helpers for report-facing JSON payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_builtin(value: Any) -> Any:
    """Recursively convert numpy / pandas values into JSON-safe builtins."""

    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def build_figure_manifest(figure_paths: dict[str, str | Path]) -> dict[str, str]:
    """Normalize a figure-path mapping into JSON-ready strings."""

    return {
        str(figure_key): str(path)
        for figure_key, path in sorted(figure_paths.items(), key=lambda item: item[0])
    }


def build_report_summary_fields(
    headline: str,
    key_findings: list[str],
    business_implications: list[str],
    figure_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build the standard summary fields added across phase outputs."""

    return {
        "headline": str(headline),
        "key_findings": [str(item) for item in key_findings],
        "business_implications": [str(item) for item in business_implications],
        "figure_manifest": build_figure_manifest(figure_paths or {}),
    }
