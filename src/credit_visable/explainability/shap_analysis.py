"""Placeholder hooks for future SHAP-based explainability."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any


def run_shap_placeholder(model: Any, X_sample: Any, max_rows: int = 1000) -> dict[str, Any]:
    """Return metadata for future SHAP analysis without requiring SHAP today."""

    return {
        "package_available": find_spec("shap") is not None,
        "max_rows": max_rows,
        "ready": False,
        "notes": (
            "TODO: add SHAP workflow after the feature matrix and chosen model "
            "interface are stable."
        ),
    }
