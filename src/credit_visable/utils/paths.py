"""Centralized project path handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def get_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root by walking parent directories."""

    candidate = Path(start).resolve() if start is not None else Path(__file__).resolve()
    search_roots = [candidate, *candidate.parents]

    for path in search_roots:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            return path

    raise FileNotFoundError("Could not determine the project root from the current path.")


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved directories used across modules and notebooks."""

    root: Path
    src: Path
    configs: Path
    data: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    notebooks: Path
    reports: Path
    reports_figures: Path
    reports_figures_v2: Path
    tests: Path


def get_paths() -> ProjectPaths:
    """Return commonly used project directories."""

    root = get_project_root()
    data_root = root / "data"
    reports_root = root / "reports"

    return ProjectPaths(
        root=root,
        src=root / "src",
        configs=root / "configs",
        data=data_root,
        data_raw=data_root / "raw",
        data_interim=data_root / "interim",
        data_processed=data_root / "processed",
        notebooks=root / "notebooks",
        reports=reports_root,
        reports_figures=reports_root / "figures",
        reports_figures_v2=reports_root / "figures.2",
        tests=root / "tests",
    )


def resolve_report_figure_dir(
    paths: ProjectPaths | None = None,
    variant: str | None = None,
) -> Path:
    """Resolve the requested report figure directory.

    The figure-repair workflow defaults to ``reports/figures.2`` while still
    supporting the legacy ``reports/figures`` directory for fallback runs.
    """

    resolved_paths = paths or get_paths()
    resolved_variant = (
        variant
        or os.environ.get("CREDIT_VISABLE_FIGURE_VARIANT")
        or "v2"
    ).strip().lower()

    if resolved_variant in {"legacy", "original", "v1", "figures"}:
        return resolved_paths.reports_figures
    if resolved_variant in {"v2", "2", "figures.2"}:
        return resolved_paths.reports_figures_v2

    raise ValueError(f"Unsupported figure directory variant: {resolved_variant}")
