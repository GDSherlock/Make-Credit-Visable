"""Centralized project path handling."""

from __future__ import annotations

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
        tests=root / "tests",
    )
