"""Smoke tests for config loading and repo-relative paths."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.config import load_settings
from credit_visable.utils.paths import get_paths, get_project_root


def test_project_root_resolution() -> None:
    root = get_project_root()
    assert root == ROOT
    assert (root / "pyproject.toml").exists()


def test_settings_loads_from_yaml() -> None:
    settings = load_settings()
    assert settings.project_name == "credit visable"
    assert settings.target_column == "TARGET"
    assert "application_train" in settings.expected_tables


def test_paths_expose_expected_directories() -> None:
    paths = get_paths()
    assert paths.data_raw == ROOT / "data" / "raw"
    assert paths.notebooks == ROOT / "notebooks"
    assert paths.reports_figures == ROOT / "reports" / "figures"
