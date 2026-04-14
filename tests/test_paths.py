"""Smoke tests for config loading and repo-relative paths."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_visable.config import load_scorecard_settings, load_settings
from credit_visable.utils.paths import get_paths, get_project_root, resolve_report_figure_dir


def test_project_root_resolution() -> None:
    root = get_project_root()
    assert root == ROOT
    assert (root / "pyproject.toml").exists()


def test_settings_loads_from_yaml() -> None:
    settings = load_settings()
    assert settings.project_name == "credit visable"
    assert settings.target_column == "TARGET"
    assert "application_train" in settings.expected_tables
    assert settings.scorecard_config == "scorecard.yaml"


def test_scorecard_settings_load_from_yaml() -> None:
    scorecard_settings = load_scorecard_settings()
    assert scorecard_settings.scorecard_type == "hybrid_xgboost_pdo"
    assert scorecard_settings.champion_model == "xgboost_traditional_plus_proxy"
    assert scorecard_settings.scaling.pdo == 40.0
    assert scorecard_settings.calibration.method == "platt"
    assert scorecard_settings.cutoff_strategy.review_buffer_points == 10
    assert "base" in scorecard_settings.sensitivity_analysis.scenarios


def test_paths_expose_expected_directories() -> None:
    paths = get_paths()
    assert paths.data_raw == ROOT / "data" / "raw"
    assert paths.notebooks == ROOT / "notebooks"
    assert paths.reports_figures == ROOT / "reports" / "figures"
    assert paths.reports_figures_v2 == ROOT / "reports" / "figures.2"


def test_resolve_report_figure_dir_defaults_to_v2() -> None:
    paths = get_paths()
    assert resolve_report_figure_dir(paths) == ROOT / "reports" / "figures.2"
    assert resolve_report_figure_dir(paths, variant="original") == ROOT / "reports" / "figures"
