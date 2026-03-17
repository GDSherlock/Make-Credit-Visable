"""Smoke tests for package imports."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def test_package_imports() -> None:
    _ensure_src_on_path()

    import credit_visable
    from credit_visable.data import load_table
    from credit_visable.data.home_credit_bootstrap import bootstrap_home_credit_data
    from credit_visable.explainability import run_shap_placeholder
    from credit_visable.features import build_basic_preprocessor, compute_iv_summary
    from credit_visable.governance import fairness_report_placeholder
    from credit_visable.modeling import (
        evaluate_binary_classifier,
        train_logistic_baseline,
        train_tree_model_placeholder,
    )
    from credit_visable.scoring import build_scorecard_placeholder
    from credit_visable.utils import get_paths

    assert credit_visable.__version__ == "0.1.0"
    assert callable(load_table)
    assert callable(bootstrap_home_credit_data)
    assert callable(build_basic_preprocessor)
    assert callable(compute_iv_summary)
    assert callable(train_logistic_baseline)
    assert callable(train_tree_model_placeholder)
    assert callable(evaluate_binary_classifier)
    assert callable(run_shap_placeholder)
    assert callable(fairness_report_placeholder)
    assert callable(build_scorecard_placeholder)
    assert callable(get_paths)
