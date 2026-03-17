"""Placeholders for advanced tree-based model training."""

from __future__ import annotations

from typing import Any


def train_tree_model_placeholder(
    model_name: str = "lightgbm",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Return a starter specification for future tree-model training.

    Advanced dependencies are intentionally not installed in the initial scaffold.
    """

    return {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "ready_to_train": False,
        "notes": (
            "TODO: add a concrete trainer after choosing libraries such as "
            "LightGBM, XGBoost, or CatBoost."
        ),
    }


def main() -> None:
    """Future script entrypoint placeholder."""

    print("TODO: implement advanced model training once the feature pipeline is ready.")


if __name__ == "__main__":
    main()
