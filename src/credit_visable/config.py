"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from credit_visable.utils.paths import get_paths


@dataclass(slots=True)
class Settings:
    """Container for lightweight project defaults."""

    project_name: str = "credit visable"
    random_state: int = 42
    target_column: str = "TARGET"
    id_column: str = "SK_ID_CURR"
    expected_tables: dict[str, str] = field(default_factory=dict)


def _default_config_path() -> Path:
    """Return the default configuration file path."""

    return get_paths().configs / "base.yaml"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load project settings from YAML.

    Parameters
    ----------
    config_path:
        Optional override for the YAML config file.
    """

    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}

    return Settings(
        project_name=payload.get("project_name", "credit visable"),
        random_state=payload.get("random_state", 42),
        target_column=payload.get("target_column", "TARGET"),
        id_column=payload.get("id_column", "SK_ID_CURR"),
        expected_tables=payload.get("expected_tables", {}),
    )
