"""Top-level package for the credit visable project."""

from credit_visable.config import Settings, load_settings
from credit_visable.utils.paths import ProjectPaths, get_paths, get_project_root

__all__ = [
    "ProjectPaths",
    "Settings",
    "get_paths",
    "get_project_root",
    "load_settings",
]

__version__ = "0.1.0"
