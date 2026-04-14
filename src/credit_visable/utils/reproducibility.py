"""Reproducibility helpers for governed model artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def compute_series_hash(values: pd.Series | None) -> str | None:
    """Return a stable hash for a pandas Series."""

    if values is None or values.empty:
        return None
    hashed = pd.util.hash_pandas_object(values.reset_index(drop=True), index=False)
    digest = hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()
    return digest


def compute_frame_fingerprint(frame: pd.DataFrame) -> str:
    """Return a lightweight, stable fingerprint for a DataFrame."""

    payload = {
        "rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def resolve_git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit SHA when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd) if cwd is not None else None,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_run_manifest(
    *,
    raw_frame: pd.DataFrame,
    config_snapshot: dict[str, Any],
    split_hashes: dict[str, str | None],
    model_manifest: dict[str, Any],
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable run manifest."""

    return {
        "git_commit": resolve_git_commit(cwd=cwd),
        "dataset_fingerprint": compute_frame_fingerprint(raw_frame),
        "split_hashes": split_hashes,
        "config_snapshot": config_snapshot,
        "model_manifest": model_manifest,
    }
