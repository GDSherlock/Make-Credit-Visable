"""Guardrails for report-facing notebook plot text."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REPORT_NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda.ipynb",
    ROOT / "notebooks" / "03_modeling_baseline.ipynb",
    ROOT / "notebooks" / "04_modeling_advanced.ipynb",
    ROOT / "notebooks" / "05_xai_fairness.ipynb",
    ROOT / "notebooks" / "06_scorecard_cutoff.ipynb",
]
PLOT_TEXT_TOKENS = (
    "set_title(",
    "suptitle(",
    "set_xlabel(",
    "set_ylabel(",
    "plt.title(",
    "title=",
    "xlabel=",
    "ylabel=",
)


def _is_non_ascii(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


def test_report_plot_text_uses_ascii_literals() -> None:
    offending_lines: list[str] = []

    for notebook_path in REPORT_NOTEBOOKS:
        notebook_payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook_payload["cells"]):
            if cell.get("cell_type") != "code":
                continue
            for line in cell.get("source", []):
                if any(token in line for token in PLOT_TEXT_TOKENS) and _is_non_ascii(line):
                    offending_lines.append(
                        f"{notebook_path.name}:cell{cell_index}:{line.rstrip()}"
                    )

    assert not offending_lines, "Non-ASCII plot text found:\n" + "\n".join(offending_lines)
