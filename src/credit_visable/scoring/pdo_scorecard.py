"""Starter placeholders for PDO-based scorecard development."""

from __future__ import annotations


def build_scorecard_placeholder(
    base_score: int = 600,
    base_odds: float = 50.0,
    points_to_double_odds: int = 20,
) -> dict[str, float | int | str]:
    """Return starter scorecard metadata without implementing score scaling yet."""

    return {
        "base_score": base_score,
        "base_odds": base_odds,
        "points_to_double_odds": points_to_double_odds,
        "ready": False,
        "notes": "TODO: implement PDO scaling once WOE features and calibration are finalized.",
    }
