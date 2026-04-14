"""PDO score scaling helpers for production scorecards."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_visable.config import RiskBandThreshold, ScoreScalingSettings, ScorecardSettings
from credit_visable.scoring.calibration import clip_probabilities


def _to_mapping(value: Any) -> dict[str, Any]:
    """Normalize config objects into plain mappings."""

    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def build_scorecard_metadata(
    base_score: float = 600.0,
    base_odds: float = 20.0,
    points_to_double_odds: float = 40.0,
) -> dict[str, float | str | bool]:
    """Build PDO score scaling metadata."""

    factor = float(points_to_double_odds / np.log(2.0))
    offset = float(base_score - factor * np.log(base_odds))
    return {
        "base_score": float(base_score),
        "base_odds": float(base_odds),
        "points_to_double_odds": float(points_to_double_odds),
        "factor": factor,
        "offset": offset,
        "ready": True,
        "score_formula": "score = base_score + factor * ln(((1-p)/p) / base_odds)",
        "notes": "Production PDO score transform for calibrated PD.",
    }


def build_scorecard_placeholder(
    base_score: int = 600,
    base_odds: float = 20.0,
    points_to_double_odds: int = 40,
) -> dict[str, float | int | str | bool]:
    """Backward-compatible alias for the old placeholder helper."""

    metadata = build_scorecard_metadata(
        base_score=float(base_score),
        base_odds=float(base_odds),
        points_to_double_odds=float(points_to_double_odds),
    )
    metadata["notes"] = "Legacy helper retained for compatibility; PDO score scaling is implemented."
    return metadata


def resolve_scaling_metadata(
    settings: ScorecardSettings | ScoreScalingSettings | dict[str, Any] | None = None,
) -> dict[str, float | str | bool]:
    """Resolve score scaling metadata from config or explicit settings."""

    if settings is None:
        return build_scorecard_metadata()

    if isinstance(settings, ScorecardSettings):
        scaling = settings.scaling
    else:
        scaling = settings

    if isinstance(scaling, ScoreScalingSettings):
        return build_scorecard_metadata(
            base_score=scaling.base_score,
            base_odds=scaling.base_odds,
            points_to_double_odds=scaling.pdo,
        )

    scaling_mapping = _to_mapping(scaling)
    return build_scorecard_metadata(
        base_score=float(scaling_mapping.get("base_score", 600.0)),
        base_odds=float(scaling_mapping.get("base_odds", 20.0)),
        points_to_double_odds=float(scaling_mapping.get("pdo", scaling_mapping.get("points_to_double_odds", 40.0))),
    )


def odds_to_score(
    odds: np.ndarray | pd.Series | float,
    metadata: dict[str, float | str | bool],
) -> np.ndarray:
    """Convert odds to score using PDO metadata."""

    odds_array = np.asarray(odds, dtype=float)
    return metadata["base_score"] + metadata["factor"] * np.log(odds_array / metadata["base_odds"])


def score_to_odds(
    score: np.ndarray | pd.Series | float,
    metadata: dict[str, float | str | bool],
) -> np.ndarray:
    """Convert score back to odds."""

    score_array = np.asarray(score, dtype=float)
    return metadata["base_odds"] * np.exp((score_array - metadata["base_score"]) / metadata["factor"])


def pd_to_score(
    pd_values: np.ndarray | pd.Series | float,
    metadata: dict[str, float | str | bool],
    name: str | None = None,
) -> pd.Series:
    """Convert calibrated PD values to score."""

    clipped = clip_probabilities(pd_values)
    odds = (1.0 - clipped) / clipped
    scores = odds_to_score(odds, metadata)
    if isinstance(pd_values, pd.Series):
        return pd.Series(scores, index=pd_values.index, name=name or pd_values.name)
    return pd.Series(scores, name=name)


def score_to_pd(
    score_values: np.ndarray | pd.Series | float,
    metadata: dict[str, float | str | bool],
    name: str | None = None,
) -> pd.Series:
    """Convert score values back to calibrated PD."""

    odds = score_to_odds(score_values, metadata)
    pd_values = 1.0 / (1.0 + odds)
    if isinstance(score_values, pd.Series):
        return pd.Series(pd_values, index=score_values.index, name=name or score_values.name)
    return pd.Series(pd_values, name=name)


def _normalize_risk_thresholds(
    thresholds: dict[str, Any],
) -> dict[str, RiskBandThreshold]:
    """Normalize band thresholds into dataclass instances."""

    resolved: dict[str, RiskBandThreshold] = {}
    for band_name, threshold in thresholds.items():
        if isinstance(threshold, RiskBandThreshold):
            resolved[band_name] = threshold
            continue
        threshold_mapping = _to_mapping(threshold)
        resolved[band_name] = RiskBandThreshold(
            max_calibrated_pd=float(threshold_mapping["max_calibrated_pd"]),
            min_score=float(threshold_mapping["min_score"]),
        )
    return resolved


def _resolve_band_order(
    thresholds: dict[str, RiskBandThreshold],
    target_population_shares: dict[str, float] | None = None,
) -> list[str]:
    """Resolve low-risk to high-risk band order."""

    if thresholds:
        return [
            band_name
            for band_name, _ in sorted(
                thresholds.items(),
                key=lambda item: item[1].max_calibrated_pd,
            )
        ]
    if target_population_shares:
        return sorted(target_population_shares)
    return ["A", "B", "C", "D", "E"]


def _allocate_band_counts(
    population_size: int,
    band_order: list[str],
    target_population_shares: dict[str, float] | None,
    minimum_band_share: float,
) -> list[int]:
    """Allocate applicant counts to hybrid risk bands."""

    if population_size <= 0 or not band_order:
        return [0] * len(band_order)

    share_values = np.asarray(
        [
            float((target_population_shares or {}).get(band_name, 0.0))
            for band_name in band_order
        ],
        dtype=float,
    )
    if not np.isfinite(share_values).all() or float(share_values.sum()) <= 0.0:
        share_values = np.repeat(1.0 / len(band_order), len(band_order))
    else:
        share_values = share_values / float(share_values.sum())

    raw_counts = share_values * population_size
    counts = np.floor(raw_counts).astype(int)
    remainder = raw_counts - counts
    remaining = int(population_size - counts.sum())
    if remaining > 0:
        for index in np.argsort(-remainder, kind="mergesort")[:remaining]:
            counts[index] += 1

    minimum_count = int(np.floor(float(minimum_band_share) * population_size))
    minimum_count = max(1 if population_size >= len(band_order) else 0, minimum_count)
    if minimum_count * len(band_order) > population_size:
        minimum_count = population_size // len(band_order)

    if minimum_count > 0:
        deficits = np.maximum(minimum_count - counts, 0)
        if deficits.sum() > 0:
            surplus = counts - minimum_count
            donor_order = np.argsort(-surplus, kind="mergesort")
            for recipient_index in np.where(deficits > 0)[0]:
                needed = int(deficits[recipient_index])
                for donor_index in donor_order:
                    available = int(counts[donor_index] - minimum_count)
                    if available <= 0 or needed <= 0:
                        continue
                    transfer = min(available, needed)
                    counts[donor_index] -= transfer
                    counts[recipient_index] += transfer
                    needed -= transfer

    counts = counts.astype(int)
    current_total = int(counts.sum())
    if current_total != population_size:
        counts[-1] += population_size - current_total
    return counts.tolist()


def assign_hybrid_risk_bands(
    frame: pd.DataFrame,
    calibrated_pd_column: str,
    score_column: str,
    risk_band_config: Any,
    target_column: str | None = None,
    name: str = "risk_band",
) -> tuple[pd.Series, pd.DataFrame]:
    """Assign operational risk bands from calibrated PD quantiles.

    The hybrid construction keeps the A->E ordering stable while shaping
    population coverage so the lower-risk bands remain the majority of the book.
    """

    config_mapping = _to_mapping(risk_band_config)
    resolved_thresholds = _normalize_risk_thresholds(config_mapping.get("thresholds", {}))
    target_population_shares = {
        str(key): float(value)
        for key, value in config_mapping.get("target_population_shares", {}).items()
    }
    minimum_band_share = float(config_mapping.get("minimum_band_share", 0.05))
    band_order = _resolve_band_order(resolved_thresholds, target_population_shares)

    if frame.empty:
        empty_series = pd.Series(dtype="object", name=name)
        empty_table = pd.DataFrame(
            columns=[
                "risk_band",
                "count",
                "population_share",
                "target_population_share",
                "bad_count",
                "actual_default_rate",
                "mean_calibrated_pd",
                "mean_score",
                "min_score",
                "max_score",
                "max_calibrated_pd",
            ]
        )
        return empty_series, empty_table

    working_columns = [calibrated_pd_column, score_column]
    if target_column is not None and target_column in frame.columns:
        working_columns.append(target_column)
    ordered = frame[working_columns].copy()
    ordered[calibrated_pd_column] = pd.to_numeric(
        ordered[calibrated_pd_column], errors="coerce"
    ).fillna(1.0)
    ordered[score_column] = pd.to_numeric(
        ordered[score_column], errors="coerce"
    ).fillna(0.0)
    if target_column is not None and target_column in ordered.columns:
        ordered[target_column] = pd.to_numeric(
            ordered[target_column], errors="coerce"
        ).fillna(0.0)

    ordered = ordered.sort_values(
        [calibrated_pd_column, score_column],
        ascending=[True, False],
        kind="mergesort",
    )
    counts = _allocate_band_counts(
        population_size=len(ordered),
        band_order=band_order,
        target_population_shares=target_population_shares,
        minimum_band_share=minimum_band_share,
    )

    band_values = pd.Series(index=ordered.index, dtype="object", name=name)
    start_index = 0
    for band_name, band_count in zip(band_order, counts):
        end_index = start_index + int(band_count)
        band_index = ordered.index[start_index:end_index]
        band_values.loc[band_index] = band_name
        start_index = end_index
    if start_index < len(ordered):
        band_values.loc[ordered.index[start_index:]] = band_order[-1]
    band_values = band_values.reindex(frame.index)

    summary_frame = frame.copy()
    summary_frame[name] = band_values
    if target_column is not None and target_column in summary_frame.columns:
        summary_frame[target_column] = pd.to_numeric(
            summary_frame[target_column], errors="coerce"
        ).fillna(0.0)
    summary_frame[calibrated_pd_column] = pd.to_numeric(
        summary_frame[calibrated_pd_column], errors="coerce"
    ).fillna(1.0)
    summary_frame[score_column] = pd.to_numeric(
        summary_frame[score_column], errors="coerce"
    ).fillna(0.0)

    grouped = (
        summary_frame.groupby(name, observed=False)
        .agg(
            count=(name, "size"),
            bad_count=(target_column, "sum") if target_column is not None and target_column in summary_frame.columns else (calibrated_pd_column, "size"),
            actual_default_rate=(target_column, "mean") if target_column is not None and target_column in summary_frame.columns else (calibrated_pd_column, "mean"),
            mean_calibrated_pd=(calibrated_pd_column, "mean"),
            mean_score=(score_column, "mean"),
            min_score=(score_column, "min"),
            max_score=(score_column, "max"),
            max_calibrated_pd=(calibrated_pd_column, "max"),
        )
        .reset_index()
        .rename(columns={name: "risk_band"})
    )
    grouped["population_share"] = grouped["count"] / len(summary_frame)
    grouped["target_population_share"] = grouped["risk_band"].map(target_population_shares).fillna(np.nan)
    grouped["band_rank"] = grouped["risk_band"].map(
        {band_name: rank for rank, band_name in enumerate(band_order)}
    )
    grouped = grouped.sort_values("band_rank").reset_index(drop=True)
    grouped = grouped[
        [
            "risk_band",
            "count",
            "population_share",
            "target_population_share",
            "bad_count",
            "actual_default_rate",
            "mean_calibrated_pd",
            "mean_score",
            "min_score",
            "max_score",
            "max_calibrated_pd",
        ]
    ]
    if target_column is None or target_column not in frame.columns:
        grouped["bad_count"] = np.nan
        grouped["actual_default_rate"] = grouped["mean_calibrated_pd"]

    return band_values.rename(name), grouped


def build_operational_risk_band_table(
    hybrid_band_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Project a hybrid risk-band summary into a compact threshold table."""

    if hybrid_band_summary.empty:
        return pd.DataFrame(
            columns=[
                "risk_band",
                "target_population_share",
                "population_share",
                "max_calibrated_pd",
                "min_score",
            ]
        )
    available_columns = [
        column
        for column in [
            "risk_band",
            "target_population_share",
            "population_share",
            "max_calibrated_pd",
            "min_score",
        ]
        if column in hybrid_band_summary.columns
    ]
    return hybrid_band_summary[available_columns].copy()


def freeze_risk_band_thresholds(
    frame: pd.DataFrame,
    *,
    calibrated_pd_column: str,
    score_column: str,
    risk_band_config: Any,
) -> pd.DataFrame:
    """Freeze development risk-band thresholds for reuse downstream."""

    _, hybrid_summary = assign_hybrid_risk_bands(
        frame=frame,
        calibrated_pd_column=calibrated_pd_column,
        score_column=score_column,
        risk_band_config=risk_band_config,
        target_column=None,
        name="development_risk_band",
    )
    if hybrid_summary.empty:
        return pd.DataFrame(
            columns=[
                "risk_band",
                "target_population_share",
                "population_share",
                "max_calibrated_pd",
                "min_score",
            ]
        )
    frozen = hybrid_summary[
        [
            "risk_band",
            "target_population_share",
            "population_share",
            "max_calibrated_pd",
            "min_score",
        ]
    ].copy()
    frozen["min_score"] = frozen["min_score"].astype(float)
    frozen["max_calibrated_pd"] = frozen["max_calibrated_pd"].astype(float)
    return frozen.sort_values("min_score", ascending=False, ignore_index=True)


def _risk_band_threshold_mapping(
    band_threshold_table: pd.DataFrame,
) -> dict[str, RiskBandThreshold]:
    if band_threshold_table.empty:
        return {}
    return {
        str(row["risk_band"]): RiskBandThreshold(
            max_calibrated_pd=float(row["max_calibrated_pd"]),
            min_score=float(row["min_score"]),
        )
        for _, row in band_threshold_table.iterrows()
    }


def assign_frozen_risk_bands(
    frame: pd.DataFrame,
    *,
    score_column: str,
    calibrated_pd_column: str,
    band_threshold_table: pd.DataFrame,
    target_column: str | None = None,
    name: str = "risk_band",
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply frozen development thresholds to a new population."""

    threshold_mapping = _risk_band_threshold_mapping(band_threshold_table)
    assigned_bands = assign_risk_band_from_score(
        frame[score_column],
        thresholds=threshold_mapping,
        name=name,
    )
    summary_frame = frame.copy()
    summary_frame[name] = assigned_bands
    summary_frame[score_column] = pd.to_numeric(summary_frame[score_column], errors="coerce")
    summary_frame[calibrated_pd_column] = pd.to_numeric(
        summary_frame[calibrated_pd_column], errors="coerce"
    )
    if target_column is not None and target_column in summary_frame.columns:
        summary_frame[target_column] = pd.to_numeric(
            summary_frame[target_column], errors="coerce"
        )

    grouped = (
        summary_frame.groupby(name, observed=False)
        .agg(
            count=(name, "size"),
            bad_count=(target_column, "sum") if target_column is not None and target_column in summary_frame.columns else (calibrated_pd_column, "size"),
            actual_default_rate=(target_column, "mean") if target_column is not None and target_column in summary_frame.columns else (calibrated_pd_column, "mean"),
            mean_calibrated_pd=(calibrated_pd_column, "mean"),
            mean_score=(score_column, "mean"),
            min_score=(score_column, "min"),
            max_score=(score_column, "max"),
            max_calibrated_pd=(calibrated_pd_column, "max"),
        )
        .reset_index()
        .rename(columns={name: "risk_band"})
    )
    grouped["population_share"] = grouped["count"] / max(len(summary_frame), 1)
    grouped = grouped.merge(
        band_threshold_table[
            [
                "risk_band",
                "target_population_share",
                "max_calibrated_pd",
                "min_score",
            ]
        ],
        on="risk_band",
        how="left",
        suffixes=("_observed", "_frozen"),
    )
    grouped["target_population_share"] = grouped["target_population_share"].astype(float)
    grouped["band_rank"] = grouped["risk_band"].map(
        {
            band_name: rank
            for rank, band_name in enumerate(
                band_threshold_table.sort_values("min_score", ascending=False)["risk_band"].tolist()
            )
        }
    )
    grouped = grouped.sort_values("band_rank").reset_index(drop=True)
    grouped["frozen_min_score"] = grouped["min_score_frozen"]
    grouped["frozen_max_calibrated_pd"] = grouped["max_calibrated_pd_frozen"]
    grouped = grouped.drop(
        columns=["band_rank", "min_score_frozen", "max_calibrated_pd_frozen"]
    )
    return assigned_bands, grouped


def assign_risk_band_from_pd(
    pd_values: np.ndarray | pd.Series,
    thresholds: dict[str, Any],
    name: str = "risk_band",
) -> pd.Series:
    """Assign calibrated PD values to risk bands."""

    resolved_thresholds = _normalize_risk_thresholds(thresholds)
    ordered_thresholds = sorted(
        resolved_thresholds.items(),
        key=lambda item: item[1].max_calibrated_pd,
    )
    pd_array = clip_probabilities(pd_values)
    band_values = np.full(pd_array.shape[0], "Unknown", dtype=object)

    for band_name, threshold in ordered_thresholds:
        mask = pd_array <= threshold.max_calibrated_pd
        band_values = np.where((band_values == "Unknown") & mask, band_name, band_values)

    if isinstance(pd_values, pd.Series):
        return pd.Series(band_values, index=pd_values.index, name=name)
    return pd.Series(band_values, name=name)


def assign_risk_band_from_score(
    score_values: np.ndarray | pd.Series,
    thresholds: dict[str, Any],
    name: str = "risk_band",
) -> pd.Series:
    """Assign score values to risk bands using minimum-score thresholds."""

    resolved_thresholds = _normalize_risk_thresholds(thresholds)
    ordered_thresholds = sorted(
        resolved_thresholds.items(),
        key=lambda item: item[1].min_score,
        reverse=True,
    )
    score_array = np.asarray(score_values, dtype=float)
    band_values = np.full(score_array.shape[0], "Unknown", dtype=object)

    for band_name, threshold in ordered_thresholds:
        mask = score_array >= threshold.min_score
        band_values = np.where((band_values == "Unknown") & mask, band_name, band_values)

    if isinstance(score_values, pd.Series):
        return pd.Series(band_values, index=score_values.index, name=name)
    return pd.Series(band_values, name=name)


def build_risk_band_table(thresholds: dict[str, Any]) -> pd.DataFrame:
    """Build a reporting table for score band thresholds."""

    resolved_thresholds = _normalize_risk_thresholds(thresholds)
    rows = [
        {
            "risk_band": band_name,
            "max_calibrated_pd": threshold.max_calibrated_pd,
            "min_score": threshold.min_score,
        }
        for band_name, threshold in resolved_thresholds.items()
    ]
    return pd.DataFrame(rows).sort_values("min_score", ascending=False).reset_index(drop=True)


def apply_cutoff_policy(
    score_values: np.ndarray | pd.Series,
    approve_min_score: float,
    review_min_score: float,
    name: str = "decision",
) -> pd.Series:
    """Assign approve/review/reject decisions from score cutoffs."""

    numeric_scores = np.asarray(score_values, dtype=float)
    decisions = np.where(
        numeric_scores >= float(approve_min_score),
        "approve",
        np.where(numeric_scores >= float(review_min_score), "review", "reject"),
    )
    if isinstance(score_values, pd.Series):
        return pd.Series(decisions, index=score_values.index, name=name)
    return pd.Series(decisions, name=name)


def build_score_cutoff_grid(
    score_values: np.ndarray | pd.Series,
    step: int = 5,
) -> list[int]:
    """Build a score cutoff grid spanning the observed score range."""

    numeric_scores = np.asarray(score_values, dtype=float)
    if numeric_scores.size == 0:
        return []

    step = max(1, int(step))
    minimum = int(np.floor(np.nanmin(numeric_scores) / step) * step)
    maximum = int(np.ceil(np.nanmax(numeric_scores) / step) * step)
    return list(range(minimum, maximum + step, step))


def build_profit_assumption_config(
    approve_good: float = 1.0,
    approve_bad: float = -5.0,
    reject_good: float = -0.2,
    reject_bad: float = 0.0,
) -> dict[str, float]:
    """Retain the legacy constant-profit helper for compatibility tests."""

    return {
        "approve_good": float(approve_good),
        "approve_bad": float(approve_bad),
        "reject_good": float(reject_good),
        "reject_bad": float(reject_bad),
    }


def compute_threshold_profit_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    thresholds: list[float] | np.ndarray,
    profit_assumptions: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute a simple approve/reject profit curve for backward compatibility."""

    resolved_profit = profit_assumptions or build_profit_assumption_config()
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    if y_true_array.shape[0] != y_score_array.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    total_bad = float(y_true_array.sum())
    rows = []
    for threshold in np.asarray(thresholds, dtype=float):
        approved_mask = y_score_array < threshold
        rejected_mask = ~approved_mask

        approved_good_count = int(np.sum(approved_mask & (y_true_array == 0)))
        approved_bad_count = int(np.sum(approved_mask & (y_true_array == 1)))
        rejected_good_count = int(np.sum(rejected_mask & (y_true_array == 0)))
        rejected_bad_count = int(np.sum(rejected_mask & (y_true_array == 1)))

        total_profit = (
            approved_good_count * resolved_profit["approve_good"]
            + approved_bad_count * resolved_profit["approve_bad"]
            + rejected_good_count * resolved_profit["reject_good"]
            + rejected_bad_count * resolved_profit["reject_bad"]
        )
        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": float(approved_mask.mean()),
                "reject_rate": float(rejected_mask.mean()),
                "approved_good_count": approved_good_count,
                "approved_bad_count": approved_bad_count,
                "rejected_good_count": rejected_good_count,
                "rejected_bad_count": rejected_bad_count,
                "approved_bad_rate": (
                    float(approved_bad_count / approved_mask.sum()) if approved_mask.any() else 0.0
                ),
                "rejected_bad_capture_rate": (
                    float(rejected_bad_count / total_bad) if total_bad > 0 else 0.0
                ),
                "total_profit": float(total_profit),
                "profit_per_applicant": float(total_profit / len(y_true_array))
                if len(y_true_array) > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def select_optimal_profit_threshold(profit_curve: pd.DataFrame) -> dict[str, float | int]:
    """Select the best threshold row from a simple profit curve frame."""

    if profit_curve.empty:
        raise ValueError("profit_curve must not be empty.")
    if "total_profit" not in profit_curve.columns:
        raise KeyError("profit_curve must contain a total_profit column.")

    best_row = profit_curve.sort_values(
        by=["total_profit", "profit_per_applicant", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return {
        key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
        if isinstance(value, (np.integer, int, np.floating, float))
        else value
        for key, value in best_row.to_dict().items()
    }
