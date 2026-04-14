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
    scorecard_config: str = "scorecard.yaml"


@dataclass(slots=True)
class ScoreScalingSettings:
    """Configuration for score-to-odds scaling."""

    pdo: float = 40.0
    base_score: float = 600.0
    base_odds: float = 20.0


@dataclass(slots=True)
class CalibrationSettings:
    """Configuration for probability calibration and monitoring."""

    method: str = "platt"
    fit_dataset: str = "calibration"
    monitor_dataset: str = "holdout_test"
    refit_frequency: str = "quarterly"
    monitoring: dict[str, float] = field(
        default_factory=lambda: {
            "brier_score_max_drift": 0.01,
            "calibration_slope_min": 0.90,
            "calibration_slope_max": 1.10,
            "calibration_intercept_abs_max": 0.05,
            "max_abs_decile_gap": 0.02,
            "score_psi_warn": 0.10,
            "score_psi_fail": 0.25,
        }
    )


@dataclass(slots=True)
class RiskBandThreshold:
    """Score and PD boundaries for one risk band."""

    max_calibrated_pd: float
    min_score: float


@dataclass(slots=True)
class RiskBandSettings:
    """Configuration for calibrated risk bands."""

    construction: str = "frozen_development_thresholds"
    minimum_band_share: float = 0.05
    target_population_shares: dict[str, float] = field(
        default_factory=lambda: {
            "A": 0.30,
            "B": 0.25,
            "C": 0.20,
            "D": 0.15,
            "E": 0.10,
        }
    )
    thresholds: dict[str, RiskBandThreshold] = field(
        default_factory=lambda: {
            "A": RiskBandThreshold(max_calibrated_pd=0.02, min_score=650.0),
            "B": RiskBandThreshold(max_calibrated_pd=0.04, min_score=610.0),
            "C": RiskBandThreshold(max_calibrated_pd=0.07, min_score=575.0),
            "D": RiskBandThreshold(max_calibrated_pd=0.12, min_score=540.0),
            "E": RiskBandThreshold(max_calibrated_pd=1.00, min_score=0.0),
        }
    )


@dataclass(slots=True)
class CutoffGuardrailSettings:
    """Constraints applied during cutoff optimization."""

    max_approved_book_bad_rate: float = 0.07
    max_manual_review_share: float = 0.15
    min_approval_rate: float = 0.25


@dataclass(slots=True)
class CutoffStrategySettings:
    """Configuration for score cutoff selection."""

    objective: str = "maximize_expected_value"
    review_buffer_points: int = 10
    guardrails: CutoffGuardrailSettings = field(default_factory=CutoffGuardrailSettings)
    final_cutoff_rule: str = "optimal_cutoff_under_guardrails"


@dataclass(slots=True)
class UnitEconomicsSettings:
    """Configuration for borrower-level decision economics."""

    data_drivers: dict[str, str] = field(
        default_factory=lambda: {
            "principal": "AMT_CREDIT",
            "payment": "AMT_ANNUITY",
            "annual_income": "AMT_INCOME_TOTAL",
            "goods_price": "AMT_GOODS_PRICE",
        }
    )
    derived_variables: dict[str, str] = field(
        default_factory=lambda: {
            "monthly_income_proxy": "AMT_INCOME_TOTAL / 12",
            "term_proxy_months": "clip(AMT_CREDIT / AMT_ANNUITY, 6, 48)",
            "payment_burden": "clip(AMT_ANNUITY / monthly_income_proxy, 0.05, 0.60)",
            "advance_ratio": "clip(AMT_CREDIT / coalesce(AMT_GOODS_PRICE, AMT_CREDIT), 0.80, 1.50)",
        }
    )
    assumptions: dict[str, float] = field(
        default_factory=lambda: {
            "net_margin_rate_annual": 0.14,
            "ead_rate": 0.85,
            "base_lgd": 0.75,
            "burden_lgd_slope": 0.25,
            "advance_lgd_slope": 0.10,
            "reject_good_capture": 0.70,
            "reject_bad_loss_avoidance": 0.85,
        }
    )
    payoffs: dict[str, str] = field(
        default_factory=lambda: {
            "lgd": "clip(base_lgd + burden_lgd_slope * max(payment_burden - 0.20, 0) + advance_lgd_slope * max(advance_ratio - 1.0, 0), 0.65, 0.90)",
            "approve_good": "AMT_CREDIT * net_margin_rate_annual * (term_proxy_months / 12)",
            "approve_bad": "-(AMT_CREDIT * ead_rate * lgd)",
            "reject_good": "-reject_good_capture * approve_good",
            "reject_bad": "reject_bad_loss_avoidance * (AMT_CREDIT * ead_rate * lgd)",
        }
    )
    costs: dict[str, str] = field(
        default_factory=lambda: {
            "processing_cost": "30 + 0.0005 * AMT_CREDIT",
            "review_cost": "90 + 0.0010 * AMT_CREDIT",
        }
    )


@dataclass(slots=True)
class SensitivityAnalysisSettings:
    """Scenario overrides used for cutoff sensitivity analysis."""

    scenarios: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "base": {
                "net_margin_rate_annual": 0.14,
                "base_lgd": 0.75,
                "ead_rate": 0.85,
                "reject_good_capture": 0.70,
                "cost_multiplier": 1.00,
            },
            "low_margin": {
                "net_margin_rate_annual": 0.10,
                "base_lgd": 0.75,
                "ead_rate": 0.85,
                "reject_good_capture": 0.70,
                "cost_multiplier": 1.00,
            },
            "high_loss": {
                "net_margin_rate_annual": 0.14,
                "base_lgd": 0.85,
                "ead_rate": 0.90,
                "reject_good_capture": 0.70,
                "cost_multiplier": 1.00,
            },
            "high_opex": {
                "net_margin_rate_annual": 0.14,
                "base_lgd": 0.75,
                "ead_rate": 0.85,
                "reject_good_capture": 0.70,
                "cost_multiplier": 1.50,
            },
            "combined_stress": {
                "net_margin_rate_annual": 0.10,
                "base_lgd": 0.85,
                "ead_rate": 0.90,
                "reject_good_capture": 0.80,
                "cost_multiplier": 1.50,
            },
        }
    )


@dataclass(slots=True)
class ScorecardSettings:
    """Top-level production scorecard configuration."""

    scorecard_type: str = "hybrid_xgboost_pdo"
    champion_model: str = "xgboost_traditional_plus_proxy"
    scaling: ScoreScalingSettings = field(default_factory=ScoreScalingSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    risk_bands: RiskBandSettings = field(default_factory=RiskBandSettings)
    cutoff_strategy: CutoffStrategySettings = field(default_factory=CutoffStrategySettings)
    unit_economics: UnitEconomicsSettings = field(default_factory=UnitEconomicsSettings)
    sensitivity_analysis: SensitivityAnalysisSettings = field(default_factory=SensitivityAnalysisSettings)


def _default_config_path() -> Path:
    """Return the default project configuration path."""

    return get_paths().configs / "base.yaml"


def _resolve_scorecard_config_path(config_path: str | Path | None = None) -> Path:
    """Return the scorecard config path, allowing project-level indirection."""

    if config_path is not None:
        return Path(config_path)

    settings = load_settings()
    configured_path = Path(settings.scorecard_config)
    if configured_path.is_absolute():
        return configured_path
    return get_paths().configs / configured_path


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    """Load a YAML document into a plain mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    return payload


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load project settings from YAML."""

    path = Path(config_path) if config_path is not None else _default_config_path()
    payload = _load_yaml_payload(path)

    return Settings(
        project_name=payload.get("project_name", "credit visable"),
        random_state=payload.get("random_state", 42),
        target_column=payload.get("target_column", "TARGET"),
        id_column=payload.get("id_column", "SK_ID_CURR"),
        expected_tables=payload.get("expected_tables", {}),
        scorecard_config=payload.get("scorecard_config", "scorecard.yaml"),
    )


def load_scorecard_settings(config_path: str | Path | None = None) -> ScorecardSettings:
    """Load the production scorecard configuration from YAML."""

    path = _resolve_scorecard_config_path(config_path)
    payload = _load_yaml_payload(path)

    scaling_payload = payload.get("scaling", {})
    calibration_payload = payload.get("calibration", {})
    monitoring_payload = calibration_payload.get("monitoring", {})
    risk_band_payload = payload.get("risk_bands", {})
    cutoff_payload = payload.get("cutoff_strategy", {})
    unit_economics_payload = payload.get("unit_economics", {})
    sensitivity_payload = payload.get("sensitivity_analysis", {})

    thresholds = {
        band_name: RiskBandThreshold(
            max_calibrated_pd=float(threshold_payload["max_calibrated_pd"]),
            min_score=float(threshold_payload["min_score"]),
        )
        for band_name, threshold_payload in risk_band_payload.get("thresholds", {}).items()
    }

    return ScorecardSettings(
        scorecard_type=payload.get("scorecard_type", "hybrid_xgboost_pdo"),
        champion_model=payload.get("champion_model", "xgboost_traditional_plus_proxy"),
        scaling=ScoreScalingSettings(
            pdo=float(scaling_payload.get("pdo", 40.0)),
            base_score=float(scaling_payload.get("base_score", 600.0)),
            base_odds=float(scaling_payload.get("base_odds", 20.0)),
        ),
        calibration=CalibrationSettings(
            method=str(calibration_payload.get("method", "platt")),
            fit_dataset=str(calibration_payload.get("fit_dataset", "calibration")),
            monitor_dataset=str(calibration_payload.get("monitor_dataset", "holdout_test")),
            refit_frequency=str(calibration_payload.get("refit_frequency", "quarterly")),
            monitoring={str(key): float(value) for key, value in monitoring_payload.items()}
            if monitoring_payload
            else CalibrationSettings().monitoring,
        ),
        risk_bands=RiskBandSettings(
            construction=str(risk_band_payload.get("construction", "frozen_development_thresholds")),
            minimum_band_share=float(risk_band_payload.get("minimum_band_share", 0.05)),
            target_population_shares={
                str(key): float(value)
                for key, value in risk_band_payload.get("target_population_shares", {}).items()
            }
            or RiskBandSettings().target_population_shares,
            thresholds=thresholds or RiskBandSettings().thresholds,
        ),
        cutoff_strategy=CutoffStrategySettings(
            objective=str(cutoff_payload.get("objective", "maximize_expected_value")),
            review_buffer_points=int(cutoff_payload.get("review_buffer_points", 10)),
            guardrails=CutoffGuardrailSettings(
                max_approved_book_bad_rate=float(
                    cutoff_payload.get("guardrails", {}).get("max_approved_book_bad_rate", 0.07)
                ),
                max_manual_review_share=float(
                    cutoff_payload.get("guardrails", {}).get("max_manual_review_share", 0.15)
                ),
                min_approval_rate=float(
                    cutoff_payload.get("guardrails", {}).get("min_approval_rate", 0.25)
                ),
            ),
            final_cutoff_rule=str(
                cutoff_payload.get("final_cutoff_rule", "optimal_cutoff_under_guardrails")
            ),
        ),
        unit_economics=UnitEconomicsSettings(
            data_drivers={
                str(key): str(value)
                for key, value in unit_economics_payload.get("data_drivers", {}).items()
            }
            or UnitEconomicsSettings().data_drivers,
            derived_variables={
                str(key): str(value)
                for key, value in unit_economics_payload.get("derived_variables", {}).items()
            }
            or UnitEconomicsSettings().derived_variables,
            assumptions={
                str(key): float(value)
                for key, value in unit_economics_payload.get("assumptions", {}).items()
            }
            or UnitEconomicsSettings().assumptions,
            payoffs={
                str(key): str(value)
                for key, value in unit_economics_payload.get("payoffs", {}).items()
            }
            or UnitEconomicsSettings().payoffs,
            costs={
                str(key): str(value)
                for key, value in unit_economics_payload.get("costs", {}).items()
            }
            or UnitEconomicsSettings().costs,
        ),
        sensitivity_analysis=SensitivityAnalysisSettings(
            scenarios={
                str(scenario_name): {
                    str(key): float(value) for key, value in scenario_payload.items()
                }
                for scenario_name, scenario_payload in sensitivity_payload.get("scenarios", {}).items()
            }
            or SensitivityAnalysisSettings().scenarios
        ),
    )
