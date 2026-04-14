"""Generate repaired figure outputs under ``reports/figures.2``."""

from __future__ import annotations

import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_visable.config import load_settings
from credit_visable.utils import (
    REPORT_COLOR_PALETTE,
    REPORT_LINESTYLES,
    add_conclusion_annotation,
    annotate_bar_values,
    apply_report_style,
    build_figure_manifest,
    build_figure_quality_fields,
    format_percent_axis,
    get_paths,
    place_legend_inside,
    to_builtin,
    wrap_tick_labels,
)


PHASE_CONTACT_SHEET_NAMES = ("phase1", "phase3", "phase4", "phase5", "phase6")
LEGACY_PHASE_SUMMARY_PATHS = {
    "phase0": ("runtime_summary.json",),
    "phase1": ("eda", "eda_summary.json"),
    "phase2": ("preprocessing", "processing_methods_summary.json"),
    "phase3": ("modeling_baseline", "summary.json"),
    "phase4": ("modeling_advanced", "summary.json"),
    "phase5": ("xai_fairness", "summary.json"),
}
FIGURE_QUALITY_STATUS = {
    "phase0": "no_figures",
    "phase1": "partially_repaired_v2",
    "phase2": "no_figures",
    "phase3": "repaired_v2",
    "phase4": "repaired_v2",
    "phase5": "partially_repaired_v2",
    "phase6": "repaired_v2",
}
PHASE1_BUSINESS_VIEW_SPECS = [
    {"column": "CODE_GENDER", "top_n": 3, "business_view": "phase1_code_gender_business_view.png", "rate_slice": "phase1_code_gender_default_rate_slice.png"},
    {"column": "FLAG_OWN_CAR", "top_n": 2, "business_view": "phase1_flag_own_car_business_view.png", "rate_slice": None},
    {"column": "FLAG_OWN_REALTY", "top_n": 2, "business_view": "phase1_flag_own_realty_business_view.png", "rate_slice": None},
    {"column": "NAME_EDUCATION_TYPE", "top_n": 5, "business_view": "phase1_name_education_type_business_view.png", "rate_slice": None},
    {"column": "NAME_FAMILY_STATUS", "top_n": 5, "business_view": "phase1_name_family_status_business_view.png", "rate_slice": "phase1_name_family_status_default_rate_slice.png"},
    {"column": "NAME_INCOME_TYPE", "top_n": 8, "business_view": "phase1_name_income_type_business_view.png", "rate_slice": "phase1_name_income_type_default_rate_slice.png"},
    {"column": "OCCUPATION_TYPE", "top_n": 8, "business_view": "phase1_occupation_type_business_view.png", "rate_slice": None},
    {"column": "ORGANIZATION_TYPE", "top_n": 8, "business_view": "phase1_organization_type_business_view.png", "rate_slice": None},
]
PHASE5_GROUP_METRIC_FILES = [
    ("age_band", "actual_default_rate", "phase5_age_band_actual_default_rate.png"),
    ("age_band", "approval_rate", "phase5_age_band_approval_rate.png"),
    ("age_band", "mean_predicted_pd", "phase5_age_band_mean_predicted_pd.png"),
    ("city_work_mismatch_group", "actual_default_rate", "phase5_city_work_mismatch_actual_default_rate.png"),
    ("family_status_group", "actual_default_rate", "phase5_family_status_actual_default_rate.png"),
    ("family_status_group", "approval_rate", "phase5_family_status_approval_rate.png"),
    ("family_status_group", "mean_predicted_pd", "phase5_family_status_mean_predicted_pd.png"),
    ("occupation_group", "actual_default_rate", "phase5_occupation_group_actual_default_rate.png"),
    ("organization_group", "actual_default_rate", "phase5_organization_group_actual_default_rate.png"),
    ("region_rating_group", "actual_default_rate", "phase5_region_rating_actual_default_rate.png"),
    ("region_rating_group", "approval_rate", "phase5_region_rating_approval_rate.png"),
]
PHASE6_GROUP_POLICY_FILES = [
    ("age_band", "age_band_final_policy"),
    ("family_status_group", "family_status_final_policy"),
    ("region_rating_group", "region_rating_final_policy"),
]
PHASE6_LEGACY_EXCLUDED_FILES = {
    "phase6_xgboost_traditional_plus_proxy_policy_scenario_composition.png",
    "phase6_xgboost_traditional_plus_proxy_age_band_balanced_policy.png",
    "phase6_xgboost_traditional_plus_proxy_family_status_balanced_policy.png",
    "phase6_xgboost_traditional_plus_proxy_region_rating_balanced_policy.png",
    "phase6_xgboost_traditional_plus_proxy_cutoff_approval_rate.png",
    "phase6_xgboost_traditional_plus_proxy_cutoff_approved_bad_rate.png",
    "phase6_xgboost_traditional_plus_proxy_cutoff_rejected_bad_capture_rate.png",
    "phase6_xgboost_traditional_plus_proxy_cutoff_profit_curve.png",
    "phase6_xgboost_traditional_plus_proxy_risk_band_mean_predicted_pd.png",
}
PHASE3_DISPLAY_NAMES = {
    "traditional_core": "LR Core",
    "traditional_plus_proxy": "LR Core+Proxy",
}
PHASE4_DISPLAY_NAMES = {
    "logistic_traditional_core": "LR Core",
    "xgboost_traditional_core": "XGB Core",
    "logistic_traditional_plus_proxy": "LR Core+Proxy",
    "xgboost_traditional_plus_proxy": "XGB Core+Proxy",
}


def _safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _register_label_audit(
    label_audit: dict[str, dict[str, object]],
    figure_stem: str,
    *,
    identifier_mode: str = "not_needed",
    overlap_warning: bool = False,
    note: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "identifier_present": identifier_mode != "not_needed",
        "identifier_mode": identifier_mode,
        "overlap_warning": overlap_warning,
    }
    if note:
        payload["note"] = note
    label_audit[figure_stem] = payload


def _save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    label_audit: dict[str, dict[str, object]] | None = None,
    identifier_mode: str = "not_needed",
    overlap_warning: bool = False,
    note: str | None = None,
) -> None:
    _safe_mkdir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if label_audit is not None:
        _register_label_audit(
            label_audit,
            path.stem,
            identifier_mode=identifier_mode,
            overlap_warning=overlap_warning,
            note=note,
        )


def _set_heatmap_colorbar_label(ax: plt.Axes, label: str) -> None:
    if ax.collections:
        colorbar = ax.collections[0].colorbar
        if colorbar is not None:
            colorbar.set_label(label)


def _series_y_at_x(x_values: np.ndarray, y_values: np.ndarray, target_x: float) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    frame = (
        pd.DataFrame({"x": x[mask], "y": y[mask]})
        .groupby("x", as_index=False, sort=True)["y"]
        .mean()
        .sort_values("x")
    )
    x_unique = frame["x"].to_numpy()
    y_unique = frame["y"].to_numpy()
    if len(x_unique) == 1:
        return float(y_unique[0])
    clipped_x = float(np.clip(target_x, x_unique.min(), x_unique.max()))
    return float(np.interp(clipped_x, x_unique, y_unique))


def _find_near_overlap_pairs(
    series_data: list[dict[str, object]],
    *,
    relative_tolerance: float = 0.05,
) -> list[tuple[str, str]]:
    overlap_pairs: list[tuple[str, str]] = []
    for left, right in combinations(series_data, 2):
        left_x = np.asarray(left["x"], dtype=float)
        left_y = np.asarray(left["y"], dtype=float)
        right_x = np.asarray(right["x"], dtype=float)
        right_y = np.asarray(right["y"], dtype=float)
        if len(left_x) < 2 or len(right_x) < 2:
            continue
        start_x = max(np.nanmin(left_x), np.nanmin(right_x))
        end_x = min(np.nanmax(left_x), np.nanmax(right_x))
        if not np.isfinite(start_x) or not np.isfinite(end_x) or start_x >= end_x:
            continue
        grid = np.linspace(start_x, end_x, 40)
        left_interp = np.array([_series_y_at_x(left_x, left_y, value) for value in grid])
        right_interp = np.array([_series_y_at_x(right_x, right_y, value) for value in grid])
        if not np.isfinite(left_interp).all() or not np.isfinite(right_interp).all():
            continue
        combined = np.concatenate([left_interp, right_interp])
        vertical_span = float(np.nanmax(combined) - np.nanmin(combined))
        vertical_span = max(vertical_span, 1e-6)
        mean_gap = float(np.mean(np.abs(left_interp - right_interp))) / vertical_span
        max_gap = float(np.max(np.abs(left_interp - right_interp))) / vertical_span
        if mean_gap <= relative_tolerance and max_gap <= relative_tolerance * 2.5:
            overlap_pairs.append((str(left["label"]), str(right["label"])))
    return overlap_pairs


def _format_overlap_note(overlap_pairs: list[tuple[str, str]]) -> str | None:
    if not overlap_pairs:
        return None
    display_pairs = [" ~ ".join(pair) for pair in overlap_pairs[:2]]
    note = "Near overlap: " + "; ".join(display_pairs)
    if len(overlap_pairs) > 2:
        note += "; ..."
    return note


def _add_direct_line_labels(
    ax: plt.Axes,
    series_data: list[dict[str, object]],
    *,
    anchor_fraction: float = 0.82,
    min_gap_fraction: float = 0.06,
) -> None:
    if not series_data:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    anchor_x = x_min + (x_range * anchor_fraction)
    text_x = x_max - (x_range * 0.02)
    label_margin = y_range * 0.04
    min_gap = y_range * min_gap_fraction

    labels: list[dict[str, float | str]] = []
    for entry in series_data:
        y_anchor = _series_y_at_x(np.asarray(entry["x"]), np.asarray(entry["y"]), anchor_x)
        if not np.isfinite(y_anchor):
            continue
        labels.append(
            {
                "label": str(entry["label"]),
                "color": str(entry["color"]),
                "y_anchor": y_anchor,
            }
        )
    if not labels:
        return

    labels.sort(key=lambda item: float(item["y_anchor"]))
    lower_bound = y_min + label_margin
    upper_bound = y_max - label_margin
    for index, item in enumerate(labels):
        desired = max(float(item["y_anchor"]), lower_bound)
        if index > 0:
            desired = max(desired, float(labels[index - 1]["y_text"]) + min_gap)
        item["y_text"] = min(desired, upper_bound)

    for index in range(len(labels) - 2, -1, -1):
        labels[index]["y_text"] = min(
            float(labels[index]["y_text"]),
            float(labels[index + 1]["y_text"]) - min_gap,
        )
    for item in labels:
        item["y_text"] = float(np.clip(float(item["y_text"]), lower_bound, upper_bound))
        ax.plot(
            [anchor_x, text_x - (x_range * 0.01)],
            [float(item["y_anchor"]), float(item["y_text"])],
            color=str(item["color"]),
            linewidth=1.0,
            alpha=0.7,
        )
        ax.text(
            text_x,
            float(item["y_text"]),
            str(item["label"]),
            ha="right",
            va="center",
            fontsize=8.5,
            color=str(item["color"]),
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": str(item["color"]),
                "alpha": 0.92,
            },
        )


def _build_phase_label_audit(
    repaired_manifest: dict[str, str],
    label_audit: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        figure_key: label_audit.get(
            figure_key,
            {
                "identifier_present": False,
                "identifier_mode": "not_needed",
                "overlap_warning": False,
            },
        )
        for figure_key in sorted(repaired_manifest)
    }


def _build_phase_manifest(directory: Path, phase_prefix: str) -> dict[str, str]:
    return build_figure_manifest({path.stem: path for path in sorted(directory.glob(f"{phase_prefix}_*.png"))})


def _copy_original_figures(paths) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    _safe_mkdir(paths.reports_figures_v2)
    for obsolete_name in PHASE6_LEGACY_EXCLUDED_FILES:
        obsolete_path = paths.reports_figures_v2 / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    for source in sorted(paths.reports_figures.glob("phase*.png")):
        if source.name in PHASE6_LEGACY_EXCLUDED_FILES:
            continue
        destination = paths.reports_figures_v2 / source.name
        shutil.copy2(source, destination)
        copied[source.name] = destination
    return copied


def _setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    apply_report_style(
        **{
            "figure.figsize": (9.5, 5.6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 160,
        }
    )


def _load_application_train(paths) -> pd.DataFrame:
    settings = load_settings()
    app_path = paths.data_raw / settings.expected_tables["application_train"]
    frame = pd.read_csv(app_path)
    frame["AGE_YEARS"] = (-pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce")) / 365.25
    employed = pd.to_numeric(frame["DAYS_EMPLOYED"], errors="coerce")
    employed = employed.mask(employed == 365243, np.nan)
    frame["YEARS_EMPLOYED"] = (-employed) / 365.25
    income = pd.to_numeric(frame["AMT_INCOME_TOTAL"], errors="coerce")
    credit = pd.to_numeric(frame["AMT_CREDIT"], errors="coerce").replace({0: np.nan})
    annuity = pd.to_numeric(frame["AMT_ANNUITY"], errors="coerce")
    frame["INCOME_CREDIT_RATIO"] = income / credit
    frame["ANNUITY_INCOME_RATIO"] = annuity / income.replace({0: np.nan})
    return frame


def _render_phase1_core_figures(paths, label_audit: dict[str, dict[str, object]]) -> None:
    output_dir = paths.reports_figures_v2
    _setup_plot_style()

    top_missing = _load_csv(paths.data_processed / "eda" / "top_missingness.csv")
    age_band_summary = _load_csv(paths.data_processed / "eda" / "age_band_summary.csv")
    age_woe_detail = _load_csv(paths.data_processed / "eda" / "age_woe_detail.csv")
    iv_summary = _load_csv(paths.data_processed / "eda" / "iv_summary.csv")
    correlation = _load_csv(paths.data_processed / "eda" / "correlation_matrix.csv", index_col=0)
    age_income = _load_csv(paths.data_processed / "eda" / "age_income_default_heatmap.csv").set_index("age_band")
    income_family = _load_csv(paths.data_processed / "eda" / "income_family_default_heatmap.csv").set_index("NAME_FAMILY_STATUS")
    app_df = _load_application_train(paths)

    target_summary = (
        app_df["TARGET"].value_counts(dropna=False).sort_index().rename_axis("target").reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.barplot(data=target_summary, x="target", y="count", palette=[REPORT_COLOR_PALETTE["good"], REPORT_COLOR_PALETTE["bad"]], ax=ax)
    ax.set_title("Phase 1 Target Distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("Application count")
    annotate_bar_values(ax, value_format="{:.0f}", padding=2500, max_bars=4)
    add_conclusion_annotation(ax, f"Observed bad rate: {app_df['TARGET'].mean():.1%}")
    _save_figure(fig, output_dir / "phase1_main_target_distribution.png", label_audit=label_audit)

    missing_frame = top_missing.sort_values("missing_pct", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    sns.barplot(data=missing_frame, x="missing_pct", y="column", color=REPORT_COLOR_PALETTE["bad"], ax=ax)
    ax.set_title("Phase 1 Top Missingness")
    ax.set_xlabel("Missing share (%)")
    ax.set_ylabel("Column")
    annotate_bar_values(ax, orientation="horizontal", value_format="{:.1f}", padding=0.5)
    add_conclusion_annotation(ax, "High-missing fields belong in governance review before model use.")
    _save_figure(fig, output_dir / "phase1_main_missingness_top20.png", label_audit=label_audit)

    mask = np.triu(np.ones_like(correlation, dtype=bool))
    fig, ax = plt.subplots(figsize=(10.2, 8.2))
    sns.heatmap(correlation, mask=mask, cmap="coolwarm", center=0.0, square=True, annot=False, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Phase 1 Key Numeric Correlation")
    _set_heatmap_colorbar_label(ax, "Correlation")
    add_conclusion_annotation(ax, "Use this view to spot redundancy before encoding and model uplift tests.")
    _save_figure(fig, output_dir / "phase1_main_correlation_heatmap.png", label_audit=label_audit)

    age_frame = age_band_summary.copy()
    age_frame["label"] = age_frame["group"].astype(str) + " (n=" + age_frame["count"].map(lambda v: f"{int(v):,}") + ")"
    age_frame = age_frame.sort_values("target_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    sns.barplot(data=age_frame, x="target_rate", y="label", color=REPORT_COLOR_PALETTE["accent"], ax=ax)
    ax.set_title("Phase 1 Age Band Default Rate")
    ax.set_xlabel("Observed default rate")
    ax.set_ylabel("Age band")
    format_percent_axis(ax, axis="x", decimals=0)
    annotate_bar_values(ax, orientation="horizontal", value_format="{:.1%}", padding=0.003)
    add_conclusion_annotation(ax, "Younger age bands show the highest observed risk in-sample.")
    _save_figure(fig, output_dir / "phase1_age_band_default_rate.png", label_audit=label_audit)

    plot_frame = age_woe_detail.loc[age_woe_detail["bin"] != "Missing"].copy()
    plot_frame = plot_frame.sort_values("bin_order")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.plot(plot_frame["bin"], plot_frame["woe"], marker="o", linewidth=2.4, color=REPORT_COLOR_PALETTE["good"])
    ax.axhline(0.0, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
    ax.set_title("Phase 1 Age WOE Trend")
    ax.set_xlabel("Age bin")
    ax.set_ylabel("WOE")
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    add_conclusion_annotation(ax, f"Highest bad-rate bin: {plot_frame.sort_values('event_rate', ascending=False).iloc[0]['bin']}")
    _save_figure(fig, output_dir / "phase1_age_band_woe_trend.png", label_audit=label_audit)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    sns.heatmap(age_income, annot=True, fmt=".1%", cmap="Reds", linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Phase 1 Age x Income Default Heatmap")
    ax.set_xlabel("Income band")
    ax.set_ylabel("Age band")
    _set_heatmap_colorbar_label(ax, "Observed default rate")
    add_conclusion_annotation(ax, "Dark cells flag concentrated risk, not policy decisions.")
    _save_figure(fig, output_dir / "phase1_age_income_default_heatmap.png", label_audit=label_audit)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    sns.heatmap(income_family, annot=True, fmt=".1%", cmap="Reds", linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Phase 1 Income x Family Status Default Heatmap")
    ax.set_xlabel("Income band")
    ax.set_ylabel("Family status")
    _set_heatmap_colorbar_label(ax, "Observed default rate")
    add_conclusion_annotation(ax, "Risk dispersion matters more than any single family-status average.")
    _save_figure(fig, output_dir / "phase1_income_family_default_heatmap.png", label_audit=label_audit)

    top_iv = iv_summary.head(20).sort_values("iv", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    sns.barplot(data=top_iv, x="iv", y="feature", color=REPORT_COLOR_PALETTE["bad"], ax=ax)
    ax.set_title("Phase 1 Top IV Features")
    ax.set_xlabel("Information value")
    ax.set_ylabel("Feature")
    annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.004)
    add_conclusion_annotation(ax, f"Top feature: {iv_summary.iloc[0]['feature']}")
    _save_figure(fig, output_dir / "phase1_top20_iv_features.png", label_audit=label_audit)

    for spec in PHASE1_BUSINESS_VIEW_SPECS:
        column = spec["column"]
        if column not in app_df.columns:
            continue
        count_summary = (
            app_df[column]
            .fillna("(Missing)")
            .astype(str)
            .value_counts()
            .head(spec["top_n"])
            .rename_axis("group")
            .reset_index(name="count")
        )
        groups = count_summary["group"].tolist()
        rate_summary = (
            app_df.assign(_group=app_df[column].fillna("(Missing)").astype(str))
            .loc[lambda df: df["_group"].isin(groups)]
            .groupby("_group", observed=False)["TARGET"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"_group": "group", "mean": "target_rate"})
        )
        plot_frame = (
            count_summary.merge(rate_summary[["group", "target_rate"]], on="group", how="left")
            .sort_values("count", ascending=True)
            .reset_index(drop=True)
        )
        plot_frame["label"] = plot_frame["group"] + " (n=" + plot_frame["count"].map(lambda v: f"{int(v):,}") + ")"

        fig, axes = plt.subplots(1, 2, figsize=(13.8, max(4.6, 0.5 * len(plot_frame))))
        sns.barplot(data=plot_frame, x="count", y="label", color=REPORT_COLOR_PALETTE["good"], ax=axes[0])
        axes[0].set_title(f"{column}: group size")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel(column)
        annotate_bar_values(axes[0], orientation="horizontal", value_format="{:.0f}", padding=max(plot_frame["count"].max() * 0.01, 1))

        sns.barplot(data=plot_frame, x="target_rate", y="label", color=REPORT_COLOR_PALETTE["highlight"], ax=axes[1])
        axes[1].set_title(f"{column}: default rate")
        axes[1].set_xlabel("Observed default rate")
        axes[1].set_ylabel("")
        format_percent_axis(axes[1], axis="x", decimals=0)
        annotate_bar_values(axes[1], orientation="horizontal", value_format="{:.1%}", padding=0.003)
        wrap_tick_labels(axes[0], axis="y", width=20)
        wrap_tick_labels(axes[1], axis="y", width=20)
        _save_figure(fig, output_dir / spec["business_view"], label_audit=label_audit)

        if spec["rate_slice"] is not None:
            fig, ax = plt.subplots(figsize=(9.4, max(4.4, 0.45 * len(plot_frame))))
            sns.barplot(data=plot_frame, x="target_rate", y="label", color=REPORT_COLOR_PALETTE["accent"], ax=ax)
            ax.set_title(f"{column}: default rate slice")
            ax.set_xlabel("Observed default rate")
            ax.set_ylabel(column)
            format_percent_axis(ax, axis="x", decimals=0)
            annotate_bar_values(ax, orientation="horizontal", value_format="{:.1%}", padding=0.003)
            add_conclusion_annotation(ax, "Rate gaps require context on support and proxy sensitivity.")
            _save_figure(fig, output_dir / spec["rate_slice"], label_audit=label_audit)


def _compute_ks_frame(y_true: pd.Series, y_score: pd.Series) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "ks": tpr - fpr, "threshold": thresholds})


def _render_phase3_figures(paths, label_audit: dict[str, dict[str, object]]) -> None:
    output_dir = paths.reports_figures_v2
    _setup_plot_style()

    summary = _load_json(paths.data_processed / "modeling_baseline" / "summary.json")
    comparison = _load_csv(paths.data_processed / "modeling_baseline" / "validation_score_comparison.csv")
    calibration = _load_csv(paths.data_processed / "modeling_baseline" / "calibration_curve_points.csv")
    gain = _load_csv(paths.data_processed / "modeling_baseline" / "gain_curve_points.csv")
    lift = _load_csv(paths.data_processed / "modeling_baseline" / "lift_curve_points.csv")
    y_true = comparison["TARGET"]
    feature_specs = [
        ("traditional_core", "traditional_core_predicted_pd", REPORT_COLOR_PALETTE["good"]),
        ("traditional_plus_proxy", "traditional_plus_proxy_predicted_pd", REPORT_COLOR_PALETTE["highlight"]),
    ]
    metrics_lookup = {row["feature_set"]: row for row in summary["comparison_metrics"]}

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    roc_series: list[dict[str, object]] = []
    for feature_set, column, color in feature_specs:
        fpr, tpr, _ = roc_curve(y_true, comparison[column])
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(fpr, tpr, color=color, linewidth=2.4)
        roc_series.append({"label": display_name, "x": fpr, "y": tpr, "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 3 ROC Comparison")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _add_direct_line_labels(ax, roc_series, anchor_fraction=0.76, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(roc_series, relative_tolerance=0.04)
    note_parts = [f"Proxy uplift delta: {summary['data_uplift_summary'][0]['roc_auc_delta']:.4f}"]
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        note_parts.append(overlap_note)
    add_conclusion_annotation(ax, "\n".join(note_parts))
    ax.text(0.60, 0.08, "Grey dashed = random baseline", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_roc_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    pr_series: list[dict[str, object]] = []
    for feature_set, column, color in feature_specs:
        precision, recall, _ = precision_recall_curve(y_true, comparison[column])
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(recall, precision, color=color, linewidth=2.4)
        pr_series.append({"label": display_name, "x": recall, "y": precision, "color": color})
    ax.set_title("Phase 3 Precision-Recall Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _add_direct_line_labels(ax, pr_series, anchor_fraction=0.72, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(pr_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    note_parts = [f"AP delta: {summary['data_uplift_summary'][0]['average_precision_delta']:.4f}"]
    if overlap_note:
        note_parts.append(overlap_note)
    add_conclusion_annotation(ax, "\n".join(note_parts))
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_pr_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ks_series: list[dict[str, object]] = []
    for feature_set, column, color in feature_specs:
        ks_frame = _compute_ks_frame(y_true, comparison[column])
        best_row = ks_frame.loc[ks_frame["ks"].idxmax()]
        finite_frame = ks_frame.loc[np.isfinite(ks_frame["threshold"])].sort_values("threshold")
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(finite_frame["threshold"], finite_frame["ks"], color=color, linewidth=2.2)
        ax.axvline(best_row["threshold"], linestyle=":", color=color, linewidth=1, alpha=0.45)
        ks_series.append({"label": display_name, "x": finite_frame["threshold"], "y": finite_frame["ks"], "color": color})
    ax.set_title("Phase 3 KS by Score Threshold")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("KS statistic")
    _add_direct_line_labels(ax, ks_series, anchor_fraction=0.76, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(ks_series, relative_tolerance=0.05)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.56, 0.08, "Light dotted lines = best threshold markers", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_ks_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    calibration_series: list[dict[str, object]] = []
    for feature_set, _, color in feature_specs:
        frame = calibration.loc[calibration["feature_set"] == feature_set].sort_values("bin_index")
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(frame["predicted_mean"], frame["observed_rate"], marker="o", linewidth=2.1, color=color)
        calibration_series.append({"label": display_name, "x": frame["predicted_mean"], "y": frame["observed_rate"], "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 3 Calibration Comparison")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    format_percent_axis(ax, axis="both", decimals=0)
    _add_direct_line_labels(ax, calibration_series, anchor_fraction=0.68, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(calibration_series, relative_tolerance=0.06)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.56, 0.08, "Grey dashed = perfect calibration", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_calibration_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    gain_series: list[dict[str, object]] = []
    for feature_set, _, color in feature_specs:
        frame = gain.loc[gain["feature_set"] == feature_set].sort_values("population_share")
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(frame["population_share"], frame["captured_bad_share"], linewidth=2.2, color=color)
        gain_series.append({"label": display_name, "x": frame["population_share"], "y": frame["captured_bad_share"], "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 3 Gain Curve")
    ax.set_xlabel("Reviewed population share")
    ax.set_ylabel("Captured bad share")
    format_percent_axis(ax, axis="both", decimals=0)
    _add_direct_line_labels(ax, gain_series, anchor_fraction=0.76, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(gain_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.52, 0.08, "Grey dashed = random review baseline", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_gain_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    lift_series: list[dict[str, object]] = []
    for feature_set, _, color in feature_specs:
        frame = lift.loc[lift["feature_set"] == feature_set].sort_values("population_share")
        plot_frame = frame.loc[frame["population_share"] > 0].copy()
        display_name = PHASE3_DISPLAY_NAMES[feature_set]
        ax.plot(plot_frame["population_share"], plot_frame["lift"], linewidth=2.2, color=color)
        lift_series.append({"label": display_name, "x": plot_frame["population_share"], "y": plot_frame["lift"], "color": color})
    ax.axhline(1.0, linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 3 Lift Curve")
    ax.set_xlabel("Reviewed population share")
    ax.set_ylabel("Lift")
    format_percent_axis(ax, axis="x", decimals=0)
    _add_direct_line_labels(ax, lift_series, anchor_fraction=0.76, min_gap_fraction=0.08)
    overlap_pairs = _find_near_overlap_pairs(lift_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.64, 0.08, "Grey dashed = baseline lift", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase3_logistic_feature_set_lift_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )


def _render_phase4_figures(paths, label_audit: dict[str, dict[str, object]]) -> None:
    output_dir = paths.reports_figures_v2
    _setup_plot_style()

    summary = _load_json(paths.data_processed / "modeling_advanced" / "summary.json")
    comparison = _load_csv(paths.data_processed / "modeling_advanced" / "validation_score_comparison.csv")
    calibration = _load_csv(paths.data_processed / "modeling_advanced" / "calibration_curve_points.csv")
    gain = _load_csv(paths.data_processed / "modeling_advanced" / "gain_curve_points.csv")
    lift = _load_csv(paths.data_processed / "modeling_advanced" / "lift_curve_points.csv")
    core_importance = _load_csv(paths.data_processed / "modeling_advanced" / "traditional_core" / "feature_importance.csv")
    proxy_importance = _load_csv(paths.data_processed / "modeling_advanced" / "traditional_plus_proxy" / "feature_importance.csv")
    y_true = comparison["TARGET"]
    model_specs = [
        ("logistic_traditional_core", "traditional_core_logistic_predicted_pd", REPORT_COLOR_PALETTE["good"], REPORT_LINESTYLES["baseline"]),
        ("xgboost_traditional_core", "traditional_core_advanced_predicted_pd", REPORT_COLOR_PALETTE["good"], REPORT_LINESTYLES["best"]),
        ("logistic_traditional_plus_proxy", "traditional_plus_proxy_logistic_predicted_pd", REPORT_COLOR_PALETTE["highlight"], REPORT_LINESTYLES["baseline"]),
        ("xgboost_traditional_plus_proxy", "traditional_plus_proxy_advanced_predicted_pd", REPORT_COLOR_PALETTE["highlight"], REPORT_LINESTYLES["best"]),
    ]
    metrics_lookup = {row["model_label"]: row for row in summary["comparison_metrics"]}

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    roc_series: list[dict[str, object]] = []
    for model_label, column, color, linestyle in model_specs:
        fpr, tpr, _ = roc_curve(y_true, comparison[column])
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2.2 if "xgboost" in model_label else 1.8)
        roc_series.append({"label": display_name, "x": fpr, "y": tpr, "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 4 ROC Comparison")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _add_direct_line_labels(ax, roc_series, anchor_fraction=0.74, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(roc_series, relative_tolerance=0.04)
    note_parts = [f"Best candidate: {PHASE4_DISPLAY_NAMES.get(summary['best_candidate']['model_label'], summary['best_candidate']['model_label'])}"]
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        note_parts.append(overlap_note)
    add_conclusion_annotation(ax, "\n".join(note_parts))
    ax.text(0.60, 0.08, "Grey dashed = random baseline", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase4_four_model_roc_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    pr_series: list[dict[str, object]] = []
    for model_label, column, color, linestyle in model_specs:
        precision, recall, _ = precision_recall_curve(y_true, comparison[column])
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=2.2 if "xgboost" in model_label else 1.8)
        pr_series.append({"label": display_name, "x": recall, "y": precision, "color": color})
    ax.set_title("Phase 4 Precision-Recall Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _add_direct_line_labels(ax, pr_series, anchor_fraction=0.70, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(pr_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    _save_figure(
        fig,
        output_dir / "phase4_four_model_pr_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ks_series: list[dict[str, object]] = []
    for model_label, column, color, linestyle in model_specs:
        ks_frame = _compute_ks_frame(y_true, comparison[column])
        best_row = ks_frame.loc[ks_frame["ks"].idxmax()]
        finite_frame = ks_frame.loc[np.isfinite(ks_frame["threshold"])].sort_values("threshold")
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(finite_frame["threshold"], finite_frame["ks"], color=color, linestyle=linestyle, linewidth=2.2 if "xgboost" in model_label else 1.8)
        ks_series.append({"label": display_name, "x": finite_frame["threshold"], "y": finite_frame["ks"], "color": color})
    ax.set_title("Phase 4 KS Comparison")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("KS statistic")
    _add_direct_line_labels(ax, ks_series, anchor_fraction=0.76, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(ks_series, relative_tolerance=0.05)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    _save_figure(
        fig,
        output_dir / "phase4_four_model_ks_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    calibration_series: list[dict[str, object]] = []
    for model_label, _, color, linestyle in model_specs:
        frame = calibration.loc[calibration["model_label"] == model_label].sort_values("bin_index")
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(frame["predicted_mean"], frame["observed_rate"], marker="o", color=color, linestyle=linestyle, linewidth=2.1 if "xgboost" in model_label else 1.7)
        calibration_series.append({"label": display_name, "x": frame["predicted_mean"], "y": frame["observed_rate"], "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 4 Calibration Comparison")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    format_percent_axis(ax, axis="both", decimals=0)
    _add_direct_line_labels(ax, calibration_series, anchor_fraction=0.66, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(calibration_series, relative_tolerance=0.06)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.56, 0.08, "Grey dashed = perfect calibration", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase4_four_model_calibration_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    gain_series: list[dict[str, object]] = []
    for model_label, _, color, linestyle in model_specs:
        frame = gain.loc[gain["model_label"] == model_label].sort_values("population_share")
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(frame["population_share"], frame["captured_bad_share"], color=color, linestyle=linestyle, linewidth=2.1 if "xgboost" in model_label else 1.7)
        gain_series.append({"label": display_name, "x": frame["population_share"], "y": frame["captured_bad_share"], "color": color})
    ax.plot([0, 1], [0, 1], linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 4 Gain Curve")
    ax.set_xlabel("Reviewed population share")
    ax.set_ylabel("Captured bad share")
    format_percent_axis(ax, axis="both", decimals=0)
    _add_direct_line_labels(ax, gain_series, anchor_fraction=0.76, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(gain_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.52, 0.08, "Grey dashed = random review baseline", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase4_four_model_gain_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    lift_series: list[dict[str, object]] = []
    for model_label, _, color, linestyle in model_specs:
        frame = lift.loc[lift["model_label"] == model_label].sort_values("population_share")
        plot_frame = frame.loc[frame["population_share"] > 0].copy()
        display_name = PHASE4_DISPLAY_NAMES[model_label]
        ax.plot(plot_frame["population_share"], plot_frame["lift"], color=color, linestyle=linestyle, linewidth=2.1 if "xgboost" in model_label else 1.7)
        lift_series.append({"label": display_name, "x": plot_frame["population_share"], "y": plot_frame["lift"], "color": color})
    ax.axhline(1.0, linestyle="--", color=REPORT_COLOR_PALETTE["neutral"], linewidth=1, alpha=0.9)
    ax.set_title("Phase 4 Lift Curve")
    ax.set_xlabel("Reviewed population share")
    ax.set_ylabel("Lift")
    format_percent_axis(ax, axis="x", decimals=0)
    _add_direct_line_labels(ax, lift_series, anchor_fraction=0.76, min_gap_fraction=0.07)
    overlap_pairs = _find_near_overlap_pairs(lift_series, relative_tolerance=0.04)
    overlap_note = _format_overlap_note(overlap_pairs)
    if overlap_note:
        add_conclusion_annotation(ax, overlap_note)
    ax.text(0.64, 0.08, "Grey dashed = baseline lift", transform=ax.transAxes, fontsize=8, color=REPORT_COLOR_PALETTE["neutral"])
    _save_figure(
        fig,
        output_dir / "phase4_four_model_lift_comparison.png",
        label_audit=label_audit,
        identifier_mode="direct_label",
        overlap_warning=bool(overlap_pairs),
        note=overlap_note,
    )

    for frame, file_name, title in [
        (core_importance, "phase4_traditional_core_feature_importance.png", "Phase 4 Native Feature Importance: traditional_core"),
        (proxy_importance, "phase4_traditional_plus_proxy_feature_importance.png", "Phase 4 Native Feature Importance: traditional_plus_proxy"),
    ]:
        plot_frame = frame.head(15).sort_values("importance", ascending=True).copy()
        plot_frame["feature_name"] = plot_frame["feature_name"].str.replace("__", " / ", regex=False)
        fig, ax = plt.subplots(figsize=(10.0, 6.6))
        sns.barplot(data=plot_frame, x="importance", y="feature_name", color=REPORT_COLOR_PALETTE["good"], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        wrap_tick_labels(ax, axis="y", width=28)
        annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.0015)
        add_conclusion_annotation(ax, "Native model importance helps ranking, but it is not SHAP.")
        _save_figure(fig, output_dir / file_name, label_audit=label_audit)


def _render_phase5_figures(paths, label_audit: dict[str, dict[str, object]]) -> None:
    output_dir = paths.reports_figures_v2
    _setup_plot_style()

    fairness_metric = _load_csv(paths.data_processed / "xai_fairness" / "fairness_metric_summary.csv")
    group_fairness = _load_csv(paths.data_processed / "xai_fairness" / "group_fairness_summary.csv")
    proxy_uplift = _load_csv(paths.data_processed / "xai_fairness" / "proxy_uplift_summary.csv")
    validation_review = _load_csv(paths.data_processed / "xai_fairness" / "validation_review_frame.csv")
    top_interactions = _load_csv(paths.data_processed / "xai_fairness" / "top_shap_interactions.csv")
    candidate_pdp = _load_csv(paths.data_processed / "xai_fairness" / "candidate_partial_dependence.csv")

    disparity = fairness_metric[["protected_attribute", "demographic_parity_diff", "equal_opportunity_diff", "equalized_odds_gap"]].melt(
        id_vars="protected_attribute",
        var_name="metric",
        value_name="gap",
    )
    disparity["metric"] = disparity["metric"].map(
        {
            "demographic_parity_diff": "Demographic parity",
            "equal_opportunity_diff": "Equal opportunity",
            "equalized_odds_gap": "Equalized odds",
        }
    ).fillna(disparity["metric"])
    disparity["protected_attribute"] = pd.Categorical(
        disparity["protected_attribute"],
        categories=fairness_metric.sort_values("equalized_odds_gap", ascending=False)["protected_attribute"].tolist(),
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    sns.barplot(data=disparity.sort_values("protected_attribute"), x="gap", y="protected_attribute", hue="metric", ax=ax)
    ax.set_title("Phase 5 Fairness Metric Gaps")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Protected attribute")
    format_percent_axis(ax, axis="x", decimals=0)
    place_legend_inside(ax, location="lower right", title="Metric")
    add_conclusion_annotation(ax, "Larger gaps indicate stronger governance review needs.")
    _save_figure(
        fig,
        output_dir / "phase5_fairness_metric_gaps.png",
        label_audit=label_audit,
        identifier_mode="legend_inside",
    )

    for protected_attribute, metric_column, file_name in PHASE5_GROUP_METRIC_FILES:
        plot_frame = group_fairness.loc[group_fairness["protected_attribute"] == protected_attribute].copy()
        if plot_frame.empty or metric_column not in plot_frame.columns:
            continue
        plot_frame = plot_frame.sort_values(metric_column, ascending=True)
        plot_frame["label"] = plot_frame["group"].astype(str) + " (n=" + plot_frame["count"].map(lambda v: f"{int(v):,}") + ")"
        fig, ax = plt.subplots(figsize=(10.2, max(4.6, 0.5 * len(plot_frame))))
        sns.barplot(data=plot_frame, x=metric_column, y="label", color=REPORT_COLOR_PALETTE["good"], ax=ax)
        ax.set_title(file_name.replace("phase5_", "").replace(".png", "").replace("_", " ").title())
        ax.set_xlabel(metric_column.replace("_", " "))
        ax.set_ylabel(protected_attribute)
        if metric_column != "count":
            format_percent_axis(ax, axis="x", decimals=0)
            annotate_bar_values(ax, orientation="horizontal", value_format="{:.1%}", padding=0.003)
        add_conclusion_annotation(ax, "Support matters when reading grouped operational gaps.")
        _save_figure(fig, output_dir / file_name, label_audit=label_audit)

    proxy_plot = proxy_uplift.sort_values("mean_abs_contribution_delta_proxy_minus_core", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    colors = [REPORT_COLOR_PALETTE["bad"] if value < 0 else REPORT_COLOR_PALETTE["good"] for value in proxy_plot["mean_abs_contribution_delta_proxy_minus_core"]]
    sns.barplot(data=proxy_plot, x="mean_abs_contribution_delta_proxy_minus_core", y="proxy_family", palette=colors, ax=ax)
    ax.set_title("Phase 5 Proxy Family Uplift Delta")
    ax.set_xlabel("Mean abs contribution delta: proxy minus core")
    ax.set_ylabel("Proxy family")
    annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.01)
    add_conclusion_annotation(ax, "EXT_SOURCE dominates proxy uplift in the current candidate model.")
    _save_figure(fig, output_dir / "phase5_proxy_family_uplift_delta.png", label_audit=label_audit)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    sns.histplot(validation_review["proxy_minus_core_predicted_pd_delta"], bins=40, color=REPORT_COLOR_PALETTE["good"], ax=ax)
    ax.axvline(0.0, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
    ax.set_title("Phase 5 Proxy Minus Core PD Delta")
    ax.set_xlabel("Proxy minus core predicted PD delta")
    ax.set_ylabel("Count")
    add_conclusion_annotation(ax, "Right tail = proxy regime assigns meaningfully higher risk.")
    _save_figure(fig, output_dir / "phase5_proxy_minus_core_score_delta_histogram.png", label_audit=label_audit)

    for feature_set in ("traditional_core", "traditional_plus_proxy"):
        global_contrib = _load_csv(paths.data_processed / "xai_fairness" / feature_set / "global_feature_contributions.csv")
        plot_frame = global_contrib.head(15).sort_values("mean_abs_contribution", ascending=True).copy()
        plot_frame["label"] = plot_frame["raw_feature_name"].fillna(plot_frame["transformed_feature_name"])
        fig, ax = plt.subplots(figsize=(10.2, 6.6))
        sns.barplot(data=plot_frame, x="mean_abs_contribution", y="label", color=REPORT_COLOR_PALETTE["good"], ax=ax)
        ax.set_title(f"Phase 5 SHAP Mean Absolute Value ({feature_set})")
        ax.set_xlabel("Mean abs contribution")
        ax.set_ylabel("Feature")
        wrap_tick_labels(ax, axis="y", width=28)
        annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.01)
        _save_figure(fig, output_dir / f"phase5_{feature_set}_shap_bar.png", label_audit=label_audit)

        proxy_family = _load_csv(paths.data_processed / "xai_fairness" / feature_set / "proxy_family_contributions.csv")
        if feature_set == "traditional_core" and len(proxy_family) == 1 and proxy_family["proxy_family"].iloc[0] == "traditional_non_proxy":
            fig, ax = plt.subplots(figsize=(8.8, 4.6))
            ax.barh(["proxy excluded by regime"], [1.0], color=REPORT_COLOR_PALETTE["neutral"])
            ax.set_title("Phase 5 Proxy Family Contributions (traditional_core)")
            ax.set_xlabel("Interpretation")
            ax.set_ylabel("")
            add_conclusion_annotation(ax, "This regime excludes proxy families by design.")
            _save_figure(fig, output_dir / "phase5_traditional_core_proxy_family_contributions.png", label_audit=label_audit)
        else:
            proxy_plot = proxy_family.head(8).sort_values("mean_abs_contribution", ascending=True).copy()
            fig, ax = plt.subplots(figsize=(9.6, 5.8))
            sns.barplot(data=proxy_plot, x="mean_abs_contribution", y="proxy_family", color=REPORT_COLOR_PALETTE["highlight"], ax=ax)
            ax.set_title(f"Phase 5 Proxy Family Contributions ({feature_set})")
            ax.set_xlabel("Mean abs contribution")
            ax.set_ylabel("Proxy family")
            annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.01)
            _save_figure(fig, output_dir / f"phase5_{feature_set}_proxy_family_contributions.png", label_audit=label_audit)

        local_shap = _load_csv(paths.data_processed / "xai_fairness" / feature_set / "local_case_explanations.csv")
        local_lime = _load_csv(paths.data_processed / "xai_fairness" / feature_set / "lime_local_case_explanations.csv")
        for case_number, case_role in enumerate(local_shap["case_role"].drop_duplicates().tolist(), start=1):
            shap_case = local_shap.loc[local_shap["case_role"] == case_role].sort_values("feature_rank").head(8).copy()
            shap_case = shap_case.sort_values("contribution")
            fig, ax = plt.subplots(figsize=(9.6, 5.4))
            colors = [REPORT_COLOR_PALETTE["bad"] if value > 0 else REPORT_COLOR_PALETTE["good"] for value in shap_case["contribution"]]
            sns.barplot(data=shap_case, x="contribution", y="raw_feature_name", palette=colors, ax=ax)
            ax.axvline(0.0, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
            ax.set_title(f"Phase 5 SHAP Local Explanation #{case_number} ({feature_set})")
            ax.set_xlabel("Contribution")
            ax.set_ylabel("Feature")
            annotate_bar_values(ax, orientation="horizontal", value_format="{:.2f}", padding=0.02)
            _save_figure(fig, output_dir / f"phase5_{feature_set}_case_{case_number}_shap_waterfall.png", label_audit=label_audit)

            lime_case = local_lime.loc[local_lime["case_role"] == case_role].sort_values("feature_rank").head(8).copy()
            lime_case = lime_case.sort_values("local_weight")
            fig, ax = plt.subplots(figsize=(9.6, 5.4))
            colors = [REPORT_COLOR_PALETTE["bad"] if value > 0 else REPORT_COLOR_PALETTE["good"] for value in lime_case["local_weight"]]
            sns.barplot(data=lime_case, x="local_weight", y="raw_feature_name", palette=colors, ax=ax)
            ax.axvline(0.0, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
            ax.set_title(f"Phase 5 LIME Local Explanation #{case_number} ({feature_set})")
            ax.set_xlabel("Local weight")
            ax.set_ylabel("Feature")
            annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.002)
            _save_figure(fig, output_dir / f"phase5_{feature_set}_case_{case_number}_lime.png", label_audit=label_audit)

    for feature_rank in sorted(candidate_pdp["feature_rank"].dropna().unique()):
        plot_frame = candidate_pdp.loc[candidate_pdp["feature_rank"] == feature_rank].copy()
        feature_name = plot_frame["feature"].iloc[0].replace("numeric__", "")
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(plot_frame["feature_value"], plot_frame["partial_dependence"], color=REPORT_COLOR_PALETTE["good"], linewidth=2.2)
        ax.set_title(f"Phase 5 Partial Dependence: {feature_name}")
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Average predicted response")
        add_conclusion_annotation(ax, f"Primary numeric driver #{int(feature_rank)}")
        _save_figure(fig, output_dir / f"phase5_traditional_plus_proxy_pdp_{int(feature_rank)}.png", label_audit=label_audit)

    interaction_plot = top_interactions.sort_values("interaction_strength", ascending=True).copy()
    interaction_plot["label"] = interaction_plot["left_raw_feature"] + " x " + interaction_plot["right_raw_feature"]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    sns.barplot(data=interaction_plot, x="interaction_strength", y="label", color=REPORT_COLOR_PALETTE["accent"], ax=ax)
    ax.set_title("Phase 5 Top Interaction Pairs")
    ax.set_xlabel("Approximate interaction strength")
    ax.set_ylabel("Feature pair")
    annotate_bar_values(ax, orientation="horizontal", value_format="{:.3f}", padding=0.003)
    _save_figure(fig, output_dir / "phase5_traditional_plus_proxy_top_interactions.png", label_audit=label_audit)


def _build_heatmap_annotations(pivot: pd.DataFrame) -> np.ndarray:
    row_totals = pivot.sum(axis=1).replace({0: np.nan})
    shares = pivot.div(row_totals, axis=0).fillna(0.0)
    annotations = np.empty_like(pivot.astype(str).to_numpy(), dtype=object)
    for row_index, row_label in enumerate(pivot.index):
        for column_index, column_label in enumerate(pivot.columns):
            count = int(pivot.loc[row_label, column_label])
            share = shares.loc[row_label, column_label]
            annotations[row_index, column_index] = f"{count:,}\n{share:.1%}"
    return annotations


def _render_phase6_figures(paths, label_audit: dict[str, dict[str, object]]) -> None:
    output_dir = paths.reports_figures_v2
    _setup_plot_style()

    phase6_candidates = sorted((paths.data_processed / "scorecard_cutoff").glob("*/"))
    if not phase6_candidates:
        raise FileNotFoundError("No scorecard_cutoff model directory found.")
    phase6_dir = phase6_candidates[0]
    score_frame = _load_csv(phase6_dir / "score_frame.csv")
    score_decile = _load_csv(phase6_dir / "score_decile_summary.csv")
    risk_band = _load_csv(phase6_dir / "risk_band_summary.csv")
    calibration = _load_csv(phase6_dir / "calibration_summary.csv")
    cutoff = _load_csv(phase6_dir / "cutoff_sweep.csv")
    final_policy_group = _load_csv(phase6_dir / "final_policy_group_summary.csv")
    score_migration = _load_csv(phase6_dir / "score_migration_matrix.csv")
    decision_migration = _load_csv(phase6_dir / "decision_migration_matrix.csv")
    score_transform_meta = _load_json(phase6_dir / "score_transform_meta.json")
    summary = _load_json(phase6_dir / "summary.json")
    final_policy_path = phase6_dir / "final_policy_summary.json"
    if final_policy_path.exists():
        final_policy = _load_json(final_policy_path)
    else:
        final_policy = {}
    candidate_label = phase6_dir.name
    calibration_rate_column = (
        "mean_calibrated_pd"
        if "mean_calibrated_pd" in calibration.columns
        else "mean_predicted_pd"
    )
    decile_rate_column = (
        "mean_calibrated_pd"
        if "mean_calibrated_pd" in score_decile.columns
        else "mean_predicted_pd"
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.histplot(score_frame["candidate_score"], bins=35, kde=True, color=REPORT_COLOR_PALETTE["good"], ax=ax)
    for stat_name, stat_value in {
        "mean": score_frame["candidate_score"].mean(),
        "median": score_frame["candidate_score"].median(),
        "p95": score_frame["candidate_score"].quantile(0.95),
    }.items():
        ax.axvline(stat_value, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
    ax.set_title(f"Phase 6 Score Distribution ({candidate_label})")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    add_conclusion_annotation(ax, f"Median score: {score_frame['candidate_score'].median():.1f}")
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_score_histogram_kde.png", label_audit=label_audit)

    score_sorted = score_frame["candidate_score"].sort_values().reset_index(drop=True)
    ecdf = pd.Series(np.arange(1, len(score_sorted) + 1) / len(score_sorted))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(score_sorted, ecdf, color=REPORT_COLOR_PALETTE["good"], linewidth=2.2)
    ax.set_title(f"Phase 6 Score ECDF ({candidate_label})")
    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative share")
    format_percent_axis(ax, axis="y", decimals=0)
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_score_ecdf.png", label_audit=label_audit)

    pd_grid = np.linspace(0.01, 0.99, 300)
    score_grid = score_transform_meta["base_score"] + score_transform_meta["factor"] * np.log(((1.0 - pd_grid) / pd_grid) / score_transform_meta["base_odds"])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(pd_grid, score_grid, color=REPORT_COLOR_PALETTE["bad"], linewidth=2.2)
    ax.set_title("Phase 6 PD-to-Score Curve")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Score")
    format_percent_axis(ax, axis="x", decimals=0)
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_pd_to_score_curve.png", label_audit=label_audit)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(calibration[calibration_rate_column], calibration["actual_default_rate"], marker="o", color=REPORT_COLOR_PALETTE["good"], linewidth=2.2, label="Observed by bin")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"], label="Perfect")
    ax.set_title(f"Phase 6 Calibration Curve ({candidate_label})")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Actual default rate")
    format_percent_axis(ax, axis="both", decimals=0)
    place_legend_inside(ax, location="lower right", title="Curve")
    worst_gap = calibration.loc[calibration["calibration_gap"].abs().idxmax()]
    add_conclusion_annotation(ax, f"Max bin gap: {abs(worst_gap['calibration_gap']):.1%}")
    _save_figure(
        fig,
        output_dir / f"phase6_{candidate_label}_calibration_curve.png",
        label_audit=label_audit,
        identifier_mode="legend_inside",
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    x_labels = score_decile["risk_decile"].astype(str)
    ax.bar(x_labels, score_decile[decile_rate_column], color=REPORT_COLOR_PALETTE["good"], alpha=0.8, label="Mean calibrated PD")
    ax.plot(x_labels, score_decile["actual_default_rate"], color=REPORT_COLOR_PALETTE["bad"], marker="o", linewidth=2.0, label="Actual default rate")
    ax.set_title(f"Phase 6 Decile Reliability ({candidate_label})")
    ax.set_xlabel("Risk decile (1 = highest risk)")
    ax.set_ylabel("Rate")
    format_percent_axis(ax, axis="y", decimals=0)
    place_legend_inside(ax, location="upper right", title="Series")
    _save_figure(
        fig,
        output_dir / f"phase6_{candidate_label}_decile_reliability.png",
        label_audit=label_audit,
        identifier_mode="legend_inside",
    )

    for metric_column, file_name, title, color in [
        ("count", f"phase6_{candidate_label}_risk_band_count.png", f"Phase 6 Risk Band Distribution ({candidate_label})", REPORT_COLOR_PALETTE["good"]),
        ("actual_default_rate", f"phase6_{candidate_label}_risk_band_actual_default_rate.png", f"Phase 6 Actual Default Rate by Risk Band ({candidate_label})", REPORT_COLOR_PALETTE["bad"]),
        ("mean_calibrated_pd", f"phase6_{candidate_label}_risk_band_mean_calibrated_pd.png", f"Phase 6 Mean Calibrated PD by Risk Band ({candidate_label})", REPORT_COLOR_PALETTE["accent"]),
    ]:
        plot_frame = risk_band.copy()
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        sns.barplot(data=plot_frame, x="risk_band", y=metric_column, color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Risk band")
        ax.set_ylabel(metric_column.replace("_", " "))
        if metric_column != "count":
            format_percent_axis(ax, axis="y", decimals=0)
            annotate_bar_values(ax, value_format="{:.1%}", padding=0.003)
        else:
            annotate_bar_values(ax, value_format="{:.0f}", padding=max(plot_frame[metric_column].max() * 0.01, 1))
            dominant_share = plot_frame["population_share"].max()
            if dominant_share >= 0.8:
                add_conclusion_annotation(ax, f"Concentration warning: top band share = {dominant_share:.1%}")
        _save_figure(fig, output_dir / file_name, label_audit=label_audit)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(
        cutoff["score_cutoff"],
        cutoff["expected_value_per_applicant"],
        marker="o",
        color=REPORT_COLOR_PALETTE["good"],
        linewidth=2.2,
        label="Expected value / applicant",
    )
    if "actual_approved_bad_rate" in cutoff.columns:
        ax2 = ax.twinx()
        ax2.plot(
            cutoff["score_cutoff"],
            cutoff["actual_approved_bad_rate"],
            color=REPORT_COLOR_PALETTE["bad"],
            linewidth=1.8,
            linestyle="--",
            label="Approved bad rate",
        )
        ax2.set_ylabel("Approved bad rate")
        format_percent_axis(ax2, axis="y", decimals=0)
    else:
        ax2 = None
    final_cutoff = final_policy.get("final_cutoff")
    if final_cutoff is not None:
        matched_curve = cutoff.loc[cutoff["approve_min_score"] == float(final_cutoff)]
        if matched_curve.empty:
            closest_index = int((cutoff["score_cutoff"] - float(final_cutoff)).abs().idxmin())
            matched_curve = cutoff.iloc[[closest_index]]
        if not matched_curve.empty:
            best_row = matched_curve.iloc[0]
            ax.scatter(
                [best_row["score_cutoff"]],
                [best_row["expected_value_per_applicant"]],
                color=REPORT_COLOR_PALETTE["highlight"],
                s=55,
                label="Final cutoff",
            )
            add_conclusion_annotation(
                ax,
                f"Final approve cutoff: {int(final_cutoff)}, review cutoff: {int(final_policy.get('final_review_cutoff', 0))}",
            )
    ax.set_title(f"Phase 6 Final Policy Cutoff Curve ({candidate_label})")
    ax.set_xlabel("Cutoff anchor score")
    ax.set_ylabel("Expected value per applicant")
    place_legend_inside(ax, location="best", title="Series")
    _save_figure(
        fig,
        output_dir / f"phase6_{candidate_label}_final_policy_cutoff_curve.png",
        label_audit=label_audit,
        identifier_mode="legend_inside",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.histplot(score_frame["candidate_score_minus_comparator"], bins=35, color=REPORT_COLOR_PALETTE["good"], ax=ax)
    ax.axvline(0.0, linestyle="--", linewidth=1, color=REPORT_COLOR_PALETTE["neutral"])
    ax.set_title(f"Phase 6 Candidate Minus Comparator Score Delta ({candidate_label})")
    ax.set_xlabel("Score delta")
    ax.set_ylabel("Count")
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_score_delta_histogram.png", label_audit=label_audit)

    score_pivot = score_migration.pivot(index="comparator_risk_band", columns="candidate_risk_band", values="count").fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(score_pivot, annot=_build_heatmap_annotations(score_pivot), fmt="", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"Phase 6 Risk Band Migration ({candidate_label})")
    ax.set_xlabel("Candidate risk band")
    ax.set_ylabel("Comparator risk band")
    _set_heatmap_colorbar_label(ax, "Count")
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_band_migration_heatmap.png", label_audit=label_audit)

    decision_pivot = decision_migration.pivot(index="comparator_final_decision", columns="candidate_final_decision", values="count").fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(decision_pivot, annot=_build_heatmap_annotations(decision_pivot), fmt="", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"Phase 6 Final Decision Migration ({candidate_label})")
    ax.set_xlabel("Candidate decision")
    ax.set_ylabel("Comparator decision")
    _set_heatmap_colorbar_label(ax, "Count")
    _save_figure(fig, output_dir / f"phase6_{candidate_label}_decision_migration_heatmap.png", label_audit=label_audit)

    for protected_attribute, figure_key in PHASE6_GROUP_POLICY_FILES:
        plot_frame = final_policy_group.loc[
            final_policy_group["protected_attribute"] == protected_attribute
        ].copy()
        if plot_frame.empty:
            continue
        plot_frame = plot_frame.sort_values("approval_rate", ascending=True)
        plot_frame["label"] = plot_frame["group"].astype(str) + " (n=" + plot_frame["count"].map(lambda v: f"{int(v):,}") + ")"
        fig, ax = plt.subplots(figsize=(10.6, max(4.8, 0.55 * len(plot_frame))))
        left = np.zeros(len(plot_frame))
        for column, color, label in [
            ("approval_rate", REPORT_COLOR_PALETTE["good"], "approve"),
            ("review_rate", REPORT_COLOR_PALETTE["highlight"], "review"),
            ("reject_rate", REPORT_COLOR_PALETTE["bad"], "reject"),
        ]:
            ax.barh(plot_frame["label"], plot_frame[column], left=left, color=color, label=label)
            left = left + plot_frame[column].to_numpy()
        ax.set_title(f"Phase 6 {protected_attribute.replace('_', ' ').title()} Final Policy")
        ax.set_xlabel("Population share within group")
        ax.set_ylabel(protected_attribute)
        format_percent_axis(ax, axis="x", decimals=0)
        place_legend_inside(ax, location="lower right", title="Action")
        _save_figure(
            fig,
            output_dir / f"phase6_{candidate_label}_{figure_key}.png",
            label_audit=label_audit,
            identifier_mode="legend_inside",
        )


def _build_contact_sheet(paths, phase_prefix: str) -> Path | None:
    files = sorted(paths.reports_figures_v2.glob(f"{phase_prefix}_*.png"))
    if not files:
        return None

    output_dir = _safe_mkdir(paths.reports_figures_v2 / "contact_sheets")
    columns = 4
    rows = math.ceil(len(files) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 2.7))
    axes_array = np.atleast_1d(axes).flatten()
    for ax, file_path in zip(axes_array, files):
        image = mpimg.imread(file_path)
        ax.imshow(image)
        ax.set_title(file_path.stem.replace(f"{phase_prefix}_", ""), fontsize=7)
        ax.axis("off")
    for ax in axes_array[len(files):]:
        ax.axis("off")
    fig.suptitle(f"{phase_prefix} figure overview", y=0.995, fontsize=13, fontweight="bold")
    output_path = output_dir / f"{phase_prefix}_overview.png"
    _save_figure(fig, output_path)
    return output_path


def _update_phase_summaries(paths, label_audit: dict[str, dict[str, object]]) -> None:
    phase_paths = {
        phase_name: paths.data_processed.joinpath(*parts)
        for phase_name, parts in LEGACY_PHASE_SUMMARY_PATHS.items()
    }
    phase6_summaries = sorted((paths.data_processed / "scorecard_cutoff").glob("*/summary.json"))
    if phase6_summaries:
        phase_paths["phase6"] = phase6_summaries[0]

    for phase_name, summary_path in phase_paths.items():
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        original_manifest = payload.get("figure_manifest_original")
        if not isinstance(original_manifest, dict) or not original_manifest:
            if isinstance(payload.get("figure_manifest"), dict):
                original_manifest = payload["figure_manifest"]
            else:
                original_manifest = _build_phase_manifest(paths.reports_figures, phase_name)

        repaired_manifest = _build_phase_manifest(paths.reports_figures_v2, phase_name)
        payload.update(
            build_figure_quality_fields(
                original_paths=original_manifest,
                repaired_paths=repaired_manifest,
                quality_status=FIGURE_QUALITY_STATUS.get(phase_name, "repaired_v2"),
            )
        )
        payload["figure_label_audit"] = _build_phase_label_audit(repaired_manifest, label_audit)
        _write_json(summary_path, payload)


def generate_repaired_figures() -> dict[str, list[str]]:
    """Create repaired figure outputs and update phase summaries."""

    paths = get_paths()
    label_audit: dict[str, dict[str, object]] = {}
    _copy_original_figures(paths)
    _render_phase1_core_figures(paths, label_audit)
    _render_phase3_figures(paths, label_audit)
    _render_phase4_figures(paths, label_audit)
    _render_phase5_figures(paths, label_audit)
    _render_phase6_figures(paths, label_audit)
    _update_phase_summaries(paths, label_audit)

    contact_sheets: list[str] = []
    for phase_prefix in PHASE_CONTACT_SHEET_NAMES:
        contact_sheet = _build_contact_sheet(paths, phase_prefix)
        if contact_sheet is not None:
            contact_sheets.append(str(contact_sheet))

    return {
        "figures_v2": [str(path) for path in sorted(paths.reports_figures_v2.glob("phase*.png"))],
        "contact_sheets": contact_sheets,
        "audited_figures": sorted(label_audit),
    }


if __name__ == "__main__":
    print(json.dumps(generate_repaired_figures(), indent=2, ensure_ascii=False))
