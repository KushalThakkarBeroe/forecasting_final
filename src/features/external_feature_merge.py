"""
Merges the external driver output (src/data/external_driver_loader.py —
dynamic per-driver lag already applied) into the internal feature matrix
(src/features/internal_features.py), producing the feature matrix the
external-variable-enhanced models train on: left-join on the date column,
keep the feature side.

Applies the data-adequacy gate: each driver's coverage is checked over the
merged (post-join) row range — below config/scoring.yaml's
data_adequacy.min_data_coverage_pct (default 70%), the driver is dropped
entirely; between that threshold and 100%, it's imputed (spline or
moving-average, config-selectable — see default_imputation_method). Any row
still incomplete after imputation (typically leading rows the lag shift
never filled) is dropped, same warmup-trim convention as
internal_features.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import yaml

from config_loader import PROJECT_ROOT, CommodityConfig
from external_driver_loader import DATE_COLUMN_NAME, ExternalDriverData

logger = logging.getLogger(__name__)

SCORING_YAML = PROJECT_ROOT / "config" / "scoring.yaml"


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _data_coverage_pct(series: pd.Series) -> float:
    return 100.0 * series.notna().sum() / len(series) if len(series) else 0.0


def _impute_spline(series: pd.Series, order: int) -> pd.Series:
    # forward-only: a gap must be filled using data up to and including the
    # gap, never values that come after it -- backward fill was only ever
    # defensible for the series' leading rows, which get dropped anyway.
    try:
        return series.interpolate(method="spline", order=order, limit_direction="forward")
    except Exception as exc:
        logger.warning("spline interpolation failed (%s) — falling back to linear", exc)
        return series.interpolate(method="linear", limit_direction="forward")


def _impute_moving_average(series: pd.Series, window: int) -> pd.Series:
    # center=False: the smoothing window only looks backward from each
    # point, matching _impute_spline's forward-only rule above.
    smoothed = series.rolling(window, min_periods=1, center=False).mean()
    return series.fillna(smoothed)


@dataclass
class DriverAssemblyDecision:
    label: str
    data_coverage_pct: float
    action: str  # "included" | "imputed" | "excluded"


@dataclass
class FeatureAssemblyResult:
    commodity_id: str
    df: pd.DataFrame  # internal features + adequate/imputed external driver columns
    decisions: list[DriverAssemblyDecision] = field(default_factory=list)


def gate_and_impute_drivers(
    commodity: CommodityConfig,
    external_driver_data: ExternalDriverData,
) -> tuple[pd.DataFrame, list[DriverAssemblyDecision]]:
    """
    The data-adequacy gate itself, factored out of assemble_features so it
    can run against external_driver_data.df directly — UNTRIMMED by
    internal_features.py's lag/rolling warmup dropna, which starts later
    than the drivers' own raw range and would otherwise wrongly read as
    missing exog for models that don't need internal features at all
    (models/arimax.py, models/sarimax.py — ARIMAX/SARIMAX use only price +
    drivers, never lag/rolling features, so gating them against a range
    trimmed for a use case they don't have was cutting real, available
    driver history out of their training window).

    Returns (DATE_COLUMN_NAME + adequate/imputed driver columns, decisions)
    — no dropna, no merge; the caller decides how to combine this with
    anything else.
    """
    scoring_cfg = _load_yaml(SCORING_YAML)
    adequacy_cfg = scoring_cfg["data_adequacy"]
    min_coverage_pct = adequacy_cfg["min_data_coverage_pct"]
    default_method = adequacy_cfg.get("default_imputation_method", "moving_average")
    allowed_methods = adequacy_cfg.get("imputation_methods_allowed", [])
    if default_method not in allowed_methods:
        raise ValueError(
            f"data_adequacy.default_imputation_method {default_method!r} is not in "
            f"imputation_methods_allowed {allowed_methods}"
        )

    driver_cols = [sel.label for sel in external_driver_data.lag_selections]
    if not driver_cols:
        return external_driver_data.df[[DATE_COLUMN_NAME]].copy(), []

    ext_df = external_driver_data.df[[DATE_COLUMN_NAME] + driver_cols].copy()

    decisions: list[DriverAssemblyDecision] = []
    for label in driver_cols:
        coverage = _data_coverage_pct(ext_df[label])
        if coverage < min_coverage_pct:
            ext_df = ext_df.drop(columns=[label])
            decisions.append(DriverAssemblyDecision(label, coverage, "excluded"))
        elif coverage < 100.0:
            if default_method == "spline":
                ext_df[label] = _impute_spline(ext_df[label], adequacy_cfg["imputation_spline_order"])
            else:
                ext_df[label] = _impute_moving_average(ext_df[label], adequacy_cfg["imputation_moving_average_window"])
            decisions.append(DriverAssemblyDecision(label, coverage, "imputed"))
        else:
            decisions.append(DriverAssemblyDecision(label, coverage, "included"))

    return ext_df, decisions


def assemble_features(
    commodity: CommodityConfig,
    internal_features_df: pd.DataFrame,
    external_driver_data: ExternalDriverData,
) -> FeatureAssemblyResult:
    """
    Left-joins external_driver_data's driver columns onto
    internal_features_df on DATE_COLUMN_NAME (keeps every internal-feature
    row; a commodity/horizon can still fall back to univariate models if
    every driver gets excluded here). Data-adequacy gate runs per driver
    over the drivers' OWN row range (see gate_and_impute_drivers), then
    the result is merged onto internal_features_df and trimmed.
    """
    ext_df, decisions = gate_and_impute_drivers(commodity, external_driver_data)
    if len(ext_df.columns) <= 1:
        logger.warning("%s: no driver columns to merge (Step 1 found none) — returning internal features unchanged", commodity.id)
        return FeatureAssemblyResult(commodity_id=commodity.id, df=internal_features_df.copy(), decisions=decisions)

    merged = internal_features_df.merge(ext_df, on=DATE_COLUMN_NAME, how="left")

    n_before = len(merged)
    merged = merged.dropna().reset_index(drop=True)
    n_after = len(merged)
    logger.info(
        "%s: assembled feature matrix, %d -> %d rows (dropped %d), %d columns, drivers: %s",
        commodity.id, n_before, n_after, n_before - n_after, len(merged.columns),
        {d.label: d.action for d in decisions},
    )

    return FeatureAssemblyResult(commodity_id=commodity.id, df=merged, decisions=decisions)
