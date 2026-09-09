"""
Forecast-on-forecast driver projection: forecasts each selected driver's
own future path using a lightweight univariate model (damped-trend
exponential smoothing), so the external-variable models have a driver value
to work with at every future target month a horizon needs — even where the
true future value isn't known yet. Methodology choices live in
config/driver_projection.yaml.

Why this is needed: a driver lagged by L periods (src/data/
external_driver_loader.py) only has REAL data up to (last known driver
date + L) periods into the future — forecasting further than that needs a
projected value for the driver's own future first. This module operates on
each driver's RAW (unlagged) series — ExternalDriverData.raw_df, added
specifically for this step — not the already-shifted df, so the projection
reflects the driver's own trend, not this commodity's lag baked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config_loader import PROJECT_ROOT, CommodityConfig
from external_driver_loader import DATE_COLUMN_NAME, ExternalDriverData

logger = logging.getLogger(__name__)

DRIVER_PROJECTION_YAML = PROJECT_ROOT / "config" / "driver_projection.yaml"
PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4}


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _scale_periods(month_value: int, periods_per_year: int) -> int:
    """Same convention as internal_features.py's _scale_periods — preserves
    real-world time span across frequencies (18 months always means 18
    monthly periods or 6 quarterly periods)."""
    return max(1, round(month_value * periods_per_year / 12))


def _future_dates(last_date: pd.Timestamp, n: int, periods_per_year: int) -> pd.DatetimeIndex:
    freq = "MS" if periods_per_year == 12 else "QS"
    return pd.date_range(last_date, periods=n + 1, freq=freq)[1:]


@dataclass
class DriverProjection:
    label: str
    method: str  # "ets_damped_trend" | "ets_damped_trend_seasonal" | "naive_last_value"
    projected: pd.Series  # future values, indexed by future dates


@dataclass
class DriverProjectionResult:
    commodity_id: str
    projections: list[DriverProjection] = field(default_factory=list)


def _project_one_driver(series: pd.Series, periods_per_year: int, horizon_periods: int, cfg: dict) -> tuple[str, pd.Series]:
    """series must be date-indexed and sorted; may contain NaN (an upstream
    data-quality concern — dropped here purely for fitting).

    Not expected to be reachable with an entirely empty/all-NaN series in
    the normal pipeline (Step 1's load_external_drivers already drops any
    driver below min_obs valid observations before it ever reaches this
    module's driver_labels loop) — guarded anyway rather than left to
    surface as a cryptic pandas error, since this function is also usable
    standalone."""
    if series.empty or series.index.max() is pd.NaT:
        raise ValueError("series has no dated observations at all — cannot project a future index")

    clean = series.dropna()
    last_date = series.index.max()
    future_index = _future_dates(last_date, horizon_periods, periods_per_year)

    if len(clean) < cfg["min_obs_for_trend_model"]:
        logger.warning("driver has only %d valid observations (< min_obs_for_trend_model=%d) — using naive_last_value",
                        len(clean), cfg["min_obs_for_trend_model"])
        flat = pd.Series(clean.iloc[-1] if len(clean) else np.nan, index=future_index)
        return "naive_last_value", flat

    seasonal_periods = periods_per_year if len(clean) >= periods_per_year * cfg["min_cycles_for_seasonal"] else None
    # Set the index freq explicitly (statsmodels otherwise warns and infers
    # it itself every fit — harmless but noisy across dozens of drivers).
    # Purely cosmetic (suppresses a warning) -- some drivers have genuine
    # gaps in their raw source data (an upstream data quality issue, not
    # introduced by this pipeline), which makes pandas correctly refuse the
    # freq assignment. A cosmetic warning-suppression step must never crash
    # the whole commodity, so this is best-effort.
    clean = clean.copy()
    try:
        clean.index.freq = "MS" if periods_per_year == 12 else "QS"
    except ValueError:
        logger.warning("driver's date index has gaps -- skipping the cosmetic freq assignment, statsmodels will infer its own")
    try:
        model = ExponentialSmoothing(
            clean,
            trend="add",
            damped_trend=cfg["damped_trend"],
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit()
        forecast = model.forecast(horizon_periods)
        forecast.index = future_index
        method = "ets_damped_trend_seasonal" if seasonal_periods else "ets_damped_trend"
        return method, forecast
    except Exception as exc:
        logger.warning("ETS fit failed (%s) — falling back to naive_last_value", exc)
        flat = pd.Series(clean.iloc[-1], index=future_index)
        return "naive_last_value", flat


def build_lagged_future_exog(
    commodity: CommodityConfig,
    external_driver_data: ExternalDriverData,
    projection_result: DriverProjectionResult,
    future_dates: list[pd.Timestamp],
    as_of: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """
    Used by arimax.py/sarimax.py only (see _common.py's docstring for why
    RF-ext/LGBM-ext use a different, "frozen at train_end" convention
    instead): for each driver, returns its LAGGED value at each future_date
    — i.e. what external_driver_loader.py's df would hold for that date,
    using the driver's REAL raw history where it reaches far enough, and
    this module's own ETS projection (project_drivers) beyond it. A driver
    lagged by L periods only has real data up to (last known raw date + L);
    forecasting further needs the driver's own projected future first, per
    this module's docstring.

    as_of: during backtesting, a "future" source_date for an early window
    is often still within the full raw_df's real historical range -- gate
    the real-value lookup to source_date <= as_of (the window's train_end)
    so a backtest window never sees the driver's actual realized value
    instead of a genuine forecast of it. Beyond that cutoff, only
    `projection_result` (which the caller must also have fit using data
    <= as_of, via project_drivers' own as_of) is used. None (the default)
    preserves the no-cutoff behavior, for live production forecasting where
    "future" genuinely means beyond all known data anyway.

    Returns one row per future_date, one column per driver (label), NaN
    where neither real (as-of-cutoff) nor projected data reaches.
    """
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    raw_df = external_driver_data.raw_df.set_index(DATE_COLUMN_NAME)
    projections_by_label = {p.label: p.projected for p in projection_result.projections}
    lag_by_label = {sel.label: sel.selected_lag for sel in external_driver_data.lag_selections}

    rows = {}
    for date in future_dates:
        row = {}
        for label, lag in lag_by_label.items():
            source_date = date - pd.DateOffset(months=lag * (12 // periods_per_year))
            real_value_allowed = as_of is None or source_date <= as_of
            if real_value_allowed and label in raw_df.columns and source_date in raw_df.index and pd.notna(raw_df.loc[source_date, label]):
                row[label] = raw_df.loc[source_date, label]
            elif label in projections_by_label and source_date in projections_by_label[label].index:
                row[label] = projections_by_label[label].loc[source_date]
            else:
                row[label] = np.nan
        rows[date] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def project_drivers(
    commodity: CommodityConfig, external_driver_data: ExternalDriverData, as_of: "pd.Timestamp | None" = None,
) -> DriverProjectionResult:
    """
    Projects every driver in external_driver_data.raw_df (the raw, unlagged
    series) config/driver_projection.yaml's max_horizon_months into the
    future (scaled to the commodity's frequency).

    as_of: fits each driver's ETS model on raw_df truncated to dates
    <= as_of, so a backtest window's driver projection only ever reflects
    what was actually known by that window's train_end. None (the default)
    uses the full raw_df -- correct for a single live-production
    projection, but NOT for backtesting (arimax.py/sarimax.py call this once
    per window with as_of=that window's train_end).
    """
    if external_driver_data.raw_df is None:
        raise ValueError(f"{commodity.id}: external_driver_data.raw_df is None — run load_external_drivers first")

    cfg = _load_yaml(DRIVER_PROJECTION_YAML)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    horizon_periods = _scale_periods(cfg["max_horizon_months"], periods_per_year)

    raw_df = external_driver_data.raw_df.set_index(DATE_COLUMN_NAME)
    if as_of is not None:
        raw_df = raw_df[raw_df.index <= as_of]
    driver_labels = [sel.label for sel in external_driver_data.lag_selections]

    projections = []
    for label in driver_labels:
        method, forecast = _project_one_driver(raw_df[label], periods_per_year, horizon_periods, cfg)
        projections.append(DriverProjection(label=label, method=method, projected=forecast))
        logger.info("%s: projected '%s' %d periods ahead via %s", commodity.id, label, horizon_periods, method)

    return DriverProjectionResult(commodity_id=commodity.id, projections=projections)
