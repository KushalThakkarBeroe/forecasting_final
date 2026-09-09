"""
Reads a commodity's {id}_ext_var.xlsx by column position and computes each
external driver's own optimal lag independently, via Pearson/Spearman/
ensemble correlation search within the commodity's configured
lag_search_range. There is no fixed-lag mode anywhere in this codebase —
every driver's lag is chosen on its own merits.

Scope note: this module only loads and lags the raw driver data. The
data-adequacy gate (exclude/impute based on coverage %,
config/scoring.yaml's data_adequacy section) is external_feature_merge.py's
job, not duplicated here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import CommodityConfig

logger = logging.getLogger(__name__)

DATE_COLUMN_NAME = "Forecasted Period"


def _to_numeric(series: pd.Series) -> pd.Series:
    """pd.to_numeric with thousands-separator commas stripped first --
    without this, a cell entered as text with a comma (e.g. "1,341.00",
    seen in sbr/cheese's source files starting once the value crosses
    1000) silently becomes NaN under plain pd.to_numeric(errors="coerce"),
    even though the cell holds a real, valid number. Only touches actual
    string values; numeric cells (int/float) and genuine blanks pass
    through untouched, matching the original behavior for real gaps."""
    cleaned = series.map(lambda v: v.replace(",", "").strip() if isinstance(v, str) else v)
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_column_name(col: str) -> str:
    """Strip special characters, collapse underscores. Works on any string,
    no hardcoded assumptions about content -- including embedded newlines/
    tabs (seen in some raw source headers, e.g. a driver label containing
    "...World\n_HS Code_..."), which would otherwise survive untouched and
    break LightGBM's model serialization ("Wrong size of feature_names")
    when used as a feature name."""
    cleaned = re.sub(r"\s+", " ", col)
    cleaned = re.sub(r'[\/\(\)\[\]{},:"\']+', "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_ ")


def _pearson_at_lag(target: pd.Series, variable: pd.Series, lag: int, min_obs: int = 5) -> float:
    lagged = variable.shift(lag)
    mask = target.notna() & lagged.notna()
    if mask.sum() < min_obs:
        return np.nan
    try:
        return target[mask].corr(lagged[mask], method="pearson")
    except Exception:
        return np.nan


def _spearman_at_lag(target: pd.Series, variable: pd.Series, lag: int, min_obs: int = 5) -> float:
    lagged = variable.shift(lag)
    mask = target.notna() & lagged.notna()
    if mask.sum() < min_obs:
        return np.nan
    try:
        return target[mask].corr(lagged[mask], method="spearman")
    except Exception:
        return np.nan


def _ensemble_score(pearson: float, spearman: float, pearson_weight: float, spearman_weight: float) -> float:
    if np.isnan(pearson) or np.isnan(spearman):
        return np.nan
    return abs(pearson) * pearson_weight + abs(spearman) * spearman_weight


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Position-based date column, dtype-agnostic. Already-datetime columns
    (the common case once pandas reads a real Excel date cell) pass straight
    through. String-formatted columns fall back to an auto-detect
    heuristic: 4-digit-year-first -> ISO, month-name-present -> '%b-%y'."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series)
    else:
        sample = str(series.dropna().iloc[0]) if series.notna().any() else ""
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if "-" in sample and len(sample.split("-")[0]) == 4:
            parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
        elif any(m in sample for m in month_names):
            parsed = pd.to_datetime(series, format="%b-%y", errors="coerce")
        else:
            parsed = pd.to_datetime(series, errors="coerce")
    return parsed.apply(lambda d: d.replace(day=1) if pd.notna(d) else d)


@dataclass
class DriverLagSelection:
    label: str
    column: int
    selected_lag: int
    ensemble_score: float
    pearson_by_lag: dict[int, float]
    spearman_by_lag: dict[int, float]
    ensemble_by_lag: dict[int, float]


@dataclass
class ExternalDriverData:
    commodity_id: str
    df: pd.DataFrame  # DATE_COLUMN_NAME, price column, one (lagged) column per driver
    price_column: str
    lag_selections: list[DriverLagSelection] = field(default_factory=list)
    raw_df: pd.DataFrame = None  # DATE_COLUMN_NAME + each driver's own RAW (unlagged)
    # series — needed by Step 6 (driver_projection.py), which forecasts a
    # driver's own future path and must not project an already-shifted
    # series (that would bake this commodity's lag into the driver's own
    # trend). Not used by Step 5's merge; that step correctly wants df
    # (lagged), not raw_df.


def load_external_drivers(
    commodity: CommodityConfig,
    min_obs: int = 5,
    pearson_weight: float = 1.0,
    spearman_weight: float = 0.0,
) -> ExternalDriverData:
    """
    Loads commodity.data_file by column position, computes each driver's own
    optimal lag (ensemble Pearson+Spearman correlation, searched within
    drivers_config['lag_search_range']), applies it, and returns a clean
    dataframe: DATE_COLUMN_NAME + price + one lagged column per driver.

    Works under either drivers_status: 'candidates' (candidate_drivers key)
    or 'selected' (selected_drivers key) — whichever list is present in
    commodity.drivers_config is what gets loaded and lagged.
    """
    if abs((pearson_weight + spearman_weight) - 1.0) > 1e-6:
        raise ValueError(f"pearson_weight + spearman_weight must sum to 1.0, got {pearson_weight + spearman_weight}")

    cfg = commodity.drivers_config
    sheet_name = cfg["sheet_name"]
    header_rows = cfg["header_rows"]
    date_col_idx = cfg["date_column"] - 1  # config is 1-indexed (Excel-style)
    price_col_idx = cfg["price_column"] - 1
    lag_lo, lag_hi = cfg["lag_search_range"]
    lag_range = list(range(lag_lo, lag_hi + 1))

    # An empty list is legitimate -- some commodities are configured with
    # selected_drivers: [] on purpose (no drivers provided; univariate
    # models only). Nothing downstream in this function actually needs
    # driver_entries to be non-empty -- result_df already has
    # DATE_COLUMN_NAME + price before the loop below runs, so an empty list
    # just yields a plain univariate series, which is exactly what these
    # commodities are meant to get.
    driver_entries = cfg.get("selected_drivers") or cfg.get("candidate_drivers") or []

    raw = pd.read_excel(commodity.data_file, sheet_name=sheet_name, header=None)
    data = raw.iloc[header_rows:].reset_index(drop=True)

    dates = _parse_date_column(data.iloc[:, date_col_idx])
    price = _to_numeric(data.iloc[:, price_col_idx])

    price_label = _clean_column_name(commodity.display_name)
    result_df = pd.DataFrame({DATE_COLUMN_NAME: dates.values, price_label: price.values})
    raw_df = pd.DataFrame({DATE_COLUMN_NAME: dates.values})

    lag_selections: list[DriverLagSelection] = []
    used_labels: set[str] = set()

    for entry in driver_entries:
        col_idx = entry["column"] - 1
        base_label = _clean_column_name(entry["label"])
        # Disambiguate label collisions: two drivers can share the same
        # base label but differ only by unit (e.g. soybean's two "Macro
        # Factors — Exchange Rate" columns, USD-CNY vs USD-BRL). Without
        # this, result_df[label] = ... would silently overwrite one
        # driver's data with the other's on the second assignment.
        if base_label not in used_labels:
            label = base_label
        else:
            label = f"{base_label}_{_clean_column_name(str(entry.get('unit', '')))}"
            if label in used_labels:
                label = f"{base_label}_col{entry['column']}"
        used_labels.add(label)

        series = _to_numeric(data.iloc[:, col_idx])

        if series.notna().sum() < min_obs:
            logger.warning("%s: driver '%s' has fewer than %d valid observations — skipped", commodity.id, label, min_obs)
            continue

        pearson_by_lag = {lag: _pearson_at_lag(price, series, lag, min_obs) for lag in lag_range}
        spearman_by_lag = {lag: _spearman_at_lag(price, series, lag, min_obs) for lag in lag_range}
        ensemble_by_lag = {
            lag: _ensemble_score(pearson_by_lag[lag], spearman_by_lag[lag], pearson_weight, spearman_weight)
            for lag in lag_range
        }
        valid = {lag: score for lag, score in ensemble_by_lag.items() if not np.isnan(score)}
        if not valid:
            logger.warning("%s: driver '%s' has no valid correlation at any lag in %s — skipped", commodity.id, label, lag_range)
            continue

        best_lag = max(valid, key=valid.get)
        result_df[label] = series.shift(best_lag).values
        raw_df[label] = series.values

        lag_selections.append(
            DriverLagSelection(
                label=label,
                column=entry["column"],
                selected_lag=best_lag,
                ensemble_score=valid[best_lag],
                pearson_by_lag=pearson_by_lag,
                spearman_by_lag=spearman_by_lag,
                ensemble_by_lag=ensemble_by_lag,
            )
        )

    return ExternalDriverData(
        commodity_id=commodity.id,
        df=result_df,
        price_column=price_label,
        lag_selections=lag_selections,
        raw_df=raw_df,
    )
