"""
Core internal feature engineering: lags, rolling stats, period-on-period
returns, trend ratios, volatility, calendar, seasonality, expanding stats —
one shared feature set used by every commodity regardless of frequency, no
commodity-specific branching.

Period-aware, not month-hardcoded: every lag/window size in
config/internal_features.yaml is expressed in months and scaled here to the
commodity's actual frequency, preserving the same real-world time span (see
_scale_periods). A quarterly commodity's "3-month lag" becomes 1 period,
not 3.

Adds boolean calendar indicator flags (is_q1..is_q4 always; is_jan..is_dec
only for monthly commodities, since a quarterly series has no sub-quarter
granularity), alongside integer/cyclical calendar encoding.

Input is the price dataframe from src/data/external_driver_loader.py's
ExternalDriverData (DATE_COLUMN_NAME + a price column) — the {id}_ext_var.xlsx
file is the canonical price+driver source for this whole project.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

from config_loader import PROJECT_ROOT, CommodityConfig
from external_driver_loader import DATE_COLUMN_NAME

logger = logging.getLogger(__name__)

INTERNAL_FEATURES_YAML = PROJECT_ROOT / "config" / "internal_features.yaml"

PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4}
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _scale_periods(month_value: int, periods_per_year: int, min_value: int = 1) -> int:
    """Converts a lag/window originally expressed in months into the
    equivalent number of periods for this frequency, preserving the same
    real-world time span. E.g. 12 always means "1 year ago" — 12 monthly
    periods or 4 quarterly periods. Monthly (periods_per_year=12) is a
    no-op scale of 1.

    min_value defaults to 1 (fine for lags), but ROLLING windows that
    compute std/variance need min_value=2 — a rolling window of size 1 has
    no variance (pandas returns NaN for every row), which would otherwise
    propagate through roll_zscore/coef_var and wipe out the whole dataframe
    at the final dropna() for low-frequency commodities (caught by testing
    a synthetic quarterly series: a 3-month window scales to 1 quarterly
    period, which is exactly this trap)."""
    return max(min_value, round(month_value * periods_per_year / 12))


def derive_scaled_periods(commodity: CommodityConfig) -> dict:
    """
    Single source of truth for every lag/window size internal_features.py
    scales from config/internal_features.yaml (months) to this commodity's
    actual periods. Shared with models/_common.py's _build_future_row via
    its caller (rf.py, lgbm.py, ...) so the recursive-forecast feature
    builder derives lag_periods/roll_windows/roll_short/roll_long/cum_a/
    cum_b identically to how build_internal_features derived them for
    training — any drift here would silently mismatch a model's training
    and inference feature columns.
    """
    if commodity.frequency not in PERIODS_PER_YEAR:
        raise ValueError(f"Unknown frequency {commodity.frequency!r}, expected one of {list(PERIODS_PER_YEAR)}")

    cfg = _load_yaml(INTERNAL_FEATURES_YAML)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    lag_periods = sorted({_scale_periods(m, periods_per_year) for m in cfg["lag_periods_months"]})
    # min_value=2: these windows feed std/variance-based stats (roll_std,
    # roll_zscore, coef_var) — a window of 1 has no variance (always NaN),
    # which would otherwise wipe out every row at the final dropna() for
    # low-frequency commodities. See _scale_periods' docstring.
    roll_windows = sorted({_scale_periods(m, periods_per_year, min_value=2) for m in cfg["roll_window_months"]})
    long_window = _scale_periods(cfg["long_window_months"], periods_per_year, min_value=2)
    long_min_periods = _scale_periods(cfg["long_window_min_periods_months"], periods_per_year)
    roll_short = _scale_periods(3, periods_per_year, min_value=2)
    roll_long = _scale_periods(12, periods_per_year, min_value=2)
    cum_a = _scale_periods(3, periods_per_year)
    cum_b = _scale_periods(6, periods_per_year)

    return {
        "periods_per_year": periods_per_year,
        "lag_periods": lag_periods,
        "roll_windows": roll_windows,
        "long_window": long_window,
        "long_min_periods": long_min_periods,
        "roll_short": roll_short,
        "roll_long": roll_long,
        "cum_a": cum_a,
        "cum_b": cum_b,
    }


def build_internal_features(commodity: CommodityConfig, price_df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    price_df must have DATE_COLUMN_NAME (datetime) and price_col (numeric),
    one row per period. Returns a new dataframe: DATE_COLUMN_NAME, price_col,
    + every engineered feature. Rows with any NaN (from warmup lags/rolling
    windows) are dropped.
    """
    scaled = derive_scaled_periods(commodity)
    periods_per_year = scaled["periods_per_year"]
    lag_periods = scaled["lag_periods"]
    roll_windows = scaled["roll_windows"]
    long_window = scaled["long_window"]
    long_min_periods = scaled["long_min_periods"]
    roll_short = scaled["roll_short"]
    roll_long = scaled["roll_long"]
    cum_a = scaled["cum_a"]
    cum_b = scaled["cum_b"]

    df = price_df[[DATE_COLUMN_NAME, price_col]].copy()
    df = df.dropna(subset=[price_col])
    df = df.sort_values(DATE_COLUMN_NAME).reset_index(drop=True)
    df.rename(columns={price_col: "Price"}, inplace=True)

    # ---- Step 4: lag features ----
    for lag in lag_periods:
        df[f"lag_{lag}"] = df["Price"].shift(lag)

    # ---- Step 4: rolling statistics ----
    for window in roll_windows:
        df[f"roll_mean_{window}"] = df["Price"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["Price"].shift(1).rolling(window).std()
        df[f"roll_min_{window}"] = df["Price"].shift(1).rolling(window).min()
        df[f"roll_max_{window}"] = df["Price"].shift(1).rolling(window).max()

    # ---- Step 4: rolling z-score ----
    # Price.shift(1) in the numerator (not bare Price) -- see the
    # period-on-period block below for why: _build_future_row (prediction
    # time) always uses history.iloc[-1] (the last known value) as its
    # stand-in for "this row's price", since the real value is what's
    # being forecast. Training must use that same stand-in, not the real
    # value, or the model learns a signal prediction time can never supply.
    for window in roll_windows:
        mean = df[f"roll_mean_{window}"]
        std = df[f"roll_std_{window}"]
        df[f"roll_zscore_{window}"] = ((df["Price"].shift(1) - mean) / std.where(std > 0, np.nan)).fillna(0)

    # ---- Step 4: period-on-period change / returns ----
    # These must NOT read today's real Price directly: _common.py's
    # _build_future_row (the prediction-time builder) uses
    # history.iloc[-1]/-2/-3/-4 -- i.e. the last KNOWN value and its
    # predecessors, never the value being forecast. Shifting by 1 here
    # makes training see the exact same "yesterday and earlier" values
    # prediction time supplies, column-for-column. mom_pct_lag1/lag2 need
    # no separate change -- they're already shift(1)/shift(2) of
    # mom_pct_change, so fixing mom_pct_change alone carries through
    # correctly.
    df["mom_change"] = df["Price"].shift(1).diff(1)
    df["mom_pct_change"] = df["Price"].shift(1).pct_change(1) * 100
    df["mom_pct_lag1"] = df["mom_pct_change"].shift(1)
    df["mom_pct_lag2"] = df["mom_pct_change"].shift(2)
    df[f"cum_return_{cum_a}p"] = df["Price"].shift(1).pct_change(cum_a) * 100
    df[f"cum_return_{cum_b}p"] = df["Price"].shift(1).pct_change(cum_b) * 100

    # ---- Step 4: trend ratios ----
    # roll_short/roll_long (min_value=2, see derive_scaled_periods) must
    # resolve to columns the roll_windows loop above actually created.
    # price_to_ma_long/short: same Price.shift(1) fix as above (matches
    # _build_future_row's price = history.iloc[-1]). ma_short_to_ma_long
    # needs no change -- both sides were already shift(1)-based rolling
    # means, no current-price term to begin with.
    df["price_to_ma_long"] = df["Price"].shift(1) / df[f"roll_mean_{roll_long}"].replace(0, np.nan)
    df["price_to_ma_short"] = df["Price"].shift(1) / df[f"roll_mean_{roll_short}"].replace(0, np.nan)
    df["ma_short_to_ma_long"] = df[f"roll_mean_{roll_short}"] / df[f"roll_mean_{roll_long}"].replace(0, np.nan)

    # ---- Step 4: volatility (coefficient of variation) ----
    for window in roll_windows:
        mean = df[f"roll_mean_{window}"]
        std = df[f"roll_std_{window}"]
        df[f"coef_var_{window}"] = std / mean.replace(0, np.nan)

    # ---- Step 3: calendar features (integer + cyclical, ported) ----
    df["month"] = df[DATE_COLUMN_NAME].dt.month
    df["quarter"] = df[DATE_COLUMN_NAME].dt.quarter
    df["year"] = df[DATE_COLUMN_NAME].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    df["years_since_start"] = df["year"] - df["year"].min()

    # ---- Step 3: boolean calendar flags ----
    for q in range(1, 5):
        df[f"is_q{q}"] = (df["quarter"] == q).astype(int)
    if commodity.frequency == "monthly":
        for m, name in enumerate(MONTH_NAMES, start=1):
            df[f"is_{name}"] = (df["month"] == m).astype(int)

    # ---- Step 4: expanding statistics (full history) ----
    # price_to_expanding_mean: Price.shift(1), same current-price fix as
    # the block above (matches _build_future_row's price = history.iloc[-1]).
    df["expanding_mean"] = df["Price"].shift(1).expanding().mean()
    df["expanding_std"] = df["Price"].shift(1).expanding().std()
    df["price_to_expanding_mean"] = df["Price"].shift(1) / df["expanding_mean"].replace(0, np.nan)

    # ---- Step 3: seasonality index ----
    # Point-in-time fix: verified against _build_future_row (prediction
    # time), which builds this from `history` -- values strictly before the
    # row being forecast. Its month_prices.mean() is "mean of all PRIOR
    # occurrences of this month"; its global_mean is "mean of all PRIOR
    # values". Training used to average over the WHOLE file, past and
    # future months alike -- prediction time never has that. groupby+
    # shift(1)+expanding() reproduces "prior occurrences of this month
    # only", and expanding_mean above already is "prior values, any month".
    month_expanding_mean = df.groupby("month")["Price"].transform(lambda s: s.shift(1).expanding().mean())
    df["seasonal_index"] = month_expanding_mean / df["expanding_mean"].replace(0, np.nan)

    # ---- Step 4: long-window ("3-year context") statistics ----
    df[f"rolling_mean_{long_window}"] = df["Price"].shift(1).rolling(window=long_window, min_periods=long_min_periods).mean()
    df[f"rolling_std_{long_window}"] = df["Price"].shift(1).rolling(window=long_window, min_periods=long_min_periods).std()
    # NOT given the Price.shift(1) fix -- price_to_rolling_mean_{long_window}
    # is a known, tracked exception: _build_future_row never computes it at
    # recursive-forecasting time (always NaN there). Shifting this alone
    # wouldn't fix that train/predict mismatch, since prediction time
    # supplies nothing for it either way.
    df[f"price_to_rolling_mean_{long_window}"] = df["Price"] / df[f"rolling_mean_{long_window}"].replace(0, np.nan)

    df.rename(columns={"Price": price_col}, inplace=True)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    logger.info(
        "%s: features built, %d -> %d rows (dropped %d for warmup), %d columns",
        commodity.id, n_before, n_after, n_before - n_after, len(df.columns),
    )

    return df
