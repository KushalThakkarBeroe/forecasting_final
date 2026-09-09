"""
Shared sliding-window backtest engine and output schema for every
forecasting technique (ARIMA, SARIMA, ETS, RF, LightGBM, ...) — one engine,
not copy-pasted per technique.

Directional Accuracy design note: this engine computes BOTH a fixed-anchor
DA (baseline pinned at the window's train-end actual for every step) and a
rolling month-over-month DA for every row, so composite_score.py can pick
the right one per horizon (config/scoring.yaml's horizon_overrides wants
Rolling MoM for Short/Medium Term, Fixed Anchor for Long Term).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler

from config_loader import CONFIG_DIR, CommodityConfig

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MULTICOLLINEARITY_YAML = CONFIG_DIR / "multicollinearity.yaml"


def _load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def drop_multicollinear_features(
    feature_df: pd.DataFrame, feature_cols: list[str], price_col: str, ext_var_cols: "list[str] | None" = None,
) -> list[str]:
    """
    Drops near-duplicate internal engineered features before they reach a
    model -- the same redundancy idea feature_selection.py already applies
    to external drivers, applied here to internal_features.py's output
    instead. Selected drivers (ext_var_cols) are never touched here, even
    when feature_cols includes them (rf_ext.py/lgbm_ext.py's merged
    feature_df) -- they already went through their own redundancy pass
    during driver selection, on a different basis (correlation with price
    at each driver's own selected lag), and re-filtering them here on raw
    pairwise correlation would risk quietly overturning that decision.

    Ranks the internal-feature candidates by |correlation with price|
    descending, keeps greedily, drops any candidate whose correlation with
    an already-kept candidate exceeds config/multicollinearity.yaml's
    correlation_threshold. Returns feature_cols unchanged (in its original
    order) if the check is disabled in config, with driver columns always
    included and always in their original relative position.
    """
    ext_var_cols = ext_var_cols or []
    internal_cols = [c for c in feature_cols if c not in ext_var_cols]

    cfg = _load_yaml(MULTICOLLINEARITY_YAML)
    if not cfg.get("enabled", True) or not internal_cols:
        return feature_cols
    threshold = cfg["correlation_threshold"]

    corr_with_price = feature_df[internal_cols].corrwith(feature_df[price_col]).abs()
    ranked = corr_with_price.sort_values(ascending=False).index.tolist()

    kept: list[str] = []
    for col in ranked:
        redundant = False
        for k in kept:
            pair_corr = feature_df[col].corr(feature_df[k])
            if pd.notna(pair_corr) and abs(pair_corr) > threshold:
                redundant = True
                break
        if not redundant:
            kept.append(col)

    kept_set = set(kept) | set(ext_var_cols)
    return [c for c in feature_cols if c in kept_set]

PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4}

# A technique's fit_and_forecast_fn: (train_series, params, steps) -> forecast Series of
# length `steps` (same convention throughout: index-free, matched positionally to the
# caller's own date list), or None if fitting failed for this window.
FitAndForecastFn = Callable[[pd.Series, dict, int], "pd.Series | None"]
# A technique's param_space_fn: (optuna.Trial) -> params dict, or None to tell
# Optuna this trial is invalid (e.g. an ARIMA order that isn't allowed).
ParamSpaceFn = Callable[[optuna.Trial], "dict | None"]


def _period_offset(n_periods: int, periods_per_year: int) -> pd.DateOffset:
    """1 period = 1 month for monthly commodities, 3 months for quarterly —
    matches every other period-scaling convention in this codebase
    (internal_features.py, driver_projection.py)."""
    months_per_period = 12 // periods_per_year
    return pd.DateOffset(months=n_periods * months_per_period)


def _freq_str(periods_per_year: int) -> str:
    return "MS" if periods_per_year == 12 else "QS"


def resolve_horizon(
    warmup_periods: int, forecast_periods: int, horizon_label: "str | None" = None,
) -> "tuple[tuple[int, int], str]":
    """
    Every run_* technique function (arima.py, sarima.py, ets.py, rf.py,
    lgbm.py, rf_ext.py, lgbm_ext.py, arimax.py, sarimax.py) takes
    warmup_periods/forecast_periods DIRECTLY, not a "short"/"medium"/"long"
    label resolved through config/horizon_defaults.yaml — any warmup/
    forecast combination works, not just the three preset buckets. This
    turns that pair into the (m_start, m_end) tuple the engine functions
    in this module (_run_windowed_backtest and friends) already accept,
    plus a human-readable label for the "Horizon" column/logging when the
    caller doesn't supply their own.

    Production batch runs that DO want the standard short/medium/long
    buckets read them from commodity.horizon_periods (config/
    horizon_defaults.yaml) and convert directly:
        m_start, m_end = commodity.horizon_periods.short
        run_arima(commodity, price_series, warmup_periods=m_start - 1,
                   forecast_periods=m_end - m_start + 1, horizon_label="short")
    """
    if warmup_periods < 0:
        raise ValueError(f"warmup_periods must be >= 0, got {warmup_periods}")
    if forecast_periods < 1:
        raise ValueError(f"forecast_periods must be >= 1, got {forecast_periods}")
    m_start = warmup_periods + 1
    m_end = warmup_periods + forecast_periods
    label = horizon_label if horizon_label is not None else f"M{m_start}-M{m_end}"
    return (m_start, m_end), label


ROW_METRIC_KEYS = ["Abs Error", "APE (%)", "Row MAE", "Row MSE", "Row RMSE", "Row MAPE (%)", "Row MPE (%)", "Row Accuracy"]


def _row_metrics(actual: float, predicted: float) -> dict:
    """
    Per-row error metrics, including MPE (%): unlike APE/MAPE (absolute,
    always >= 0), MPE is SIGNED — (actual - predicted) — positive means the
    model under-forecast, negative means it over-forecast. This is what
    config/scoring.yaml's normalized_bias_penalty needs to detect systematic
    bias, which an absolute-error metric can't reveal (a model that's +5%
    high half the time and -5% low half the time has 0% bias by MPE despite
    a non-zero MAPE).
    """
    actual = float(actual)
    predicted = float(predicted)
    abs_error = abs(actual - predicted)
    mse = (actual - predicted) ** 2
    ape = abs((actual - predicted) / actual) * 100 if actual != 0 else np.nan
    pe = ((actual - predicted) / actual) * 100 if actual != 0 else np.nan
    accuracy = max(0, 100 - ape) if not np.isnan(ape) else np.nan
    return {
        "Abs Error": round(abs_error, 4),
        "APE (%)": round(ape, 4) if not np.isnan(ape) else np.nan,
        "Row MAE": round(abs_error, 4),
        "Row MSE": round(mse, 4),
        "Row RMSE": round(np.sqrt(mse), 4),
        "Row MAPE (%)": round(ape, 4) if not np.isnan(ape) else np.nan,
        "Row MPE (%)": round(pe, 4) if not np.isnan(pe) else np.nan,
        "Row Accuracy": round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
    }


_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def _build_future_row(
    history: pd.Series,
    future_date: pd.Timestamp,
    feature_cols: list[str],
    lag_periods: list[int],
    roll_windows: list[int],
    roll_short: int,
    roll_long: int,
    cum_a: int,
    cum_b: int,
    is_monthly: bool,
    long_window: "int | None" = None,
    long_min_periods: "int | None" = None,
    ext_df: "pd.DataFrame | None" = None,
) -> dict:
    """
    Recursive-forecasting counterpart to internal_features.py's
    build_internal_features: builds ONE feature row for future_date from
    `history` (known/already-predicted values up to but not including
    future_date — future_date's own price is what's being forecast, so it
    can never be used to build its own features), matching
    internal_features.py's column names so a model trained on those columns
    can be fed this row directly.

    lag_periods/roll_windows/roll_short/roll_long/cum_a/cum_b must be the
    SAME period-scaled values internal_features.py derived for this
    commodity (via its _scale_periods) — passed in by the caller rather than
    recomputed here, so this module stays decoupled from
    features/internal_features.py's config loading.

    roll_zscore falls back to 0 (not NaN) when std is 0/undefined, matching
    internal_features.py's own fillna(0) — keeps this function's output on
    the same distribution the model actually trained on.

    The "3-year context" block (rolling_mean_{long_window},
    rolling_std_{long_window}, price_to_rolling_mean_{long_window}) mirrors
    what internal_features.py computes for training. long_window/
    long_min_periods must be the SAME values internal_features.py derived
    for this commodity. None (the default) skips this block entirely — used
    by _build_prediction_row_ext's own call to this function, which only
    ever asks for calendar/expanding features here (the 3-year-context
    columns are handled separately there, via its own frozen-at-train_end +
    safety-horizon mechanism).
    """
    row: dict = {}

    for lag in lag_periods:
        row[f"lag_{lag}"] = history.iloc[-lag] if len(history) >= lag else np.nan

    for window in roll_windows:
        past = history.iloc[-window:] if len(history) >= window else history
        row[f"roll_mean_{window}"] = past.mean()
        row[f"roll_std_{window}"] = past.std()
        row[f"roll_min_{window}"] = past.min()
        row[f"roll_max_{window}"] = past.max()

    for window in roll_windows:
        mean = row[f"roll_mean_{window}"]
        std = row[f"roll_std_{window}"]
        z = (history.iloc[-1] - mean) / std if std and std > 0 else np.nan
        row[f"roll_zscore_{window}"] = 0.0 if pd.isna(z) else z

    row["mom_change"] = history.iloc[-1] - history.iloc[-2] if len(history) >= 2 else np.nan
    row["mom_pct_change"] = (
        (history.iloc[-1] - history.iloc[-2]) / history.iloc[-2] * 100
        if len(history) >= 2 and history.iloc[-2] != 0 else np.nan
    )
    row["mom_pct_lag1"] = (
        (history.iloc[-2] - history.iloc[-3]) / history.iloc[-3] * 100
        if len(history) >= 3 and history.iloc[-3] != 0 else np.nan
    )
    row["mom_pct_lag2"] = (
        (history.iloc[-3] - history.iloc[-4]) / history.iloc[-4] * 100
        if len(history) >= 4 and history.iloc[-4] != 0 else np.nan
    )
    row[f"cum_return_{cum_a}p"] = (
        (history.iloc[-1] - history.iloc[-1 - cum_a]) / history.iloc[-1 - cum_a] * 100
        if len(history) >= 1 + cum_a and history.iloc[-1 - cum_a] != 0 else np.nan
    )
    row[f"cum_return_{cum_b}p"] = (
        (history.iloc[-1] - history.iloc[-1 - cum_b]) / history.iloc[-1 - cum_b] * 100
        if len(history) >= 1 + cum_b and history.iloc[-1 - cum_b] != 0 else np.nan
    )

    price = history.iloc[-1]
    ma_long = row.get(f"roll_mean_{roll_long}", np.nan)
    ma_short = row.get(f"roll_mean_{roll_short}", np.nan)
    row["price_to_ma_long"] = _safe_div(price, ma_long)
    row["price_to_ma_short"] = _safe_div(price, ma_short)
    row["ma_short_to_ma_long"] = _safe_div(ma_short, ma_long)

    for window in roll_windows:
        mean = row[f"roll_mean_{window}"]
        std = row[f"roll_std_{window}"]
        row[f"coef_var_{window}"] = _safe_div(std, mean)

    row["month"] = future_date.month
    row["quarter"] = future_date.quarter
    row["year"] = future_date.year
    row["month_sin"] = np.sin(2 * np.pi * future_date.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * future_date.month / 12)
    row["quarter_sin"] = np.sin(2 * np.pi * future_date.quarter / 4)
    row["quarter_cos"] = np.cos(2 * np.pi * future_date.quarter / 4)
    row["years_since_start"] = future_date.year - history.index.min().year

    for q in range(1, 5):
        row[f"is_q{q}"] = int(future_date.quarter == q)
    if is_monthly:
        for m, name in enumerate(_MONTH_NAMES, start=1):
            row[f"is_{name}"] = int(future_date.month == m)

    month_prices = history[history.index.month == future_date.month]
    global_mean = history.mean()
    row["seasonal_index"] = _safe_div(month_prices.mean() if len(month_prices) else np.nan, global_mean)

    row["expanding_mean"] = history.mean()
    row["expanding_std"] = history.std()
    row["price_to_expanding_mean"] = _safe_div(price, history.mean())

    # "3-year context" block -- matches internal_features.py's
    # df["Price"].shift(1).rolling(window=long_window, min_periods=
    # long_min_periods): the last long_window values of `history` (already
    # "up to but not including future_date", same alignment as shift(1)),
    # NaN if fewer than long_min_periods of them are available. long_window
    # is None only for _build_prediction_row_ext's call (see docstring) --
    # that path handles these 3 columns separately and never requests them
    # here.
    if long_window is not None:
        long_window_past = history.iloc[-long_window:] if len(history) >= long_window else history
        if len(long_window_past) >= long_min_periods:
            row[f"rolling_mean_{long_window}"] = long_window_past.mean()
            row[f"rolling_std_{long_window}"] = long_window_past.std()
        else:
            row[f"rolling_mean_{long_window}"] = np.nan
            row[f"rolling_std_{long_window}"] = np.nan
        row[f"price_to_rolling_mean_{long_window}"] = _safe_div(price, row[f"rolling_mean_{long_window}"])

    if ext_df is not None:
        if future_date in ext_df.index:
            for col in ext_df.columns:
                row[col] = ext_df.loc[future_date, col]
        else:
            for col in ext_df.columns:
                col_hist = ext_df[col].dropna()
                row[col] = col_hist.iloc[-1] if not col_hist.empty else np.nan

    return {k: row.get(k, np.nan) for k in feature_cols}


@dataclass
class TechniqueResult:
    commodity_id: str
    technique: str
    horizon: str  # free-form label (see resolve_horizon) — "short"/"medium"/"long" by convention for production runs, not enforced
    best_params: dict
    detail_df: pd.DataFrame
    summary_df: pd.DataFrame
    matrix_df: pd.DataFrame


def search_best_params(
    price_series: pd.Series,
    window_ends: list[pd.Timestamp],
    forecast_periods: int,
    periods_per_year: int,
    param_space_fn: ParamSpaceFn,
    fit_and_forecast_fn: FitAndForecastFn,
    n_trials: int,
    fixed_params: dict | None,
    min_train_size: int = 15,
) -> tuple[dict, "optuna.Study | None"]:
    """
    Global Optuna search (or fixed_params passthrough) across ALL windows
    combined — one set of hyperparameters applied to every window, not
    per-window tuning. Objective = mean out-of-sample MAPE over the LAST
    forecast_periods steps of each window's training data, averaged across
    every window with enough history.

    Note: the search objective does NOT carve out a warmup gap even when the
    technique's real per-window run does (see run_sliding_window_backtest) —
    it always forecasts exactly forecast_periods steps directly.
    Hyperparameters are tuned for "forecast N steps ahead with no gap", then
    reused for the warmup+forecast execution.
    """
    if fixed_params is not None:
        return fixed_params.copy(), None

    freq = _freq_str(periods_per_year)

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        if params is None:
            return float("inf")

        window_mapes = []
        for train_end in window_ends:
            train_series = price_series[price_series.index <= train_end]
            if len(train_series) < min_train_size:
                continue

            val_series = train_series.iloc[-forecast_periods:]
            train_only = train_series.iloc[:-forecast_periods]
            train_only = train_only.copy()
            train_only.index = pd.DatetimeIndex(train_only.index, freq=freq)

            forecast = fit_and_forecast_fn(train_only, params, forecast_periods)
            if forecast is None:
                continue

            mape = np.mean(np.abs((val_series.values - np.asarray(forecast)) / val_series.values)) * 100
            window_mapes.append(mape)

        return np.mean(window_mapes) if window_mapes else float("inf")

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params.copy(), study


def search_best_params_recursive(
    feature_df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    target_col: str,
    window_ends: list[pd.Timestamp],
    val_size: int,
    param_space_fn: ParamSpaceFn,
    fit_predict_fn: "Callable[[dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], object]",
    n_trials: int,
    fixed_params: dict | None,
    min_train_size: int,
    recency_weights: "list[float] | None" = None,
) -> tuple[dict, "optuna.Study | None"]:
    """
    Global Optuna search for RF/LightGBM. Unlike search_best_params
    (ARIMA/SARIMA/ETS), this is a NON-recursive holdout validation: for each
    window's train_end, split that window's feature rows by position (last
    val_size rows = validation, using their REAL historical feature values —
    no recursion, no _build_future_row), fit on the train split, predict the
    validation split, score MAPE. Window MAPEs combine via a weighted
    average (recency_weights; uniform 1s by default — the weighting
    mechanism is available but off unless a caller opts in).

    fit_predict_fn(params, X_tr, y_tr, X_val, y_val) -> predictions on X_val
    is the only thing that differs between RF (plain .fit(X_tr,y_tr)) and
    LightGBM (fits WITH eval_set=[(X_val,y_val)] + early stopping — the same
    train/val split doubles as both the MAPE-scoring holdout and the
    early-stopping validation set).

    window_ends must be the SAME list the caller will later pass to
    run_recursive_backtest (both independently derived the same way
    run_sliding_window_backtest does — see that function/rf.py).
    """
    if fixed_params is not None:
        return fixed_params.copy(), None

    n_windows = len(window_ends)
    weights_base = recency_weights if recency_weights is not None else [1.0] * n_windows

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        if params is None:
            return float("inf")

        window_mapes = []
        window_weights = []
        for w_idx, train_end in enumerate(window_ends):
            train_df = feature_df[feature_df[date_col] <= train_end]
            if len(train_df) < min_train_size:
                continue

            X = train_df[feature_cols]
            y = train_df[target_col]
            X_tr, y_tr = X.iloc[:-val_size], y.iloc[:-val_size]
            X_val, y_val = X.iloc[-val_size:], y.iloc[-val_size:]
            # An additional floor on the train split itself, beyond the
            # train_df-level min_train_size check above.
            if len(X_tr) < 10 or len(X_val) == 0:
                continue

            preds = fit_predict_fn(params, X_tr, y_tr, X_val, y_val)

            mape = np.mean(np.abs((y_val.values - np.asarray(preds)) / y_val.values)) * 100
            window_mapes.append(mape)
            window_weights.append(weights_base[w_idx] if w_idx < len(weights_base) else 1.0)

        return np.average(window_mapes, weights=window_weights) if window_mapes else float("inf")

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params.copy(), study


def run_sliding_window_backtest(
    commodity: CommodityConfig,
    price_series: pd.Series,
    technique: str,
    horizon: str,
    horizon_periods: tuple[int, int],
    fit_and_forecast_fn: FitAndForecastFn,
    best_params: dict,
    n_windows: int = 15,
    min_train_size: int = 15,
    step_numbering: str = "absolute",
) -> TechniqueResult:
    """
    Runs the sliding-window backtest with best_params already resolved (see
    search_best_params). horizon_periods = (m_start, m_end) from
    commodity.horizon_periods (config/horizon_defaults.yaml) — e.g. (1,3)
    short, (4,6) medium, (7,18) long. warmup = m_start-1 periods forecast
    but discarded; forecast_periods = m_end-m_start+1 periods recorded.

    Native multi-step family (ARIMA/SARIMA/ETS): fit_and_forecast_fn fits
    once per window and forecasts all total_steps in one call. See
    run_recursive_backtest for the RF/LightGBM counterpart (one step at a
    time via _build_future_row) — both share _run_windowed_backtest for the
    window/DA/metrics/matrix assembly, which is identical across families.
    """
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    freq = _freq_str(periods_per_year)

    def forecast_fn(train_series: pd.Series, train_end: pd.Timestamp, total_steps: int):
        train_indexed = train_series.copy()
        train_indexed.index = pd.DatetimeIndex(train_indexed.index, freq=freq)
        return fit_and_forecast_fn(train_indexed, best_params, total_steps)

    return _run_windowed_backtest(
        commodity=commodity, price_series=price_series, technique=technique,
        horizon=horizon, horizon_periods=horizon_periods, forecast_fn=forecast_fn,
        n_windows=n_windows, min_train_size=min_train_size, best_params=best_params,
        step_numbering=step_numbering,
    )


def run_recursive_backtest(
    commodity: CommodityConfig,
    feature_df: pd.DataFrame,
    date_col: str,
    price_series: pd.Series,
    feature_cols: list[str],
    target_col: str,
    technique: str,
    horizon: str,
    horizon_periods: tuple[int, int],
    fit_fn: "Callable[[dict, pd.DataFrame, pd.Series, int], object]",
    best_params: dict,
    lag_periods: list[int],
    roll_windows: list[int],
    roll_short: int,
    roll_long: int,
    cum_a: int,
    cum_b: int,
    is_monthly: bool,
    long_window: int,
    long_min_periods: int,
    n_windows: int = 15,
    min_train_size: int = 20,
    val_size: "int | None" = None,
    ext_df: "pd.DataFrame | None" = None,
) -> TechniqueResult:
    """
    RF/LightGBM execution: fits ONCE per window on that window's feature_df
    rows (up to train_end) via fit_fn, then forecasts total_steps
    recursively — one step at a time, appending each prediction into
    `history` before building the next step's feature row via
    _build_future_row (distinct from search_best_params_recursive's own
    non-recursive validation objective).

    fit_fn(best_params, X_train, y_train, val_size) -> fitted model (with
    .predict()) is the only thing that differs between RF (plain
    .fit(X_train, y_train), val_size ignored) and LightGBM (holds out the
    LAST val_size rows of X_train/y_train for early-stopping — a SEPARATE
    split from any search-time holdout, with a "not enough rows to split ->
    fit on everything" fallback).

    Every predicted step, including warmup steps, is cascaded into history
    so later steps see it; warmup steps are cascaded but not recorded, same
    convention as run_sliding_window_backtest.

    min_train_size here gates on feature_df's row count (post dropna, i.e.
    after warmup lags/rolling windows are dropped) — NOT price_series's raw
    row count, which is what run_sliding_window_backtest/
    _run_windowed_backtest gate on for the native multi-step family. The
    two lengths differ (feature_df is shorter), so this function does its
    own gating inside forecast_fn and passes min_train_size=1 down to
    _run_windowed_backtest's own (irrelevant, for this family) price-series
    check so it never fires prematurely.
    """
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]

    def forecast_fn(train_series: pd.Series, train_end: pd.Timestamp, total_steps: int):
        train_df = feature_df[feature_df[date_col] <= train_end]
        if len(train_df) < min_train_size:
            return None

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        model = fit_fn(best_params, X_train, y_train, val_size)

        # history is built from feature_df's own (warmup-dropped) rows, not
        # price_series's raw row count -- the two differ in length
        # (feature_df starts later, once lag/rolling warmup is satisfied),
        # and that length feeds expanding_mean/std, seasonal_index, and
        # rolling_mean_36/std_36 in _build_future_row.
        history = train_df.set_index(date_col)[target_col].sort_index()
        preds = []
        for step_idx in range(total_steps):
            future_date = train_end + _period_offset(step_idx + 1, periods_per_year)
            row = _build_future_row(
                history=history, future_date=future_date, feature_cols=feature_cols,
                lag_periods=lag_periods, roll_windows=roll_windows,
                roll_short=roll_short, roll_long=roll_long,
                cum_a=cum_a, cum_b=cum_b, is_monthly=is_monthly,
                long_window=long_window, long_min_periods=long_min_periods, ext_df=ext_df,
            )
            X_future = pd.DataFrame([row], columns=feature_cols)
            predicted = float(model.predict(X_future)[0])
            preds.append(predicted)
            history.loc[future_date] = predicted

        return np.asarray(preds)

    return _run_windowed_backtest(
        commodity=commodity, price_series=price_series, technique=technique,
        horizon=horizon, horizon_periods=horizon_periods, forecast_fn=forecast_fn,
        n_windows=n_windows, min_train_size=1, best_params=best_params,
    )


def _run_windowed_backtest(
    commodity: CommodityConfig,
    price_series: pd.Series,
    technique: str,
    horizon: str,
    horizon_periods: tuple[int, int],
    forecast_fn: Callable[[pd.Series, pd.Timestamp, int], "np.ndarray | None"],
    best_params: dict,
    n_windows: int = 15,
    min_train_size: int = 15,
    step_numbering: str = "absolute",
) -> TechniqueResult:
    """
    Shared window/DA/metrics/matrix assembly for both run_sliding_window_backtest
    (native multi-step) and run_recursive_backtest (RF/LightGBM, one step at
    a time). forecast_fn(train_series, train_end, total_steps) -> array of
    `total_steps` values (or None to skip the window) is the only thing
    that differs between the two families.

    step_numbering controls the "Step" column's convention:
      "absolute" (default): Step = the actual M-number, counting from 1 at
        the very start of the window including warmup (e.g. M4-M18 combined
        records Step 4..18). Used by ARIMA/SARIMA/ETS and RF/LightGBM's
        recursive engine.
      "relative": Step = position within the RECORDED portion only, always
        starting at 1 regardless of warmup (e.g. M4-M18 records Step 1..15).
        Used by RF-ext/LGBM-ext/ARIMAX/SARIMAX. Caller must pick the right
        one; this function does not infer it from the technique.
    """
    if step_numbering not in ("absolute", "relative"):
        raise ValueError(f"step_numbering must be 'absolute' or 'relative', got {step_numbering!r}")

    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    m_start, m_end = horizon_periods
    warmup_periods = m_start - 1
    forecast_periods = m_end - m_start + 1
    total_steps = warmup_periods + forecast_periods

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]
    window_sort_order = {train_end: idx + 1 for idx, train_end in enumerate(sorted(window_ends))}

    detail_records = []
    summary_records = []
    matrix_data: dict[str, dict] = {}

    for w_idx, train_end in enumerate(window_ends):
        window_num = w_idx + 1
        forecast_start_date = train_end + _period_offset(warmup_periods + 1, periods_per_year)
        col_name = forecast_start_date.strftime("%b-%Y")

        train_series = price_series[price_series.index <= train_end]
        if len(train_series) < min_train_size:
            logger.warning("%s/%s window %d: insufficient training data (%d rows) — skipped",
                            technique, horizon, window_num, len(train_series))
            continue

        forecast = forecast_fn(train_series, train_end, total_steps)
        if forecast is None:
            logger.warning("%s/%s window %d: model fit failed — skipped", technique, horizon, window_num)
            continue
        forecast = np.asarray(forecast)

        all_dates = [train_end + _period_offset(i + 1, periods_per_year) for i in range(total_steps)]
        col_data = {date: round(float(val), 4) for date, val in train_series.items()}

        window_preds = []
        baseline = float(train_series.iloc[-1])
        prev_predicted = baseline

        for step_idx, fdate in enumerate(all_dates):
            step = step_idx + 1
            predicted = float(forecast[step_idx])

            # A per-step NaN (not a whole-window None) means THIS horizon
            # specifically couldn't be predicted -- e.g. run_direct_horizon_
            # backtest's safe-feature row came back with a null column for
            # this h. Skip it exactly as if it never happened (no cascade,
            # no record) -- ARIMA/SARIMA/ETS/RF/LightGBM's forecast_fn never
            # individually returns NaN, so this never triggers there.
            if np.isnan(predicted):
                continue

            if step <= warmup_periods:
                prev_predicted = predicted
                continue

            actual = (
                float(price_series.loc[fdate])
                if fdate in price_series.index and pd.notna(price_series.loc[fdate])
                else np.nan
            )

            # Fixed-anchor DA -- direction vs. the window's train-end actual
            abs_pred_dir = 1 if predicted > baseline else -1
            abs_act_dir = 1 if actual > baseline else -1 if not np.isnan(actual) else np.nan
            abs_dir_acc = int(abs_pred_dir == abs_act_dir) if not np.isnan(actual) else np.nan

            # Rolling-MoM DA -- direction vs. the previous predicted step
            mom_baseline = baseline if step == warmup_periods + 1 else prev_predicted
            mom_pred_dir = 1 if predicted > mom_baseline else -1
            mom_act_dir = 1 if actual > mom_baseline else -1 if not np.isnan(actual) else np.nan
            mom_dir_acc = int(mom_pred_dir == mom_act_dir) if not np.isnan(actual) else np.nan

            if not np.isnan(actual):
                metrics = _row_metrics(actual, predicted)
            else:
                metrics = {k: np.nan for k in ROW_METRIC_KEYS}

            output_step = step if step_numbering == "absolute" else step - warmup_periods

            record = {
                "commodity_name": commodity.id,
                "sort_order": window_sort_order[train_end],
                "Model": technique,
                "Horizon": horizon,
                "Window": window_num,
                "Train End": train_end.strftime("%b-%Y"),
                "Forecast Start": forecast_start_date.strftime("%b-%Y"),
                "Step": output_step,
                "Date": fdate.strftime("%b-%Y"),
                "Predicted": round(predicted, 4),
                "Actual": round(actual, 4) if not np.isnan(actual) else np.nan,
                "Directional Accuracy": mom_dir_acc,
                "Abs Directional Accuracy": abs_dir_acc,
                "Flag": "Actual" if not np.isnan(actual) else "No Actual Yet",
                **metrics,
            }
            detail_records.append(record)
            window_preds.append(record)
            col_data[fdate] = round(predicted, 4)
            prev_predicted = predicted

        matrix_data[col_name] = col_data

        eval_rows = [r for r in window_preds if r["Flag"] == "Actual"]
        if eval_rows:
            mapes = [r["Row MAPE (%)"] for r in eval_rows if not np.isnan(r["Row MAPE (%)"])]
            mpes = [r["Row MPE (%)"] for r in eval_rows if not np.isnan(r["Row MPE (%)"])]
            accs = [r["Row Accuracy"] for r in eval_rows if not np.isnan(r["Row Accuracy"])]
            das = [r["Directional Accuracy"] for r in eval_rows if not np.isnan(r["Directional Accuracy"])]
            abs_das = [r["Abs Directional Accuracy"] for r in eval_rows if not np.isnan(r["Abs Directional Accuracy"])]
            abs_errs = [r["Abs Error"] for r in eval_rows if not np.isnan(r["Abs Error"])]

            summary_records.append({
                "commodity_name": commodity.id, "sort_order": window_sort_order[train_end],
                "Model": technique, "Horizon": horizon, "Window": window_num,
                "Train End": train_end.strftime("%b-%Y"), "Forecast Start": forecast_start_date.strftime("%b-%Y"),
                "Months Evaluated": len(eval_rows),
                "MAE": round(np.mean(abs_errs), 4) if abs_errs else np.nan,
                "RMSE": round(np.sqrt(np.mean(np.square(abs_errs))), 4) if abs_errs else np.nan,
                "MAPE (%)": round(np.mean(mapes), 2) if mapes else np.nan,
                "MPE (%)": round(np.mean(mpes), 2) if mpes else np.nan,
                "Avg Accuracy (%)": round(np.mean(accs), 2) if accs else np.nan,
                "Directional Accuracy (%)": round(np.mean(das) * 100, 4) if das else np.nan,
                "Abs Directional Accuracy (%)": round(np.mean(abs_das) * 100, 4) if abs_das else np.nan,
            })
        else:
            summary_records.append({
                "commodity_name": commodity.id, "sort_order": window_sort_order[train_end],
                "Model": technique, "Horizon": horizon, "Window": window_num,
                "Train End": train_end.strftime("%b-%Y"), "Forecast Start": forecast_start_date.strftime("%b-%Y"),
                "Months Evaluated": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "MPE (%)": np.nan,
                "Avg Accuracy (%)": np.nan, "Directional Accuracy (%)": np.nan, "Abs Directional Accuracy (%)": np.nan,
            })

    detail_df = pd.DataFrame(detail_records)
    summary_df = pd.DataFrame(summary_records)

    all_dates_sorted = sorted(set(d for col in matrix_data.values() for d in col.keys()))
    matrix_records = []
    for date in all_dates_sorted:
        row = {"Forecasted Period": date.strftime("%b-%Y")}
        for col_name, col_data in matrix_data.items():
            row[col_name] = col_data.get(date, None)
        matrix_records.append(row)
    matrix_df = pd.DataFrame(matrix_records)

    return TechniqueResult(
        commodity_id=commodity.id, technique=technique, horizon=horizon,
        best_params=best_params, detail_df=detail_df, summary_df=summary_df, matrix_df=matrix_df,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DIRECT MULTI-HORIZON engine, shared by rf_ext.py and lgbm_ext.py.
#
# This is a THIRD forecasting mechanism, distinct from both families above:
# instead of one model forecasting all steps (native multi-step, ARIMA/
# SARIMA/ETS) or one model predicting recursively step-by-step (RF/LightGBM
# univariate), this trains a SEPARATE model per forecast horizon (H1, H2,
# ..., Htotal) via shifted targets (target_h = price.shift(-h)), each fit
# only on rows whose OWN target date is safely <= train_end
# ("label_safe_cutoff"). External driver columns are frozen at their last
# known value at train_end for every horizon -- unlike arimax.py/sarimax.py,
# which use real/projected future driver values via driver_projection.py.
# Optuna optimizes a composite of MAPE and directional accuracy
# (da_weight/da_mode) rather than pure MAPE.
#
# Every internal_features.py column is used (same as univariate RF/
# LightGBM), with each one's safety cap for a given horizon derived
# programmatically from its own lag/window size (_feature_safety_horizon) --
# a feature whose lag/window is longer than the horizon being predicted
# would otherwise leak information a real forecast wouldn't have.
# ═══════════════════════════════════════════════════════════════════════════

_ALWAYS_SAFE_FEATURES = {
    "month", "quarter", "year", "month_sin", "month_cos", "quarter_sin", "quarter_cos",
    "years_since_start", "seasonal_index", "expanding_mean", "expanding_std",
    "is_q1", "is_q2", "is_q3", "is_q4",
    *{f"is_{m}" for m in _MONTH_NAMES},
}


def _feature_safety_horizon(col: str, roll_short: int, roll_long: int, long_window: int) -> "int | None":
    """
    Returns the max horizon this internal feature can be trusted at when
    read from a "frozen" anchor row (see _build_prediction_row_ext) that
    grows progressively staler as horizon increases -- or None if it's
    always safe (calendar features, deterministic for any future date; or
    expanding_mean/std, built purely from backward-looking full history,
    no dependency on the anchor row's own current price).

    Reads the lag/window size back out of each feature's own name -- see
    this module's comment block above for why this matters.
    """
    if col in _ALWAYS_SAFE_FEATURES:
        return None
    if col.startswith("lag_"):
        return int(col[len("lag_"):])
    for prefix in ("roll_mean_", "roll_std_", "roll_min_", "roll_max_", "roll_zscore_", "coef_var_"):
        if col.startswith(prefix):
            return int(col[len(prefix):])
    if col.startswith("cum_return_") and col.endswith("p"):
        return int(col[len("cum_return_"):-1])
    if col in ("mom_change", "mom_pct_change"):
        return 1
    if col == "mom_pct_lag1":
        return 2
    if col == "mom_pct_lag2":
        return 3
    if col == "price_to_ma_short":
        return roll_short
    if col == "price_to_ma_long":
        return roll_long
    if col == "ma_short_to_ma_long":
        return roll_long
    if col == "price_to_expanding_mean":
        return 1
    if col.startswith("rolling_mean_") or col.startswith("rolling_std_") or col.startswith("price_to_rolling_mean_"):
        return long_window
    raise ValueError(f"_feature_safety_horizon: unrecognized internal feature column {col!r}")


def _safe_features_for_horizon(
    horizon: int, feature_cols: list[str], ext_var_cols: list[str],
    roll_short: int, roll_long: int, long_window: int,
) -> list[str]:
    safe = []
    for col in feature_cols:
        if col in ext_var_cols:
            safe.append(col)
            continue
        cap = _feature_safety_horizon(col, roll_short, roll_long, long_window)
        if cap is None or horizon <= cap:
            safe.append(col)
    return safe


def _build_prediction_row_ext(
    future_date: pd.Timestamp,
    train_end: pd.Timestamp,
    train_rows: pd.DataFrame,
    price_series_upto_train_end: pd.Series,
    safe_features: list[str],
    ext_var_cols: list[str],
    df_full: pd.DataFrame,
    date_col: str,
    lag_periods: list[int],
    roll_windows: list[int],
    roll_short: int,
    roll_long: int,
    cum_a: int,
    cum_b: int,
    is_monthly: bool,
) -> pd.DataFrame:
    """
    Builds ONE feature row for future_date without recursion: ext driver
    columns are frozen at their last known value at-or-before train_end;
    "always safe" calendar/expanding/seasonal columns are recomputed
    properly for future_date (delegated to _build_future_row, since the
    formulas are identical); every other safe feature is read from
    train_rows' LAST row (the most recent row not newer than
    train_end - horizon, i.e. train_end - horizon periods stale -- a real,
    fully-known historical value, never leaked from the future).
    """
    calendar_cols = [c for c in safe_features if c in _ALWAYS_SAFE_FEATURES]
    calendar_row = (
        _build_future_row(
            history=price_series_upto_train_end, future_date=future_date, feature_cols=calendar_cols,
            lag_periods=lag_periods, roll_windows=roll_windows,
            roll_short=roll_short, roll_long=roll_long, cum_a=cum_a, cum_b=cum_b, is_monthly=is_monthly,
        )
        if calendar_cols else {}
    )

    last_row = train_rows.iloc[-1:]
    feature_row = {}
    for col in safe_features:
        if col in ext_var_cols:
            known = df_full[(df_full[date_col] <= train_end) & (df_full[col].notna())]
            feature_row[col] = known[col].iloc[-1] if not known.empty else np.nan
        elif col in calendar_row:
            feature_row[col] = calendar_row[col]
        elif col in last_row.columns:
            feature_row[col] = last_row[col].values[0]
        else:
            feature_row[col] = np.nan

    return pd.DataFrame([feature_row])[safe_features]


def _composite_loss(actuals: list, preds: list, baseline: float, da_weight: float = 0.0, da_mode: str = "mom") -> float:
    """
    da_weight=0.0 (default) -> pure MAPE, matching every other technique's
    objective; da_weight>0 blends in a directional-accuracy penalty.
    """
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if len(actuals) == 0:
        return float("inf")

    nonzero = actuals != 0
    if nonzero.sum() == 0:
        return float("inf")
    mape = np.mean(np.abs((actuals[nonzero] - preds[nonzero]) / actuals[nonzero])) * 100
    if da_weight == 0.0:
        return mape

    if da_mode == "mom":
        actuals_ext = np.concatenate([[baseline], actuals])
        preds_ext = np.concatenate([[baseline], preds])
        actual_dirs = np.sign(np.diff(actuals_ext))
        pred_dirs = np.sign(np.diff(preds_ext))
    elif da_mode == "absolute":
        actual_dirs = np.sign(actuals - baseline)
        pred_dirs = np.sign(preds - baseline)
    else:
        raise ValueError(f"da_mode must be 'mom' or 'absolute', got {da_mode!r}")

    da_penalty = 50.0 if len(actual_dirs) == 0 else (1 - np.mean(actual_dirs == pred_dirs)) * 100
    return mape + da_weight * da_penalty


def search_best_params_direct_horizon(
    feature_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    price_series: pd.Series,
    periods_per_year: int,
    window_ends: list[pd.Timestamp],
    all_horizons: list[int],
    ext_var_cols: list[str],
    lag_periods: list[int],
    roll_windows: list[int],
    roll_short: int,
    roll_long: int,
    long_window: int,
    cum_a: int,
    cum_b: int,
    is_monthly: bool,
    param_space_fn: ParamSpaceFn,
    fit_predict_fn: "Callable[[dict, pd.DataFrame, pd.Series, pd.DataFrame], float]",
    n_trials: int,
    fixed_params: "dict | None",
    da_weight: float,
    da_mode: str,
    min_train_rows: int = 5,
) -> dict[int, dict]:
    """
    One Optuna study PER HORIZON (unlike search_best_params_recursive's one
    shared study across all steps): each forecast distance gets its own
    tuned model, since safe_features_by_horizon[h] shrinks as h grows and
    longer horizons benefit from different regularization.

    fit_predict_fn(params, X_train, y_train, X_pred) -> single predicted
    value (fits fresh on X_train/y_train, predicts X_pred's one row).

    fixed_params may be a single dict (applied to every horizon) or a dict
    keyed by horizon ({1: {...}, 2: {...}, ...}).
    """
    feature_cols = [c for c in feature_df.columns if c not in (date_col, target_col)]
    feature_cols = drop_multicollinear_features(feature_df, feature_cols, target_col, ext_var_cols=ext_var_cols)
    df_clean = feature_df.dropna(subset=feature_cols).reset_index(drop=True)
    for h in all_horizons:
        df_clean[f"target_h{h}"] = df_clean[target_col].shift(-h)

    safe_features_by_horizon = {
        h: _safe_features_for_horizon(h, feature_cols, ext_var_cols, roll_short, roll_long, long_window)
        for h in all_horizons
    }

    best_params_by_horizon: dict[int, dict] = {}
    for h in all_horizons:
        safe_feats = safe_features_by_horizon[h]
        target_h_col = f"target_h{h}"

        if fixed_params is not None:
            if isinstance(fixed_params, dict) and h in fixed_params:
                best_params_by_horizon[h] = fixed_params[h].copy()
            else:
                best_params_by_horizon[h] = fixed_params.copy()
            continue

        # Everything below is independent of the trial's hyperparameters
        # (the window filter, _build_prediction_row_ext, and the actual/
        # baseline lookup never read `params`) -- precompute it ONCE per
        # (horizon, window) here rather than inside objective(), which
        # study.optimize() calls n_trials times. Pure perf refactor: the
        # per-window skip conditions (train_rows too small, pred_row has a
        # NaN, no actual value) are unchanged, just evaluated once instead
        # of on every trial.
        precomputed_windows = []
        for train_end in window_ends:
            label_safe_cutoff = train_end - _period_offset(h, periods_per_year)
            train_rows = df_clean[
                (df_clean[date_col] <= label_safe_cutoff)
                & (df_clean[target_h_col].notna())
                & (df_clean[safe_feats].notna().all(axis=1))
            ]
            if len(train_rows) < min_train_rows:
                continue

            future_date = train_end + _period_offset(h, periods_per_year)
            pred_row = _build_prediction_row_ext(
                future_date=future_date, train_end=train_end, train_rows=train_rows,
                price_series_upto_train_end=price_series[price_series.index <= train_end],
                safe_features=safe_feats, ext_var_cols=ext_var_cols,
                df_full=feature_df, date_col=date_col,
                lag_periods=lag_periods, roll_windows=roll_windows,
                roll_short=roll_short, roll_long=roll_long, cum_a=cum_a, cum_b=cum_b,
                is_monthly=is_monthly,
            )
            if pred_row.isnull().any(axis=1).values[0]:
                continue

            actual_row = feature_df[feature_df[date_col] == future_date]
            if actual_row.empty:
                continue
            actual = actual_row[target_col].values[0]
            if pd.isna(actual):
                continue

            baseline = float(train_rows[target_col].dropna().iloc[-1])
            precomputed_windows.append((train_rows[safe_feats], train_rows[target_h_col], pred_row, actual, baseline))

        def objective(trial: optuna.Trial, windows=precomputed_windows) -> float:
            params = param_space_fn(trial)
            if params is None:
                return float("inf")

            # MedianPruner (attached to the study below) needs an
            # intermediate value reported per step to compare a trial
            # against the running median of prior trials at the same
            # step -- without trial.report()/should_prune(), a pruner
            # attached to the study has nothing to act on. Reporting the
            # running mean loss after each window lets an obviously bad
            # trial get cut short instead of finishing all of them.
            window_losses = []
            for step_idx, (X_train, y_train, pred_row, actual, baseline) in enumerate(windows):
                predicted = fit_predict_fn(params, X_train, y_train, pred_row)
                loss = _composite_loss([actual], [predicted], baseline, da_weight, da_mode)
                window_losses.append(loss)

                trial.report(np.mean(window_losses), step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(window_losses) if window_losses else float("inf")

        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params_by_horizon[h] = study.best_params.copy()

    return best_params_by_horizon


def run_direct_horizon_backtest(
    commodity: CommodityConfig,
    feature_df: pd.DataFrame,
    date_col: str,
    price_series: pd.Series,
    target_col: str,
    ext_var_cols: list[str],
    technique: str,
    horizon: str,
    horizon_periods: tuple[int, int],
    fit_predict_fn: "Callable[[dict, pd.DataFrame, pd.Series, pd.DataFrame], float]",
    best_params_by_horizon: dict[int, dict],
    lag_periods: list[int],
    roll_windows: list[int],
    roll_short: int,
    roll_long: int,
    long_window: int,
    cum_a: int,
    cum_b: int,
    is_monthly: bool,
    n_windows: int = 15,
    min_train_rows: int = 5,
) -> TechniqueResult:
    """
    RF-ext/LGBM-ext execution: for each window, fits ONE model PER HORIZON
    h=1..m_end (a fresh fit_predict_fn call each time -- direct, not
    recursive) and assembles the total_steps-length forecast array
    _run_windowed_backtest expects. A horizon whose safe-feature row comes
    back incomplete predicts NaN for that step only (see
    _run_windowed_backtest's per-step NaN skip) rather than failing the
    whole window.
    """
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    feature_cols = [c for c in feature_df.columns if c not in (date_col, target_col)]
    feature_cols = drop_multicollinear_features(feature_df, feature_cols, target_col, ext_var_cols=ext_var_cols)
    df_clean = feature_df.dropna(subset=feature_cols).reset_index(drop=True)

    m_start, m_end = horizon_periods
    all_horizons = list(range(1, m_end + 1))
    for h in all_horizons:
        df_clean[f"target_h{h}"] = df_clean[target_col].shift(-h)

    safe_features_by_horizon = {
        h: _safe_features_for_horizon(h, feature_cols, ext_var_cols, roll_short, roll_long, long_window)
        for h in all_horizons
    }

    def forecast_fn(train_series: pd.Series, train_end: pd.Timestamp, total_steps: int):
        preds = []
        for h in range(1, total_steps + 1):
            safe_feats = safe_features_by_horizon[h]
            tgt_col = f"target_h{h}"
            label_safe_cutoff = train_end - _period_offset(h, periods_per_year)
            train_rows = df_clean[
                (df_clean[date_col] <= label_safe_cutoff)
                & (df_clean[tgt_col].notna())
                & (df_clean[safe_feats].notna().all(axis=1))
            ]
            if len(train_rows) < min_train_rows:
                preds.append(np.nan)
                continue

            future_date = train_end + _period_offset(h, periods_per_year)
            pred_row = _build_prediction_row_ext(
                future_date=future_date, train_end=train_end, train_rows=train_rows,
                price_series_upto_train_end=train_series,
                safe_features=safe_feats, ext_var_cols=ext_var_cols,
                df_full=feature_df, date_col=date_col,
                lag_periods=lag_periods, roll_windows=roll_windows,
                roll_short=roll_short, roll_long=roll_long, cum_a=cum_a, cum_b=cum_b,
                is_monthly=is_monthly,
            )
            if pred_row.isnull().any(axis=1).values[0]:
                preds.append(np.nan)
                continue

            predicted = fit_predict_fn(best_params_by_horizon[h], train_rows[safe_feats], train_rows[tgt_col], pred_row)
            preds.append(predicted)
        return np.asarray(preds, dtype=float)

    return _run_windowed_backtest(
        commodity=commodity, price_series=price_series, technique=technique,
        horizon=horizon, horizon_periods=horizon_periods, forecast_fn=forecast_fn,
        best_params=best_params_by_horizon, n_windows=n_windows, min_train_size=1,
        # RF-ext/LGBM-ext number steps relative to the recorded portion, not
        # the same convention as ARIMA/SARIMA/ETS/RF/LightGBM. See
        # _run_windowed_backtest's step_numbering docstring.
        step_numbering="relative",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Deep-learning techniques (TST, TFT): shared, TORCH-FREE helpers.
#
# Both models fit ONE model per window and predict all total_steps in a
# single forward pass -- the same "native multi-step" shape as ARIMA/
# SARIMA/ETS/ARIMAX/SARIMAX, not the recursive or direct-multi-horizon
# families. So tst.py/tft.py reuse run_sliding_window_backtest directly,
# supplying a fit_and_forecast_fn that trains a fresh model per window.
#
# IMPORTANT: this module (_common.py) is imported by every technique,
# including ones that must keep working WITHOUT torch installed (ARIMA, RF,
# LightGBM, etc.). Only torch-free pieces belong here -- WindowNormaliser is
# pure numpy, and build_dl_features is pure pandas/numpy. The actual torch
# Dataset/nn.Module/LightningModule/training-loop code lives in tst.py/
# tft.py themselves, which are only importable where torch is installed.
# ═══════════════════════════════════════════════════════════════════════════


class WindowNormaliser:
    """
    Min-max normaliser fit on the training window only -- never on
    validation/future data, so there's no leakage of future scale into the
    model.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, arr) -> "WindowNormaliser":
        self.min_ = float(np.min(arr))
        self.max_ = float(np.max(arr))
        return self

    def transform(self, arr) -> np.ndarray:
        rng = self.max_ - self.min_
        if rng == 0:
            return np.zeros_like(arr, dtype=float)
        return (np.asarray(arr, dtype=float) - self.min_) / rng

    def inverse_transform(self, arr) -> np.ndarray:
        rng = self.max_ - self.min_
        return np.asarray(arr, dtype=float) * rng + self.min_

    def fit_transform(self, arr) -> np.ndarray:
        return self.fit(arr).transform(arr)


def build_dl_features(price_df: pd.DataFrame, date_col: str, price_col: str) -> pd.DataFrame:
    """
    The minimal feature set every deep-learning technique here uses (TST,
    TFT) -- DL models handle temporal patterns internally, so unlike
    internal_features.py's lag/rolling feature set (built for tree/
    statistical models), this is deliberately just calendar context:
    cyclical month encoding + year. No frequency branching needed --
    month_sin/month_cos degrade gracefully for quarterly data (fewer
    distinct values, still a valid signal) rather than requiring a separate
    quarterly encoding.

    Returns date_col, price_col, month_sin, month_cos, Year -- ready to
    merge with external driver columns for TFT's ext-var mode (Step 8), or
    used as-is for TST's univariate-only mode.
    """
    df = price_df[[date_col, price_col]].copy()
    df = df.dropna(subset=[price_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    months = df[date_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * months / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * months / 12).round(6)
    df["Year"] = df[date_col].dt.year.astype(int)

    return df
