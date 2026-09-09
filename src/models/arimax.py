"""
ARIMAX: ARIMA with exogenous (external driver) inputs, via the same native
multi-step engine in _common.py that ARIMA/SARIMA/ETS use
(run_sliding_window_backtest) — unlike RF-ext/LGBM-ext, ARIMAX fits ONE
model per window (not one per horizon) and forecasts all steps natively via
SARIMAX.get_forecast(exog=...), so it slots into the same engine with a
fit_and_forecast_fn closure that threads exogenous data through.

Two notable design choices:
  1. Order selection uses pmdarima.auto_arima on a held-out tuning slice,
     not Optuna (unlike every other technique here).
  2. Training/exog data comes from features/external_feature_merge.py's
     dynamic per-driver lag, and future exogenous values come from
     models/driver_projection.py's build_lagged_future_exog() — there is
     no fixed-lag mode anywhere in this codebase.

Fallback when a step ahead has no real or projected exog value for some
driver: use the training window's own last known exog row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config_loader import CommodityConfig
from internal_features import DATE_COLUMN_NAME
from driver_projection import DriverProjectionResult, build_lagged_future_exog, project_drivers
from external_driver_loader import ExternalDriverData
from external_feature_merge import gate_and_impute_drivers
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest

TECHNIQUE_NAME = "ARIMAX"
MIN_TUNE_ROWS = 24
MIN_TRAIN_SIZE = 12


def _find_best_order(
    price_series: pd.Series, exog_indexed: pd.DataFrame, exog_cols: list[str],
    total_steps: int, periods_per_year: int, max_p: int, max_d: int, max_q: int,
) -> tuple[int, int, int]:
    """Holds out a recent slice (6 months, period-scaled) plus total_steps
    so the order search never sees data a real window wouldn't have had,
    then runs pmdarima's stepwise auto_arima once."""
    pullback_periods = round(6 * periods_per_year / 12)
    tune_train_end = price_series.index.max() - _period_offset(pullback_periods, periods_per_year)
    label_safe_cutoff = tune_train_end - _period_offset(total_steps, periods_per_year)
    tune_slice = price_series[price_series.index <= label_safe_cutoff]

    if len(tune_slice) < MIN_TUNE_ROWS:
        return (1, 1, 1)

    exog_tune = exog_indexed.reindex(tune_slice.index)
    exog_values = exog_tune.values if not exog_tune.isnull().all().all() else None
    try:
        model = auto_arima(
            tune_slice, exogenous=exog_values, start_p=0, max_p=max_p, start_d=0, max_d=max_d,
            start_q=0, max_q=max_q, seasonal=False, stepwise=True, information_criterion="aic",
            trace=False, error_action="ignore", suppress_warnings=True, maxiter=200,
        )
        return model.order
    except Exception:
        return (1, 1, 1)


def _make_fit_and_forecast(
    exog_indexed: pd.DataFrame, future_exog_by_train_end: dict[pd.Timestamp, pd.DataFrame],
    exog_cols: list[str], periods_per_year: int,
):
    """
    future_exog_by_train_end: one future-exog table per backtest window,
    keyed by that window's train_end, each built with
    build_lagged_future_exog(..., as_of=that train_end) so it can never
    contain a driver value the window's own cutoff hadn't reached yet —
    this is what prevents a backtest window from seeing real future driver
    values that hadn't happened yet as of its own train_end.
    """
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "np.ndarray | None":
        try:
            # A driver's OWN dynamic lag shift (Step 1) leaves it NaN for
            # its first `lag` periods -- real, expected, not fixable by
            # imputation (there's nothing to impute from before any data
            # exists). Trim those leading rows out of THIS window's
            # training data rather than failing the window outright; only
            # bail if that leaves too little to fit at all.
            if exog_cols:
                valid_dates = train_series.index.intersection(exog_indexed.dropna().index)
                if len(valid_dates) < MIN_TRAIN_SIZE:
                    return None
                freq = train_series.index.freq
                train_series = train_series.loc[sorted(valid_dates)]
                train_series.index = pd.DatetimeIndex(train_series.index, freq=freq)
            exog_train = exog_indexed.reindex(train_series.index)
            if exog_cols and exog_train.isnull().any().any():
                return None

            model = SARIMAX(
                train_series, exog=exog_train.values if exog_cols else None, order=params["order"],
                seasonal_order=params.get("seasonal_order", (0, 0, 0, 0)),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

            train_end = train_series.index.max()
            # Only future_exog_this_window (built with as_of=train_end, see
            # run_arimax below) is used for a future step, with the last
            # known training value as fallback. Looking up a future step
            # directly in the full real driver history would leak the
            # driver's actual realized future value into an early backtest
            # window instead of a genuine forecast of it.
            future_exog_this_window = future_exog_by_train_end.get(train_end, pd.DataFrame())
            exog_future_rows = []
            for step_idx in range(steps):
                fdate = train_end + _period_offset(step_idx + 1, periods_per_year)
                if fdate in future_exog_this_window.index and future_exog_this_window.loc[fdate, exog_cols].notna().all():
                    exog_future_rows.append(future_exog_this_window.loc[fdate, exog_cols].values)
                else:
                    exog_future_rows.append(exog_train.iloc[-1].values)

            exog_for_forecast = np.array(exog_future_rows) if exog_cols else None
            forecast = model.get_forecast(steps=steps, exog=exog_for_forecast)
            return forecast.predicted_mean.values
        except Exception:
            return None

    return _fit_and_forecast


def run_arimax(
    commodity: CommodityConfig,
    price_series: pd.Series,
    external_driver_data: ExternalDriverData,
    driver_projection_result: DriverProjectionResult,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series: date-indexed, sorted, the commodity's own price.
    external_driver_data: Step 1's load_external_drivers(commodity) output.
    Exog columns come from features/external_feature_merge.py's
    gate_and_impute_drivers (data-adequacy gate) run directly against
    external_driver_data.df — NOT assemble_features()'s merged/trimmed
    output, since that's trimmed to internal_features.py's lag/rolling
    warmup range, which ARIMAX has no use for (it only ever uses price +
    drivers, never lag/rolling features) and which would otherwise cut
    real, available driver history out of the training window.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
    """
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    total_steps = warmup_periods + forecast_periods

    ext_df, _decisions = gate_and_impute_drivers(commodity, external_driver_data)
    ext_var_cols = [c for c in ext_df.columns if c != DATE_COLUMN_NAME]
    if not ext_var_cols:
        # With zero drivers this used to silently fall through to a
        # plain-ARIMA fit (empty exog) while still showing up labeled
        # "ARIMAX" -- misleading, since it isn't using any external variable
        # at all. Raise instead, same as var_vecm.py already does, so it's
        # cleanly skipped rather than shown as a disguised duplicate of
        # ARIMA.
        raise ValueError(
            f"run_arimax: {commodity.id} has no drivers passing the data-adequacy gate — "
            "ARIMAX requires at least one external driver, otherwise it's just ARIMA."
        )
    exog_indexed = ext_df.set_index(DATE_COLUMN_NAME)[ext_var_cols]

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    # The incoming driver_projection_result parameter is not reused for
    # future exog here, since it's a single projection fit on the driver's
    # full history -- reusing it across every window would let an early
    # backtest window see a projection informed by data from after its own
    # train_end. Each window below gets its own projection instead, fit
    # fresh with as_of=that window's train_end (project_drivers), and its
    # own future-exog table gated the same way (build_lagged_future_exog).
    future_exog_by_train_end: dict[pd.Timestamp, pd.DataFrame] = {}
    if ext_var_cols:
        for train_end in window_ends:
            future_dates_this_window = [train_end + _period_offset(step + 1, periods_per_year) for step in range(total_steps)]
            window_projection = project_drivers(commodity, external_driver_data, as_of=train_end)
            future_exog_by_train_end[train_end] = build_lagged_future_exog(
                commodity, external_driver_data, window_projection, future_dates_this_window, as_of=train_end,
            )
    else:
        for train_end in window_ends:
            future_dates_this_window = [train_end + _period_offset(step + 1, periods_per_year) for step in range(total_steps)]
            future_exog_by_train_end[train_end] = pd.DataFrame(index=future_dates_this_window)

    best_order = _find_best_order(price_series, exog_indexed, ext_var_cols, total_steps, periods_per_year, max_p, max_d, max_q)
    fit_and_forecast = _make_fit_and_forecast(exog_indexed, future_exog_by_train_end, ext_var_cols, periods_per_year)

    return run_sliding_window_backtest(
        commodity=commodity,
        price_series=price_series,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_and_forecast_fn=fit_and_forecast,
        best_params={"order": best_order},
        n_windows=n_windows,
        min_train_size=MIN_TRAIN_SIZE,
        # Step numbers here are relative to the recorded portion, not the
        # "absolute M-number" convention ARIMA/SARIMA/ETS use. See
        # _common.py's step_numbering docstring.
        step_numbering="relative",
    )
