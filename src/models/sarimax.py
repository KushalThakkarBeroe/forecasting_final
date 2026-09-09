"""
SARIMAX: SARIMA with exogenous (external driver) inputs. Mirrors arimax.py
exactly (see its docstring for the shared architecture — native multi-step
via _common.py's run_sliding_window_backtest, dynamic-lag feature_df +
driver_projection.py for future exog).

One real difference from ARIMAX: order search also grid-searches a
seasonal period via auto_arima(seasonal=True, m=s), picking whichever s
gives the lowest AIC, over candidates [1, periods_per_year] (12 for
monthly, 4 for quarterly — the commodity's own natural full cycle,
period-aware like sarima.py). s=1 (no seasonality) is always one of the
candidates, so the search can also conclude no seasonal component helps.
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

TECHNIQUE_NAME = "SARIMAX"
MIN_TUNE_ROWS = 24
MIN_TRAIN_SIZE = 12


def _find_best_order(
    price_series: pd.Series, exog_indexed: pd.DataFrame, exog_cols: list[str],
    total_steps: int, periods_per_year: int,
    max_p: int, max_d: int, max_q: int, max_P: int, max_D: int, max_Q: int,
    seasonal_periods: list[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Same held-out tuning slice as ARIMAX, but tries each candidate
    seasonal period and keeps whichever auto_arima run has the lowest
    AIC."""
    pullback_periods = round(6 * periods_per_year / 12)
    tune_train_end = price_series.index.max() - _period_offset(pullback_periods, periods_per_year)
    label_safe_cutoff = tune_train_end - _period_offset(total_steps, periods_per_year)
    tune_slice = price_series[price_series.index <= label_safe_cutoff]

    if len(tune_slice) < MIN_TUNE_ROWS:
        return (1, 1, 1), (1, 1, 1, 1)

    exog_tune = exog_indexed.reindex(tune_slice.index)
    exog_values = exog_tune.values if not exog_tune.isnull().all().all() else None

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    for s in seasonal_periods:
        try:
            model = auto_arima(
                tune_slice, exogenous=exog_values, start_p=0, max_p=max_p, start_d=0, max_d=max_d,
                start_q=0, max_q=max_q, start_P=0, max_P=max_P, start_D=0, max_D=max_D,
                start_Q=0, max_Q=max_Q, m=s, seasonal=True, stepwise=True,
                information_criterion="aic", trace=False, error_action="ignore",
                suppress_warnings=True, maxiter=200,
            )
            aic = model.aic()
            if aic < best_aic:
                best_aic = aic
                best_order = model.order
                best_seasonal_order = model.seasonal_order
        except Exception:
            continue

    if best_order is None:
        return (1, 1, 1), (1, 1, 1, 1)
    return best_order, best_seasonal_order


def _make_fit_and_forecast(
    exog_indexed: pd.DataFrame, future_exog_by_train_end: dict[pd.Timestamp, pd.DataFrame],
    exog_cols: list[str], periods_per_year: int,
):
    """See arimax.py: future_exog_by_train_end holds one per-window
    future-exog table, built with as_of=that window's train_end, so a
    window can never see a driver value beyond its own cutoff."""
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "np.ndarray | None":
        try:
            # See arimax.py's identical comment: a driver's own dynamic lag
            # shift leaves it NaN for its first `lag` periods -- trim that
            # leading gap out of this window's training data instead of
            # failing the window outright.
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
                seasonal_order=params["seasonal_order"],
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

            train_end = train_series.index.max()
            # See arimax.py: only future_exog_this_window is used for a
            # future step, to avoid leaking a driver's real future value
            # into an early backtest window.
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


def run_sarimax(
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
    max_P: int = 2,
    max_D: int = 1,
    max_Q: int = 2,
    seasonal_periods: "list[int] | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series/external_driver_data: same contract as arimax.run_arimax
    (exog sourced from gate_and_impute_drivers, not the internal-feature-
    trimmed assemble_features output — see that function's docstring).
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon. seasonal_periods defaults to
    [1, periods_per_year] when not given (see module docstring).
    """
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    if seasonal_periods is None:
        seasonal_periods = [1, periods_per_year]

    total_steps = warmup_periods + forecast_periods

    ext_df, _decisions = gate_and_impute_drivers(commodity, external_driver_data)
    ext_var_cols = [c for c in ext_df.columns if c != DATE_COLUMN_NAME]
    if not ext_var_cols:
        # With zero drivers this used to silently fall through to a
        # plain-SARIMA fit (empty exog) while still showing up labeled
        # "SARIMAX" -- misleading, since it isn't using any external
        # variable at all. Raise instead, same as var_vecm.py already does,
        # so it's cleanly skipped rather than shown as a disguised duplicate
        # of SARIMA.
        raise ValueError(
            f"run_sarimax: {commodity.id} has no drivers passing the data-adequacy gate — "
            "SARIMAX requires at least one external driver, otherwise it's just SARIMA."
        )
    exog_indexed = ext_df.set_index(DATE_COLUMN_NAME)[ext_var_cols]

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    # See arimax.py: driver_projection_result isn't reused for future exog
    # (it's a single projection fit on the full history). Each window gets
    # its own projection and future-exog table instead, both gated to
    # as_of=that window's train_end.
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

    best_order, best_seasonal_order = _find_best_order(
        price_series, exog_indexed, ext_var_cols, total_steps, periods_per_year,
        max_p, max_d, max_q, max_P, max_D, max_Q, seasonal_periods,
    )
    fit_and_forecast = _make_fit_and_forecast(exog_indexed, future_exog_by_train_end, ext_var_cols, periods_per_year)

    return run_sliding_window_backtest(
        commodity=commodity,
        price_series=price_series,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_and_forecast_fn=fit_and_forecast,
        best_params={"order": best_order, "seasonal_order": best_seasonal_order},
        n_windows=n_windows,
        min_train_size=MIN_TRAIN_SIZE,
        # See arimax.py -- step numbers here are relative, not the absolute
        # M-number convention.
        step_numbering="relative",
    )
