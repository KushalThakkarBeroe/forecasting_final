"""
Univariate SARIMA forecasting via the shared sliding-window backtest engine
in _common.py — same per-horizon backtest split as arima.py (short/medium/
long each independently tuned).

Seasonal period is period-aware (periods_per_year: 12 monthly, 4
quarterly), not hardcoded to 12 — this project never assumes monthly data.

min_train_size=24 (vs. ARIMA's 15): a seasonal model needs at least 2 full
cycles of history to be trustworthy.
"""

from __future__ import annotations

import warnings

import optuna
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config_loader import CommodityConfig
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest, search_best_params

TECHNIQUE_NAME = "SARIMA"
MIN_TRAIN_SIZE = 24


def _param_space(trial: optuna.Trial) -> "dict | None":
    # Beroe methodology: p/P >= 1 (a p=0 or P=0 model has no trend term,
    # not acceptable for trend-based forecasting), each of p/d/q/P/D/Q
    # capped at 3 individually, and p+d+q / P+D+Q each capped at 5 total
    # (checked separately, non-seasonal vs seasonal).
    p = trial.suggest_int("p", 1, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)
    P = trial.suggest_int("P", 1, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 0, 2)

    if d == 2 and D == 1:  # avoid double-differencing instability
        return None
    if p + d + q > 5 or P + D + Q > 5:
        return None
    return {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q}


def _make_fit_and_forecast(seasonal_periods: int):
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "pd.Series | None":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train_series,
                    order=(params["p"], params["d"], params["q"]),
                    seasonal_order=(params["P"], params["D"], params["Q"], seasonal_periods),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            return model.forecast(steps=steps)
        except Exception:
            return None

    return _fit_and_forecast


def run_sarima(
    commodity: CommodityConfig,
    price_series: pd.Series,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 20,
    fixed_params: "dict | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series: date-indexed, sorted, the commodity's own price.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
    n_trials default 20 (lower than most other techniques' 50) since
    SARIMA is comparatively slow to fit.
    """
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    fit_and_forecast = _make_fit_and_forecast(seasonal_periods=periods_per_year)

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    best_params, _study = search_best_params(
        price_series=price_series,
        window_ends=window_ends,
        forecast_periods=forecast_periods,
        periods_per_year=periods_per_year,
        param_space_fn=_param_space,
        fit_and_forecast_fn=fit_and_forecast,
        n_trials=n_trials,
        fixed_params=fixed_params,
        min_train_size=MIN_TRAIN_SIZE,
    )

    return run_sliding_window_backtest(
        commodity=commodity,
        price_series=price_series,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_and_forecast_fn=fit_and_forecast,
        best_params=best_params,
        n_windows=n_windows,
        min_train_size=MIN_TRAIN_SIZE,
    )
