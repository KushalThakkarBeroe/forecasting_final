"""
Univariate ETS (Exponential Smoothing) forecasting via the shared
sliding-window backtest engine in _common.py — same per-horizon backtest
split as arima.py/sarima.py. Seasonal period is period-aware
(periods_per_year), and min_train_size=24, same treatment as sarima.py.

Optuna search space quirk: damped_trend is searched as the STRINGS
'True'/'False', not Python bools, and converted to a real bool only after
the search picks a winner — changing this would change what
TPESampler(seed=42) explores, so it's kept exactly as-is.
"""

from __future__ import annotations

import optuna
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config_loader import CommodityConfig
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest, search_best_params

TECHNIQUE_NAME = "Exponential Smoothing"
MIN_TRAIN_SIZE = 24


def _param_space(trial: optuna.Trial) -> "dict | None":
    trend = trial.suggest_categorical("trend", ["add", "mul", None])
    seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
    damped = trial.suggest_categorical("damped_trend", ["True", "False"])
    init_method = trial.suggest_categorical("initialization_method", ["estimated", "heuristic"])

    damped_bool = damped == "True"
    if damped_bool and trend is None:  # damped_trend requires trend to be set
        return None

    return {
        "trend": trend,
        "seasonal": seasonal,
        "damped_trend": damped_bool,
        "initialization_method": init_method,
    }


def _make_fit_and_forecast(seasonal_periods: int):
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "pd.Series | None":
        try:
            model = ExponentialSmoothing(
                train_series,
                trend=params.get("trend"),
                seasonal=params.get("seasonal"),
                seasonal_periods=seasonal_periods,
                damped_trend=params.get("damped_trend", False),
                initialization_method=params.get("initialization_method", "estimated"),
            ).fit(optimized=True, remove_bias=False)
            return model.forecast(steps)
        except Exception:
            return None

    return _fit_and_forecast


def run_ets(
    commodity: CommodityConfig,
    price_series: pd.Series,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 50,
    fixed_params: "dict | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series: date-indexed, sorted, the commodity's own price.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
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

    # search_best_params returns Optuna's own record of the winning trial
    # (study.best_params) when no fixed_params is given -- that's the raw
    # suggested categorical value, the string 'True'/'False', not what
    # _param_space's return dict converted it to. A bare string is truthy
    # in Python regardless of its text ('False' included), so passing it
    # through unconverted would silently mean "damped_trend=True" every time.
    if isinstance(best_params.get("damped_trend"), str):
        best_params["damped_trend"] = best_params["damped_trend"] == "True"

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
