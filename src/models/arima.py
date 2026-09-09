"""
Univariate ARIMA forecasting via the shared sliding-window backtest engine
in _common.py.

Short, Medium, and Long Term are backtested independently, each with its
own tuned order: short -> warmup 0, forecast M1-M3; medium -> warmup 3,
forecast M4-M6; long -> warmup 6, forecast M7-M18. Each horizon also uses
its own Directional Accuracy method (config/scoring.yaml's
horizon_overrides).
"""

from __future__ import annotations

import warnings

import optuna
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config_loader import CommodityConfig
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest, search_best_params

TECHNIQUE_NAME = "ARIMA"


def _param_space(trial: optuna.Trial) -> dict | None:
    # Beroe methodology: p >= 1 (a p=0 model is a random walk / has no
    # trend term, not acceptable for trend-based forecasting), each of
    # p/d/q capped at 3 individually, and p+d+q capped at 5 total.
    p = trial.suggest_int("p", 1, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)
    if p + d + q > 5:
        return None
    return {"p": p, "d": d, "q": q}


def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "pd.Series | None":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_series, order=(params["p"], params["d"], params["q"])).fit()
        return model.forecast(steps=steps)
    except Exception:
        return None


def run_arima(
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
    price_series: date-indexed (DATE_COLUMN_NAME), sorted, the commodity's
    own price (e.g. from src/data/external_driver_loader.py's
    ExternalDriverData.df, or internal_features.py's output — this
    technique is univariate, only the price column is used).
    warmup_periods/forecast_periods: any combination — not limited to
    config/horizon_defaults.yaml's short/medium/long buckets (see
    _common.py's resolve_horizon). warmup_periods forecast steps are
    produced and cascaded but not recorded; the next forecast_periods
    steps are recorded/scored.
    horizon_label: optional label for the "Horizon" column/logging
    (defaults to "M{m_start}-M{m_end}").
    """
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    best_params, _study = search_best_params(
        price_series=price_series,
        window_ends=window_ends,
        forecast_periods=forecast_periods,
        periods_per_year=periods_per_year,
        param_space_fn=_param_space,
        fit_and_forecast_fn=_fit_and_forecast,
        n_trials=n_trials,
        fixed_params=fixed_params,
    )

    return run_sliding_window_backtest(
        commodity=commodity,
        price_series=price_series,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_and_forecast_fn=_fit_and_forecast,
        best_params=best_params,
        n_windows=n_windows,
    )
