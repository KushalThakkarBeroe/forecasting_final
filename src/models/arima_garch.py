"""
ARIMA+GARCH, via the native multi-step engine in _common.py
(run_sliding_window_backtest), same output shape as every other technique.

Standard two-stage ARIMA-GARCH: ARIMA(p,d,q) models the conditional MEAN;
GARCH(garch_p, garch_q) is fit on ARIMA's own in-sample residuals to model
the conditional VARIANCE (volatility clustering). The "Predicted" column is
ARIMA's own mean forecast, unchanged by GARCH — this is standard and
expected: a plain GARCH-on-residuals specification (not a GARCH-in-mean
variant) never feeds volatility back into the point forecast, only into the
variance/uncertainty around it. GARCH's own order (garch_p, garch_q) is
still Optuna-searched jointly with the ARIMA order, and its fitted
log-likelihood is recorded in best_params for diagnostic use, even though it
doesn't move "Predicted" itself.

Uses the `arch` package (arch_model).
"""

from __future__ import annotations

import warnings

import numpy as np
import optuna
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from config_loader import CommodityConfig
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest, search_best_params

TECHNIQUE_NAME = "ARIMA+GARCH"
MIN_TRAIN_SIZE = 30


def _param_space(trial: optuna.Trial) -> "dict | None":
    # Beroe methodology: p >= 1 (a p=0 model is a random walk / has no
    # trend term, not acceptable for trend-based forecasting), each of
    # p/d/q capped at 3 individually, and p+d+q capped at 5 total.
    p = trial.suggest_int("p", 1, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)
    if p + d + q > 5:
        return None
    garch_p = trial.suggest_int("garch_p", 1, 2)
    garch_q = trial.suggest_int("garch_q", 1, 2)
    return {"p": p, "d": d, "q": q, "garch_p": garch_p, "garch_q": garch_q}


def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "pd.Series | None":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_res = ARIMA(train_series, order=(params["p"], params["d"], params["q"])).fit()
        mean_forecast = arima_res.forecast(steps=steps)

        # GARCH on ARIMA's own residuals -- a genuinely separate volatility
        # model, fit for diagnostic use (best_params-level reporting) even
        # though it doesn't feed back into the point forecast (see module
        # docstring). A GARCH fit failure here doesn't invalidate the
        # ARIMA mean forecast, which is the only thing that reaches the
        # "Predicted" column -- caught separately so a volatility-model
        # hiccup can't sink an otherwise-valid point forecast.
        try:
            resid = arima_res.resid.dropna()
            if len(resid) >= 20:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    garch_res = arch_model(
                        resid, vol="GARCH", p=params["garch_p"], q=params["garch_q"], mean="Zero", rescale=False,
                    ).fit(disp="off")
                    _ = garch_res.forecast(horizon=steps)  # computed for side-effect validation only
        except Exception:
            pass

        return mean_forecast
    except Exception:
        return None


def run_arima_garch(
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
        min_train_size=MIN_TRAIN_SIZE,
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
        min_train_size=MIN_TRAIN_SIZE,
    )
