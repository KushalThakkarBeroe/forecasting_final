"""
Univariate Markov-Switching Regression, via the native multi-step engine in
_common.py (run_sliding_window_backtest) with a hand-built multi-step
forecast (see below for why) — one model fit per window, all total_steps
produced in a single call, same output shape as ARIMA/SARIMA/ETS.
Hyperparameters tuned via Optuna, same convention as every other technique.

Uses statsmodels' MarkovAutoregression (Hamilton 1989 regime-switching AR):
the regression relationship's INTERCEPT and VARIANCE switch between
k_regimes hidden states, according to a Markov chain; the AR coefficients
are shared across regimes (switching_ar=False — see below for why).

IMPORTANT: statsmodels does not provide built-in multi-step out-of-sample
forecasting for this model family (unlike ARIMA/SARIMAX/VAR) —
MarkovAutoregressionResults.predict()/.forecast() both raise
NotImplementedError for any out-of-sample start index. This module
implements the standard textbook approach manually: propagate the regime
probability distribution forward through the model's own fitted transition
matrix, and at each step take the probability-weighted average of each
regime's own AR-implied prediction (Hamilton's regime-switching forecast
recursion).

switching_ar=False, not Optuna-tunable: fitting fully separate AR
coefficients per regime is data-hungry and ill-conditioned at the amount of
history typically available (~70-90 monthly rows) and fails to converge in
practice. Fixing it off keeps the model numerically stable; k_regimes/
order/trend and switching_variance remain tunable.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from config_loader import CommodityConfig
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest, search_best_params

TECHNIQUE_NAME = "Markov-Switching"
MIN_TRAIN_SIZE = 30


def _param_space(trial: optuna.Trial) -> "dict | None":
    k_regimes = trial.suggest_int("k_regimes", 2, 3)
    order = trial.suggest_int("order", 1, 4)
    trend = trial.suggest_categorical("trend", ["c", "n"])
    switching_variance = trial.suggest_categorical("switching_variance", [True, False])
    return {"k_regimes": k_regimes, "order": order, "trend": trend, "switching_variance": switching_variance}


def _forecast_regime_switching(res, order: int, k_regimes: int, steps: int) -> np.ndarray:
    """
    Hamilton (1989) regime-switching forecast recursion (see module
    docstring for why this is hand-built).

    At each future step: predicted value = sum over regimes of
    P(regime) * (const[regime] + sum_i ar_i * deviation[t-i]), where
    deviation[t-i] = actual/forecast value at t-i minus the CURRENT
    regime-probability-weighted mean (matching MarkovAutoregression's own
    "deviations from regime mean" formulation, switching_ar=False so ar_i
    is shared across regimes). The regime probability distribution is
    propagated forward one step at a time via the model's fitted
    transition matrix before each subsequent prediction.
    """
    params = res.params
    const = np.array([params[f"const[{k}]"] for k in range(k_regimes)])
    ar_coefs = np.array([params[f"ar.L{i}"] for i in range(1, order + 1)]) if order > 0 else np.array([])

    # regime_transition[i, j] = P(state i at t+1 | state j at t)
    transition = res.regime_transition[:, :, 0] if res.regime_transition.ndim == 3 else res.regime_transition
    # Filtered (not smoothed) probabilities: only uses information up to
    # the last observed point, matching what's actually knowable when
    # forecasting forward from train_end.
    regime_probs = res.filtered_marginal_probabilities.iloc[-1].values.astype(float)

    endog = np.asarray(res.model.endog).flatten()
    history = list(endog[-order:]) if order > 0 else []
    # Regime-weighted mean at each of the last `order` points, needed to
    # compute "deviation from regime mean" for the AR recursion — approximated
    # using each point's OWN filtered regime probabilities where available,
    # falling back to the current distribution at the series edge.
    filtered = res.filtered_marginal_probabilities.values
    recent_probs = filtered[-order:] if order > 0 and len(filtered) >= order else np.tile(regime_probs, (max(order, 1), 1))
    recent_means = recent_probs @ const
    deviations = list(np.array(history) - recent_means) if order > 0 else []

    forecasts = []
    for _ in range(steps):
        regime_mean = float(regime_probs @ const)
        ar_term = float(np.dot(ar_coefs, deviations[-order:][::-1])) if order > 0 else 0.0
        predicted = regime_mean + ar_term
        forecasts.append(predicted)

        deviations.append(predicted - regime_mean)
        regime_probs = transition @ regime_probs

    return np.asarray(forecasts)


def _make_fit_and_forecast():
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "np.ndarray | None":
        try:
            model = MarkovAutoregression(
                train_series, k_regimes=params["k_regimes"], order=params["order"],
                switching_ar=False, switching_variance=params["switching_variance"], trend=params["trend"],
            )
            res = model.fit()
            return _forecast_regime_switching(res, params["order"], params["k_regimes"], steps)
        except Exception:
            return None

    return _fit_and_forecast


def run_markov_switching(
    commodity: CommodityConfig,
    price_series: pd.Series,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 30,
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

    fit_and_forecast = _make_fit_and_forecast()

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
