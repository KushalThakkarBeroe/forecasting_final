"""
Univariate Random Forest forecasting via the shared recursive-forecast
engine in _common.py — same per-horizon backtest split as arima.py/
sarima.py/ets.py (short vs. medium/long, independently tuned).

Unlike ARIMA/SARIMA/ETS (native multi-step .forecast(steps)), RF predicts
ONE step at a time: fit once per window, then recursively build each future
step's feature row via _common.py's _build_future_row, predict, append the
prediction into history, repeat (search_best_params_recursive +
run_recursive_backtest, not search_best_params + run_sliding_window_
backtest). See _build_future_row's docstring for a known limitation in that
recursive feature reconstruction (long-window/3-year context features are
not available yet at that point and are carried over as-is rather than
recomputed).

Search space is horizon-dependent:
  short:       n_estimators [100,1000 step 50], min_samples_leaf [1,10]
  medium/long: n_estimators [20,200 step 5],     min_samples_leaf [3,10]
max_depth [3,15], min_samples_split [2,10], and max_features
{'sqrt','log2',None} are the same across all three. random_state=42 and
n_jobs=1 are fixed, not searched.

min_train_size differs for the OBJECTIVE (Optuna search) vs. EXECUTION
(the actual backtest), and for short vs. medium/long:
  objective, short:        len(train_df) < 20          -> skip window
  objective, medium/long:  len(train_df) < 20+val_size  -> skip window
  execution, all horizons: len(train_df) < 20           -> skip window
"""

from __future__ import annotations

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config_loader import CommodityConfig
from internal_features import DATE_COLUMN_NAME, derive_scaled_periods
from _common import (
    PERIODS_PER_YEAR,
    TechniqueResult,
    _period_offset,
    drop_multicollinear_features,
    resolve_horizon,
    run_recursive_backtest,
    search_best_params_recursive,
    select_important_internal_features,
)

TECHNIQUE_NAME = "Random Forest"
EXECUTION_MIN_TRAIN_SIZE = 20
OBJECTIVE_MIN_TRAIN_SIZE_SHORT = 20

# RandomForestRegressor is randomized by construction -- every tree's
# bootstrap sample and per-split feature subset depend on random_state.
# Fixing random_state/n_jobs here (used both during the Optuna search and
# the final fit) keeps every RF fit reproducible run to run.
#
# n_jobs=1, not -1: parallel tree-building introduces tiny floating-point
# noise between runs (non-associative summation across threads) even with
# random_state fixed -- negligible on its own, but enough to occasionally
# flip which Optuna trial scores as "best" on a near-tie. n_jobs=1 is fully
# single-threaded and bit-exact reproducible, at the cost of slower
# individual fits (no multi-core use per fit).
_FIXED_PARAMS = {"random_state": 42, "n_jobs": 1}


def _fit_predict(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    return RandomForestRegressor(**params, **_FIXED_PARAMS).fit(X_tr, y_tr).predict(X_val)


def _fit_model(best_params: dict, X_train: pd.DataFrame, y_train: pd.Series, val_size: "int | None"):
    # RF always fits on the full training window -- no early-stopping
    # split, unlike LightGBM. val_size is accepted only to match
    # run_recursive_backtest's fit_fn contract; unused here.
    return RandomForestRegressor(**best_params).fit(X_train, y_train)


def _param_space_short(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _param_space_long(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=5),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def run_rf(
    commodity: CommodityConfig,
    price_series: pd.Series,
    feature_df: pd.DataFrame,
    price_col: str,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 50,
    fixed_params: "dict | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series: date-indexed, sorted, the commodity's own price (same
    series build_internal_features was built from).
    feature_df: build_internal_features(commodity, ...) output — DATE_COLUMN_NAME
    + price_col + every engineered feature, one row per period (warmup rows
    already dropped).
    price_col: the price column's name in both price_series and feature_df
    (e.g. driver_data.price_column). feature_cols = every feature_df column
    except DATE_COLUMN_NAME and price_col.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon. Search space and objective threshold
    follow whether warmup_periods == 0 (short) or > 0 (medium/long), not a
    "short"/"medium"/"long" label.
    """
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    scaled = derive_scaled_periods(commodity)

    feature_cols = [c for c in feature_df.columns if c not in (DATE_COLUMN_NAME, price_col)]
    feature_cols = drop_multicollinear_features(feature_df, feature_cols, price_col)
    feature_cols = select_important_internal_features(feature_df, feature_cols, price_col)

    param_space_fn = _param_space_short if warmup_periods == 0 else _param_space_long
    objective_min_train_size = (
        OBJECTIVE_MIN_TRAIN_SIZE_SHORT if warmup_periods == 0 else OBJECTIVE_MIN_TRAIN_SIZE_SHORT + forecast_periods
    )

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    best_params, _study = search_best_params_recursive(
        feature_df=feature_df,
        date_col=DATE_COLUMN_NAME,
        feature_cols=feature_cols,
        target_col=price_col,
        window_ends=window_ends,
        val_size=forecast_periods,
        param_space_fn=param_space_fn,
        fit_predict_fn=_fit_predict,
        n_trials=n_trials,
        fixed_params=fixed_params,
        min_train_size=objective_min_train_size,
    )
    # random_state/n_jobs are fixed, never searched -- defaulted here too
    # (redundant with _fit_predict's own _FIXED_PARAMS during a live search)
    # since a caller that passes fixed_params directly (e.g. the Robustness
    # Gate's permutation test) skips the search entirely and still needs
    # this defaulting applied.
    for key, value in _FIXED_PARAMS.items():
        best_params.setdefault(key, value)

    return run_recursive_backtest(
        commodity=commodity,
        feature_df=feature_df,
        date_col=DATE_COLUMN_NAME,
        price_series=price_series,
        feature_cols=feature_cols,
        target_col=price_col,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_fn=_fit_model,
        best_params=best_params,
        lag_periods=scaled["lag_periods"],
        roll_windows=scaled["roll_windows"],
        roll_short=scaled["roll_short"],
        roll_long=scaled["roll_long"],
        cum_a=scaled["cum_a"],
        cum_b=scaled["cum_b"],
        is_monthly=(commodity.frequency == "monthly"),
        long_window=scaled["long_window"],
        long_min_periods=scaled["long_min_periods"],
        n_windows=n_windows,
        min_train_size=EXECUTION_MIN_TRAIN_SIZE,
    )
