"""
Univariate LightGBM forecasting via the shared recursive-forecast engine in
_common.py, shared with rf.py (same _build_future_row / search_best_params_
recursive / run_recursive_backtest, same per-horizon backtest split).

The one real mechanical difference from RF: LightGBM fits WITH early
stopping (eval_set + lgb.early_stopping(20)/lgb.log_evaluation(-1)), RF
does not. _common.py's engine is pluggable (fit_predict_fn / fit_fn)
specifically so this and rf.py can share it:
  - search-time: the same train/val split used for MAPE scoring doubles as
    the early-stopping validation set.
  - execution-time: the last val_size rows of the window's training data
    are held out purely for early stopping (separate from RF's "fit on
    everything"); if that leaves fewer than 10 training rows, it falls back
    to fitting on the full window with no early stopping.

Search space is horizon-dependent, same pattern as rf.py:
  short:       n_estimators [50,1000 step 50],  max_depth [3,7],
               num_leaves [3,25],  min_child_samples [5,20],
               reg_lambda [1.0,10.0]
  medium/long: n_estimators [20,200 step 5],    max_depth [4,6],
               num_leaves [6,7],   min_child_samples [25,40],
               reg_lambda [6.0,12.0]
learning_rate is the same range in both: log-uniform [0.005, 0.07].
random_state=42, n_jobs=1, verbose=-1 are fixed, not searched.

min_train_size thresholds match rf.py's:
  objective, short:        len(train_df) < 20          -> skip window
  objective, medium/long:  len(train_df) < 20+val_size  -> skip window
  execution, all horizons: len(train_df) < 20           -> skip window
"""

from __future__ import annotations

import lightgbm as lgb
import optuna
import pandas as pd

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
)

TECHNIQUE_NAME = "LightGBM"
EXECUTION_MIN_TRAIN_SIZE = 20
OBJECTIVE_MIN_TRAIN_SIZE_SHORT = 20
EARLY_STOPPING_ROUNDS = 20
MIN_FIT_ROWS_FOR_EARLY_STOPPING = 10

# Same reasoning as rf.py's _FIXED_PARAMS: fixed here and merged into
# _fit_predict too, so every fit (search or final) is reproducible even if
# the search space is ever extended to include LightGBM's own
# row/column-subsampling knobs (bagging_fraction/feature_fraction).
_FIXED_PARAMS = {"random_state": 42, "n_jobs": 1, "verbose": -1}
# n_jobs=1 (not -1): LightGBM builds trees via multi-threaded histogram
# summation, and the summation order depends on how many cores actually do
# the work -- a confirmed source of cross-machine result divergence even
# with the same random_state. n_jobs=-1 uses every available core, so two
# machines with different core counts could genuinely build different trees.


def _param_space_short(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.07, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, 25),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
    }


def _param_space_long(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=5),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.07, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 6, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 25, 40),
        "reg_lambda": trial.suggest_float("reg_lambda", 6.0, 12.0),
    }


def _early_stopping_callbacks():
    return [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=-1)]


def _fit_predict(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    model = lgb.LGBMRegressor(**params, **_FIXED_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=_early_stopping_callbacks())
    return model.predict(X_val)


def _fit_model(best_params: dict, X_train: pd.DataFrame, y_train: pd.Series, val_size: int):
    X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
    X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
    model = lgb.LGBMRegressor(**best_params)
    if len(X_tr) >= MIN_FIT_ROWS_FOR_EARLY_STOPPING:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=_early_stopping_callbacks())
    else:
        # Not enough rows to hold out a val split for early stopping ->
        # fit on the full window instead.
        model.fit(X_train, y_train)
    return model


def run_lgbm(
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
    price_col: the price column's name in both price_series and feature_df.
    feature_cols = every feature_df column except DATE_COLUMN_NAME and
    price_col.
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
    # random_state/n_jobs/verbose are fixed, never searched, added after
    # the search resolves.
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
        val_size=forecast_periods,
    )
