"""
LightGBM + external drivers, via the same direct-multi-horizon engine
rf_ext.py uses (_common.py's search_best_params_direct_horizon /
run_direct_horizon_backtest) — see rf_ext.py's docstring for the shared
architecture.

feature_df must be assemble_features()'s output (dynamic per-driver lag) —
same reasoning as rf_ext.py.

Unlike lgbm.py (univariate), this does NOT use early stopping: fit calls
are plain LGBMRegressor(**params).fit(X, y).

Search space (same range for every horizon): n_estimators [50,200],
max_depth [2,4], num_leaves [4, max(4, 2**max_depth - 1)] (the upper bound
depends on the sampled max_depth), min_child_samples [5,20],
subsample [0.6,1.0], colsample_bytree [0.6,1.0], learning_rate
log-uniform [0.01,0.1]. random_state=42/verbosity=-1/n_jobs=1 fixed, not
searched.
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
    resolve_horizon,
    run_direct_horizon_backtest,
    search_best_params_direct_horizon,
)

TECHNIQUE_NAME = "LightGBM + Ext"
MIN_TRAIN_ROWS = 5

# Same fix as rf_ext.py's _FIXED_PARAMS: fixed params merged directly into
# _fit_predict so every call is reproducible, search trial or final refit.
# n_jobs=1 matters here in particular because this search space tunes
# subsample/colsample_bytree, so LightGBM's own row/column sampling is
# genuinely active (same floating-point-noise risk n_jobs=-1 carries in
# rf.py).
_FIXED_PARAMS = {"random_state": 42, "verbosity": -1, "n_jobs": 1}


def _param_space(trial: optuna.Trial) -> dict:
    max_depth = trial.suggest_int("max_depth", 2, 4)
    max_leaves = max(4, 2 ** max_depth - 1)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 4, max_leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    }


def _fit_predict(params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> float:
    model = lgb.LGBMRegressor(**params, **_FIXED_PARAMS)
    model.fit(X_train, y_train)
    return float(model.predict(X_pred)[0])


def run_lgbm_ext(
    commodity: CommodityConfig,
    price_series: pd.Series,
    feature_df: pd.DataFrame,
    price_col: str,
    ext_var_cols: list[str],
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    # search_best_params_direct_horizon's objective prunes obviously-bad
    # trials early (MedianPruner), so fewer full trials are needed to find
    # the same optimum.
    n_trials: int = 25,
    fixed_params: "dict | None" = None,
    da_weight: float = 0.0,
    da_mode: str = "mom",
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    price_series: date-indexed, sorted, the commodity's own price.
    feature_df: assemble_features(...).df — DATE_COLUMN_NAME + price_col +
    every internal feature + adequate/imputed external driver columns.
    ext_var_cols: the driver column labels within feature_df.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
    da_weight/da_mode: default 0.0/'mom' is a pure MAPE objective.
    """
    if not ext_var_cols:
        # With zero drivers this used to silently fit LightGBM on internal
        # features alone while still showing up labeled "LightGBM +
        # External Variables" -- misleading, since it isn't using any
        # external variable at all. Raise instead, same as var_vecm.py
        # already does, so it's cleanly skipped rather than shown as a
        # disguised duplicate of plain LightGBM.
        raise ValueError(
            f"run_lgbm_ext: {commodity.id} has no drivers passing the data-adequacy gate — "
            "LGBM + External Variables requires at least one external driver, otherwise it's just LightGBM."
        )
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    scaled = derive_scaled_periods(commodity)
    is_monthly = commodity.frequency == "monthly"

    m_start, m_end = horizon_periods
    all_horizons = list(range(1, m_end + 1))

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    best_params_by_horizon = search_best_params_direct_horizon(
        feature_df=feature_df,
        date_col=DATE_COLUMN_NAME,
        target_col=price_col,
        price_series=price_series,
        periods_per_year=periods_per_year,
        window_ends=window_ends,
        all_horizons=all_horizons,
        ext_var_cols=ext_var_cols,
        lag_periods=scaled["lag_periods"],
        roll_windows=scaled["roll_windows"],
        roll_short=scaled["roll_short"],
        roll_long=scaled["roll_long"],
        long_window=scaled["long_window"],
        cum_a=scaled["cum_a"],
        cum_b=scaled["cum_b"],
        is_monthly=is_monthly,
        param_space_fn=_param_space,
        fit_predict_fn=_fit_predict,
        n_trials=n_trials,
        fixed_params=fixed_params,
        da_weight=da_weight,
        da_mode=da_mode,
        min_train_rows=MIN_TRAIN_ROWS,
    )

    return run_direct_horizon_backtest(
        commodity=commodity,
        feature_df=feature_df,
        date_col=DATE_COLUMN_NAME,
        price_series=price_series,
        target_col=price_col,
        ext_var_cols=ext_var_cols,
        technique=TECHNIQUE_NAME,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_predict_fn=_fit_predict,
        best_params_by_horizon=best_params_by_horizon,
        lag_periods=scaled["lag_periods"],
        roll_windows=scaled["roll_windows"],
        roll_short=scaled["roll_short"],
        roll_long=scaled["roll_long"],
        long_window=scaled["long_window"],
        cum_a=scaled["cum_a"],
        cum_b=scaled["cum_b"],
        is_monthly=is_monthly,
        n_windows=n_windows,
        min_train_rows=MIN_TRAIN_ROWS,
    )
