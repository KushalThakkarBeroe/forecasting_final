"""
Random Forest + external drivers, via _common.py's direct-multi-horizon
engine (search_best_params_direct_horizon / run_direct_horizon_backtest) —
a third forecasting mechanism alongside the native multi-step (ARIMA/
SARIMA/ETS) and recursive (RF/LightGBM univariate) engines: one model is
fit per forecast horizon, with external drivers frozen at each window's
train_end.

feature_df must be the output of features/external_feature_merge.py's
assemble_features() — internal features + data-adequacy-gated external
drivers, using this project's dynamic per-driver lag (there is no
fixed-lag mode anywhere in this codebase).

Search space (same range for every horizon, unlike rf.py's short-vs-
medium/long split): n_estimators [50,300], max_depth [2,6],
min_samples_split [2,10], max_features as a FLOAT fraction [0.4,1.0]
(not rf.py's categorical 'sqrt'/'log2'/None), min_samples_leaf [1,5].
random_state=42 fixed, not searched.
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
    resolve_horizon,
    run_direct_horizon_backtest,
    search_best_params_direct_horizon,
)

TECHNIQUE_NAME = "Random Forest + Ext"
MIN_TRAIN_ROWS = 5

# Optuna's study.best_params only tracks parameters obtained via
# trial.suggest_*() -- a plain literal placed inside _param_space's dict
# (e.g. random_state) would silently be dropped from best_params after the
# search, leaving the real backtest refit unseeded even though every search
# trial itself was seeded. Same fix as rf.py/lgbm.py: a _FIXED_PARAMS
# constant merged into _fit_predict directly, so it's present on every call
# regardless of what's in whatever params dict gets passed in.
_FIXED_PARAMS = {"random_state": 42, "n_jobs": 1}


def _param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }


def _fit_predict(params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_pred: pd.DataFrame) -> float:
    model = RandomForestRegressor(**params, **_FIXED_PARAMS)
    model.fit(X_train, y_train)
    return float(model.predict(X_pred)[0])


def run_rf_ext(
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
    ext_var_cols: the driver column labels within feature_df (e.g.
    [sel.label for sel in external_driver_data.lag_selections] filtered to
    whatever assemble_features actually kept — pass the SAME list used to
    build feature_df, not the pre-gate candidate list).
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
    da_weight/da_mode: default 0.0/'mom' is a pure MAPE objective; the
    composite loss mechanism (blending in Directional Accuracy) is present
    but off unless explicitly requested.
    """
    if not ext_var_cols:
        # With zero drivers this used to silently fit RF on internal
        # features alone while still showing up labeled "RF + External
        # Variables" -- misleading, since it isn't using any external
        # variable at all. Raise instead, same as var_vecm.py already does,
        # so it's cleanly skipped rather than shown as a disguised duplicate
        # of plain RF.
        raise ValueError(
            f"run_rf_ext: {commodity.id} has no drivers passing the data-adequacy gate — "
            "RF + External Variables requires at least one external driver, otherwise it's just RF."
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
