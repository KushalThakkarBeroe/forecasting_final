"""
VAR and VECM — multivariate joint techniques, via the native multi-step
engine in _common.py (run_sliding_window_backtest) since both fit once per
window and forecast all steps in a single call, same shape as ARIMA/SARIMA/
ETS/ARIMAX/SARIMAX. Hyperparameters are tuned via Optuna, averaged across
all n_windows, same pattern as every other technique.

Requires external driver data — the whole purpose of VAR/VECM is JOINT
multivariate modeling of price + selected drivers together, unlike ARIMAX/
SARIMAX which treat drivers as purely exogenous inputs to a univariate
price equation. A commodity with no adequate drivers (gate_and_impute_
drivers returns none) cannot meaningfully run either technique — raises
rather than silently degrading to a single-variable model. Drivers come
from the same dynamic per-driver lag gate every ext-var technique uses.

Split into two separate techniques (previously one combined "VAR/VECM"
technique that picked internally via a cointegration test and only ever
showed one row, hiding which model actually ran): run_var() and run_vecm()
are now two separate techniques with their own rows in every output.
  - run_var(): always fits VAR on price + drivers. VAR doesn't require a
    cointegrating relationship to be a valid model, so it always runs.
  - run_vecm(): fits VECM ONLY when a Johansen cointegration test
    (statsmodels' select_coint_rank, run once on the fullest available
    training data — same "single representative check" pattern as
    ARIMAX/SARIMAX's order search, not per-window) confirms price and the
    drivers actually share a genuine long-run relationship (rank >= 1).
    VECM's whole design assumes that relationship exists; without it
    there's nothing statistically valid to fit, so run_vecm() raises
    rather than silently substituting VAR or forcing a meaningless fit.
    Deliberately NOT re-tested per window and NOT left to Optuna to pick
    freely — running the test once keeps best_params well-defined (one
    hyperparameter set applied to every window, this project's standard
    convention throughout).
  A commodity may therefore show a VAR row only, or both a VAR row and a
  VECM row, depending on that one cointegration check — never a VECM row
  on its own with no VAR counterpart.

Only the price column's forecast path is recorded — the other jointly-
modeled variables' forecasts are a necessary byproduct of estimation, not
either technique's own output.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

from config_loader import CommodityConfig
from internal_features import DATE_COLUMN_NAME
from external_driver_loader import ExternalDriverData
from external_feature_merge import gate_and_impute_drivers
from _common import PERIODS_PER_YEAR, TechniqueResult, _period_offset, resolve_horizon, run_sliding_window_backtest

TECHNIQUE_NAME_VAR = "VAR"
TECHNIQUE_NAME_VECM = "VECM"
MIN_TRAIN_SIZE = 24
MIN_COINT_TEST_ROWS = 30
PRICE_COL = "__price__"


def _select_model_type(joint_df: pd.DataFrame, k_ar_diff_for_test: int = 1) -> "tuple[str, int]":
    """Returns (model_type, coint_rank) — coint_rank is 0 for "VAR",
    the Johansen-recommended rank for "VECM"."""
    if len(joint_df) < MIN_COINT_TEST_ROWS:
        return "VAR", 0
    try:
        rank_result = select_coint_rank(joint_df, det_order=0, k_ar_diff=k_ar_diff_for_test)
        rank = int(rank_result.rank)
    except Exception:
        return "VAR", 0
    return ("VECM", rank) if rank >= 1 else ("VAR", 0)


def _param_space_var(trial: optuna.Trial) -> dict:
    return {"maxlags": trial.suggest_int("maxlags", 1, 6)}


def _param_space_vecm(trial: optuna.Trial) -> dict:
    return {
        "k_ar_diff": trial.suggest_int("k_ar_diff", 1, 6),
        "deterministic": trial.suggest_categorical("deterministic", ["n", "co", "ci", "lo", "li"]),
    }


def _build_joint_frame(train_series: pd.Series, exog_indexed: pd.DataFrame) -> "pd.DataFrame | None":
    """Aligns train_series (price) with the exog drivers on the same
    index, trimming any leading NaN rows from a driver's own dynamic-lag
    shift (same issue already found/fixed in arimax.py/sarimax.py/tft.py —
    a driver is only valid from (its own last known date + lag) onward)."""
    exog_train = exog_indexed.reindex(train_series.index)
    joint = exog_train.copy()
    joint.insert(0, PRICE_COL, train_series.values)
    joint = joint.dropna()
    if len(joint) < MIN_TRAIN_SIZE:
        return None
    return joint


def _make_fit_and_forecast(exog_indexed: pd.DataFrame, model_type: str, coint_rank: int):
    def _fit_and_forecast(train_series: pd.Series, params: dict, steps: int) -> "np.ndarray | None":
        joint = _build_joint_frame(train_series, exog_indexed)
        if joint is None:
            return None
        try:
            if model_type == "VECM":
                model = VECM(joint, k_ar_diff=params["k_ar_diff"], coint_rank=coint_rank, deterministic=params["deterministic"])
                res = model.fit()
                forecast = res.predict(steps=steps)
            else:
                model = VAR(joint)
                res = model.fit(maxlags=params["maxlags"])
                forecast = res.forecast(joint.values[-res.k_ar :], steps=steps)
            return np.asarray(forecast)[:, 0]
        except Exception:
            return None

    return _fit_and_forecast


def _prepare_exog(commodity: CommodityConfig, external_driver_data: ExternalDriverData, technique_name: str) -> pd.DataFrame:
    ext_df, _decisions = gate_and_impute_drivers(commodity, external_driver_data)
    ext_var_cols = [c for c in ext_df.columns if c != DATE_COLUMN_NAME]
    if not ext_var_cols:
        raise ValueError(
            f"run_{technique_name.lower()}: {commodity.id} has no drivers passing the data-adequacy gate — "
            f"{technique_name} requires at least one external driver to model jointly with price."
        )
    return ext_df.set_index(DATE_COLUMN_NAME)[ext_var_cols]


def _run_backtest(
    commodity: CommodityConfig,
    price_series: pd.Series,
    exog_indexed: pd.DataFrame,
    warmup_periods: int,
    forecast_periods: int,
    model_type: str,
    coint_rank: int,
    param_space_fn,
    technique_name: str,
    n_windows: int,
    n_trials: int,
    fixed_params: "dict | None",
    horizon_label: "str | None",
) -> TechniqueResult:
    horizon_periods, horizon_label = resolve_horizon(warmup_periods, forecast_periods, horizon_label)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]

    last_date = price_series.index.max()
    window_ends = [last_date - _period_offset(i, periods_per_year) for i in range(n_windows - 1, -1, -1)]

    fit_and_forecast = _make_fit_and_forecast(exog_indexed, model_type, coint_rank)

    if fixed_params is not None:
        best_params = fixed_params.copy()
    else:
        def objective(trial: optuna.Trial) -> float:
            params = param_space_fn(trial)
            window_mapes = []
            for train_end in window_ends:
                train_series = price_series[price_series.index <= train_end]
                if len(train_series) < MIN_TRAIN_SIZE:
                    continue
                val_series = train_series.iloc[-forecast_periods:]
                train_only = train_series.iloc[:-forecast_periods]
                forecast = fit_and_forecast(train_only, params, forecast_periods)
                if forecast is None or len(forecast) < forecast_periods:
                    continue
                mape = np.mean(np.abs((val_series.values - forecast[:forecast_periods]) / val_series.values)) * 100
                window_mapes.append(mape)
            return np.mean(window_mapes) if window_mapes else float("inf")

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params.copy()

    return run_sliding_window_backtest(
        commodity=commodity,
        price_series=price_series,
        technique=technique_name,
        horizon=horizon_label,
        horizon_periods=horizon_periods,
        fit_and_forecast_fn=fit_and_forecast,
        best_params=best_params,
        n_windows=n_windows,
        min_train_size=MIN_TRAIN_SIZE,
    )


def run_var(
    commodity: CommodityConfig,
    price_series: pd.Series,
    external_driver_data: ExternalDriverData,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 30,
    fixed_params: "dict | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    Always fits VAR on price + drivers together — no cointegration test
    gates this one, since VAR (unlike VECM) doesn't require a long-run
    equilibrium relationship to be a valid model.

    price_series: date-indexed, sorted, the commodity's own price.
    external_driver_data: Step 1's load_external_drivers(commodity) output
    — required, not optional (see module docstring). Raises ValueError if
    no drivers pass the data-adequacy gate.
    warmup_periods/forecast_periods: any combination — see
    _common.py's resolve_horizon.
    """
    exog_indexed = _prepare_exog(commodity, external_driver_data, TECHNIQUE_NAME_VAR)
    return _run_backtest(
        commodity, price_series, exog_indexed, warmup_periods, forecast_periods,
        "VAR", 0, _param_space_var, TECHNIQUE_NAME_VAR,
        n_windows, n_trials, fixed_params, horizon_label,
    )


def run_vecm(
    commodity: CommodityConfig,
    price_series: pd.Series,
    external_driver_data: ExternalDriverData,
    warmup_periods: int,
    forecast_periods: int,
    n_windows: int = 15,
    n_trials: int = 30,
    fixed_params: "dict | None" = None,
    horizon_label: "str | None" = None,
) -> TechniqueResult:
    """
    Fits VECM ONLY when a Johansen cointegration test confirms price and
    the drivers share a genuine long-run relationship (rank >= 1) — raises
    ValueError otherwise (caller treats this the same as any other fit
    failure: skipped, logged, never crashes the batch). VECM's core
    assumption doesn't hold without cointegration, so there's nothing
    statistically valid to fit; this deliberately does NOT fall back to
    VAR (that's run_var()'s own, separate, always-on row).

    price_series/external_driver_data/warmup_periods/forecast_periods:
    same contract as run_var().
    """
    exog_indexed = _prepare_exog(commodity, external_driver_data, TECHNIQUE_NAME_VECM)
    joint_full = _build_joint_frame(price_series, exog_indexed)
    if joint_full is None:
        raise ValueError(
            f"run_vecm: {commodity.id} has too little overlapping price+driver history "
            f"(< {MIN_TRAIN_SIZE} rows) to test for cointegration."
        )
    model_type, coint_rank = _select_model_type(joint_full)
    if model_type != "VECM":
        raise ValueError(
            f"run_vecm: {commodity.id} shows no cointegrating relationship between price and its "
            "drivers (Johansen test, rank=0) — VECM's core assumption doesn't hold, so it isn't fit "
            "for this commodity."
        )
    return _run_backtest(
        commodity, price_series, exog_indexed, warmup_periods, forecast_periods,
        "VECM", coint_rank, _param_space_vecm, TECHNIQUE_NAME_VECM,
        n_windows, n_trials, fixed_params, horizon_label,
    )
