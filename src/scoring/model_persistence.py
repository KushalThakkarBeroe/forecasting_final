"""
Saves the top-N ranked techniques' final fitted models to disk, so a later,
separate process can reload one and produce a real forecast without
retraining or re-tuning (src/scripts/predict_from_saved_model.py is that
later process).

Deliberately NOT wired into the shared backtest engine (_common.py) or into
any individual technique's run_*() function -- this only runs after a
horizon's scoring is fully finished, as one small, standalone step. Each
technique's own fit routine is duplicated here in miniature (a handful of
lines, not the full search+backtest machinery) rather than modifying the
original technique files at all, so a mistake in this module can only ever
affect what gets saved, never the actual backtest numbers, scores, or
recommendation any commodity's run depends on. Every technique's best_params
are reused exactly as already found (TechniqueResult.best_params) -- nothing
here re-runs Optuna or re-searches anything, it only re-fits once, on all
available history, using settings already proven to work.

Supports all 14 real techniques. The 8 that fit on price history alone
(the two GARCH techniques' volatility stage is diagnostic-only in the
original technique too -- see arima_garch.py/sarima_garch.py's own
comments -- so what actually needs saving to forecast later is the same
mean model plain ARIMA/SARIMA already save): arima, sarima, ets,
arima_garch, sarima_garch, markov_switching, rf, lgbm. And the 6
driver-based ones: var, vecm (jointly model price + drivers, no separate
driver projection needed), arimax, sarimax (price + drivers as exog,
in-sample only here -- future driver values are only needed at predict
time, not at save time), rf_ext, lgbm_ext (direct-horizon: these two
don't have one model to save, they fit a separate small model per forecast
month, so their saved "fitted" value is a dict of {step: model}, not a
single model object).

What bundle["fitted"] actually holds, per technique -- predict_from_saved_
model.py's own dispatch mirrors this exactly:
  - arima, sarima, ets, arima_garch, sarima_garch, var, vecm: the bare
    fitted statsmodels result object -- self-contained, needs nothing else
    from the bundle to forecast forward (VAR/VECM already know their own
    joint price+driver history; a separate exog projection isn't needed).
  - markov_switching: the bare MarkovAutoregressionResults object --
    forecasting it needs bundle["best_params"]["order"]/["k_regimes"] too
    (see markov_switching.py's own _forecast_regime_switching, statsmodels
    has no built-in .forecast() for this model class).
  - rf, lgbm: {"model": ..., "feature_cols": [...]} -- feature_cols is the
    post-multicollinearity-filter list the model was actually trained on.
  - rf_ext, lgbm_ext: {step: {"model": ..., "feature_cols": [...]}} -- one
    entry per forecast month this horizon's search actually tuned.
  - arimax, sarimax: {"model": ..., "exog_cols": [...]} -- exog_cols is
    needed because SARIMAX is fit on a plain array (exog_train.values),
    so the fitted object itself has no memory of which driver each exog
    column was or what order they were in.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from _common import PERIODS_PER_YEAR, _safe_features_for_horizon, drop_multicollinear_features
from config_loader import PROJECT_ROOT, CommodityConfig
from external_driver_loader import ExternalDriverData
from external_feature_merge import gate_and_impute_drivers
from internal_features import DATE_COLUMN_NAME, derive_scaled_periods
from var_vecm import PRICE_COL as VAR_PRICE_COL, _select_model_type

logger = logging.getLogger(__name__)

SAVED_MODELS_ROOT = PROJECT_ROOT / "saved_models"

SUPPORTED_TECHNIQUES = {
    "arima", "sarima", "ets", "arima_garch", "sarima_garch", "markov_switching", "rf", "lgbm",
    "var", "vecm", "arimax", "sarimax", "rf_ext", "lgbm_ext",
}


def _fit_arima(price_series: pd.Series, best_params: dict) -> "object":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(price_series, order=(best_params["p"], best_params["d"], best_params["q"])).fit()


def _fit_sarima(price_series: pd.Series, best_params: dict, periods_per_year: int) -> "object":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            price_series,
            order=(best_params["p"], best_params["d"], best_params["q"]),
            seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], periods_per_year),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)


def _fit_ets(price_series: pd.Series, best_params: dict, periods_per_year: int) -> "object":
    return ExponentialSmoothing(
        price_series,
        trend=best_params.get("trend"), seasonal=best_params.get("seasonal"),
        seasonal_periods=periods_per_year,
        damped_trend=best_params.get("damped_trend", False),
        initialization_method=best_params.get("initialization_method", "estimated"),
    ).fit(optimized=True, remove_bias=False)


def _fit_arima_garch(price_series: pd.Series, best_params: dict) -> "object":
    # Same rationale as arima_garch.py's own _fit_and_forecast: GARCH is
    # fit for diagnostic use only there and never feeds back into the point
    # forecast, so the thing actually worth saving to forecast later is the
    # same ARIMA mean model plain "arima" saves.
    return _fit_arima(price_series, best_params)


def _fit_sarima_garch(price_series: pd.Series, best_params: dict, periods_per_year: int) -> "object":
    return _fit_sarima(price_series, best_params, periods_per_year)


def _fit_markov_switching(price_series: pd.Series, best_params: dict) -> "object":
    model = MarkovAutoregression(
        price_series, k_regimes=best_params["k_regimes"], order=best_params["order"],
        switching_ar=False, switching_variance=best_params["switching_variance"], trend=best_params["trend"],
    )
    return model.fit()


def _feature_cols_for(feature_df: pd.DataFrame, price_col: str) -> list[str]:
    # Same two-step derivation rf.py/lgbm.py themselves use: every column
    # except date/price, then thinned by the same multicollinearity check
    # (config/multicollinearity.yaml) those techniques were actually
    # trained with -- so the saved model's feature list matches its real
    # training, not a naive "every column" guess.
    feature_cols = [c for c in feature_df.columns if c not in (DATE_COLUMN_NAME, price_col)]
    return drop_multicollinear_features(feature_df, feature_cols, price_col)


def _fit_rf(feature_df: pd.DataFrame, price_col: str, best_params: dict) -> dict:
    feature_cols = _feature_cols_for(feature_df, price_col)
    # best_params already carries random_state/n_jobs merged in by rf.py's
    # own run_rf (see rf.py: "for key, value in _FIXED_PARAMS.items():
    # best_params.setdefault(key, value)") before it's stored on
    # TechniqueResult, so no extra merging is needed here.
    model = RandomForestRegressor(**best_params).fit(feature_df[feature_cols], feature_df[price_col])
    return {"model": model, "feature_cols": feature_cols}


def _fit_lgbm(feature_df: pd.DataFrame, price_col: str, best_params: dict) -> dict:
    feature_cols = _feature_cols_for(feature_df, price_col)
    # Same as rf.py: lgbm.py's own run_lgbm already merges random_state/
    # n_jobs/verbose into best_params before storing it on TechniqueResult.
    model = LGBMRegressor(**best_params).fit(feature_df[feature_cols], feature_df[price_col])
    return {"model": model, "feature_cols": feature_cols}


def _joint_price_and_drivers(
    price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, technique_label: str,
) -> pd.DataFrame:
    """Shared by VAR and VECM: same gate_and_impute_drivers -> exog_indexed
    -> joint-with-price construction var_vecm.py's own _prepare_exog/
    _build_joint_frame use, just on all available history (no window
    cutoff) instead of one backtest window's train_end."""
    ext_df, _decisions = gate_and_impute_drivers(commodity, external_driver_data)
    ext_var_cols = [c for c in ext_df.columns if c != DATE_COLUMN_NAME]
    if not ext_var_cols:
        raise ValueError(f"model_persistence: {commodity.id} has no drivers passing the data-adequacy gate for {technique_label}")
    exog_indexed = ext_df.set_index(DATE_COLUMN_NAME)[ext_var_cols]
    joint = exog_indexed.reindex(price_series.index)
    joint.insert(0, VAR_PRICE_COL, price_series.values)
    return joint.dropna()


def _fit_var(price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, best_params: dict) -> "object":
    joint = _joint_price_and_drivers(price_series, commodity, external_driver_data, "VAR")
    return VAR(joint).fit(maxlags=best_params["maxlags"])


def _fit_vecm(price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, best_params: dict) -> "object":
    joint = _joint_price_and_drivers(price_series, commodity, external_driver_data, "VECM")
    model_type, coint_rank = _select_model_type(joint)
    if model_type != "VECM":
        raise ValueError(
            f"model_persistence: {commodity.id} shows no cointegrating relationship between price and its "
            "drivers (Johansen test) -- VECM's core assumption doesn't hold, matching var_vecm.py's own check"
        )
    return VECM(joint, k_ar_diff=best_params["k_ar_diff"], coint_rank=coint_rank, deterministic=best_params["deterministic"]).fit()


def _exog_train_for_x_technique(
    price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, technique_label: str,
) -> "tuple[pd.Series, pd.DataFrame]":
    """Shared by ARIMAX and SARIMAX: same in-sample exog construction
    arimax.py/sarimax.py's own _fit_and_forecast uses (trim price to
    whatever dates the drivers actually have valid values for -- a
    driver's own lag shift leaves it NaN for its first few periods, real
    and expected, not something to impute across). Future driver values
    (needed to forecast forward, not to fit) are deliberately NOT built
    here -- that's the predict script's job, using project_drivers the
    same way arimax.py/sarimax.py do per backtest window."""
    ext_df, _decisions = gate_and_impute_drivers(commodity, external_driver_data)
    ext_var_cols = [c for c in ext_df.columns if c != DATE_COLUMN_NAME]
    if not ext_var_cols:
        raise ValueError(f"model_persistence: {commodity.id} has no drivers passing the data-adequacy gate for {technique_label}")
    exog_indexed = ext_df.set_index(DATE_COLUMN_NAME)[ext_var_cols]
    valid_dates = sorted(price_series.index.intersection(exog_indexed.dropna().index))
    train_series = price_series.loc[valid_dates]
    exog_train = exog_indexed.reindex(train_series.index)
    return train_series, exog_train


def _fit_arimax(price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, best_params: dict) -> dict:
    train_series, exog_train = _exog_train_for_x_technique(price_series, commodity, external_driver_data, "ARIMAX")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series, exog=exog_train.values, order=best_params["order"],
            seasonal_order=best_params.get("seasonal_order", (0, 0, 0, 0)),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
    # exog_cols travels with the fitted model because SARIMAX is fit on
    # exog_train.values (a plain array, not a DataFrame) -- the fitted
    # object itself has no memory of which driver each exog column was, or
    # what order they were in. predict_from_saved_model.py needs this list
    # to rebuild a future exog row in the exact column order the model was
    # trained on.
    return {"model": model, "exog_cols": exog_train.columns.tolist()}


def _fit_sarimax(price_series: pd.Series, commodity: CommodityConfig, external_driver_data: ExternalDriverData, best_params: dict) -> dict:
    train_series, exog_train = _exog_train_for_x_technique(price_series, commodity, external_driver_data, "SARIMAX")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series, exog=exog_train.values, order=best_params["order"],
            seasonal_order=best_params["seasonal_order"],
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
    return {"model": model, "exog_cols": exog_train.columns.tolist()}


def _fit_direct_horizon_ext(
    model_cls, fixed_params: dict, ext_feature_df: pd.DataFrame, price_col: str, ext_var_cols: list[str],
    commodity: CommodityConfig, best_params_by_horizon: "dict[int, dict]",
) -> dict:
    """Shared by RF+ext and LGBM+ext: unlike every other technique here,
    these don't have one model to save -- run_rf_ext/run_lgbm_ext already
    fit a separate small model per forecast month (best_params_by_horizon
    is keyed by how many months out), since each distance benefits from
    its own tuned settings. Returns {step: {"model":..., "feature_cols":...}}
    covering every step this horizon's own search already tuned, using the
    same safe-feature filtering (_safe_features_for_horizon) those
    techniques use to avoid a longer-horizon prediction leaning on a
    feature that would already be too stale by then."""
    scaled = derive_scaled_periods(commodity)
    feature_cols = [c for c in ext_feature_df.columns if c not in (DATE_COLUMN_NAME, price_col)]
    df_clean = ext_feature_df.dropna(subset=feature_cols).reset_index(drop=True)

    models_by_step = {}
    for h, params in best_params_by_horizon.items():
        safe_feats = _safe_features_for_horizon(
            h, feature_cols, ext_var_cols, scaled["roll_short"], scaled["roll_long"], scaled["long_window"],
        )
        target_h_col = f"__target_h{h}__"
        df_clean[target_h_col] = df_clean[price_col].shift(-h)
        rows = df_clean[df_clean[target_h_col].notna() & df_clean[safe_feats].notna().all(axis=1)]
        if rows.empty:
            continue
        model = model_cls(**params, **fixed_params).fit(rows[safe_feats], rows[target_h_col])
        models_by_step[h] = {"model": model, "feature_cols": safe_feats}

    if not models_by_step:
        raise ValueError(f"model_persistence: {commodity.id} had no usable training rows for any forecast step")
    return models_by_step


def _fit_rf_ext(ext_feature_df, price_col, ext_var_cols, commodity, best_params_by_horizon) -> dict:
    return _fit_direct_horizon_ext(
        RandomForestRegressor, {"random_state": 42, "n_jobs": 1},
        ext_feature_df, price_col, ext_var_cols, commodity, best_params_by_horizon,
    )


def _fit_lgbm_ext(ext_feature_df, price_col, ext_var_cols, commodity, best_params_by_horizon) -> dict:
    return _fit_direct_horizon_ext(
        LGBMRegressor, {"random_state": 42, "n_jobs": 1, "verbosity": -1},
        ext_feature_df, price_col, ext_var_cols, commodity, best_params_by_horizon,
    )


def _fit_final_model(
    technique_key: str, best_params: dict, commodity: CommodityConfig,
    price_series: pd.Series, feature_df: "pd.DataFrame | None", price_col: str,
    external_driver_data: "ExternalDriverData | None" = None,
    ext_feature_df: "pd.DataFrame | None" = None, ext_var_cols: "list[str] | None" = None,
) -> "object | dict":
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    if technique_key == "arima":
        return _fit_arima(price_series, best_params)
    if technique_key == "sarima":
        return _fit_sarima(price_series, best_params, periods_per_year)
    if technique_key == "ets":
        return _fit_ets(price_series, best_params, periods_per_year)
    if technique_key == "arima_garch":
        return _fit_arima_garch(price_series, best_params)
    if technique_key == "sarima_garch":
        return _fit_sarima_garch(price_series, best_params, periods_per_year)
    if technique_key == "markov_switching":
        return _fit_markov_switching(price_series, best_params)
    if technique_key == "rf":
        return _fit_rf(feature_df, price_col, best_params)
    if technique_key == "lgbm":
        return _fit_lgbm(feature_df, price_col, best_params)
    if technique_key == "var":
        return _fit_var(price_series, commodity, external_driver_data, best_params)
    if technique_key == "vecm":
        return _fit_vecm(price_series, commodity, external_driver_data, best_params)
    if technique_key == "arimax":
        return _fit_arimax(price_series, commodity, external_driver_data, best_params)
    if technique_key == "sarimax":
        return _fit_sarimax(price_series, commodity, external_driver_data, best_params)
    if technique_key == "rf_ext":
        return _fit_rf_ext(ext_feature_df, price_col, ext_var_cols, commodity, best_params)
    if technique_key == "lgbm_ext":
        return _fit_lgbm_ext(ext_feature_df, price_col, ext_var_cols, commodity, best_params)
    raise ValueError(f"model_persistence: {technique_key!r} is not a supported technique (see SUPPORTED_TECHNIQUES)")


def save_top_n_models(
    commodity: CommodityConfig,
    horizon_bucket: str,
    results_by_technique: dict,
    score_results: list,
    price_series: pd.Series,
    feature_df: "pd.DataFrame | None",
    price_col: str,
    external_driver_data: "ExternalDriverData | None" = None,
    ext_feature_df: "pd.DataFrame | None" = None,
    ext_var_cols: "list[str] | None" = None,
    n: int = 3,
    output_root: "Path | None" = None,
) -> list[Path]:
    """
    Ranks score_results by composite_score, highest first -- same ordering
    as the review sheet's own "Rank" column -- takes the top n, and for
    whichever of those are currently supported (SUPPORTED_TECHNIQUES), fits
    each one fresh on all available history using its own already-found
    best_params, then pickles the result.

    Deliberately NOT filtered by eligible_for_recommendation (the softer
    85%/55% bar recommend() uses to pick the client-facing final_* file) --
    a Long Term horizon in particular can leave every technique short of
    that bar (see e.g. copper) while a top-ranked technique is still real
    and usable, just not confident enough to auto-recommend. Saving is a
    separate, lower-stakes decision than recommending: the person loading a
    saved model later can judge for themselves via the Scoring Detail
    sheet, the same way an analyst already judges eligible_for_
    recommendation vs. disqualified before confirming a final output.

    Still excludes `disqualified` (hard fail -- MAPE<75% AND DA<55%,
    horizon-ineligible, or a Long-Term ext-var model that failed the
    Robustness Gate; composite_score is forced to 0 for these) and
    `insufficient_history` (too few evaluated cycles to have a real score
    at all) -- both mean there's no genuine model worth saving, not just
    one that missed a confidence bar.

    Non-blocking throughout: one technique's save failing is logged and
    skipped, never raised -- this must never be able to fail a horizon run
    that would otherwise have succeeded. An unsupported technique landing
    in the top n does NOT consume one of the n save slots -- ranking
    continues down to the next-best eligible technique instead, so a
    caller asking for 3 saved models gets 3 whenever 3 supported techniques
    exist at all, rather than silently getting fewer just because e.g. the
    #3-ranked technique happens to be one this module doesn't support yet.
    """
    # output_root's real default is looked up here, not bound into the
    # function signature -- a default bound at def-time would freeze
    # whatever SAVED_MODELS_ROOT was when this module was first imported,
    # immune to being pointed elsewhere later (e.g. a test redirecting it
    # to a scratch directory), silently falling back to the real one.
    output_root = output_root if output_root is not None else SAVED_MODELS_ROOT

    ranked = sorted(
        (s for s in score_results if not s.disqualified and not s.insufficient_history),
        key=lambda s: s.composite_score, reverse=True,
    )

    saved_paths: list[Path] = []
    trained_through = price_series.index.max()
    timestamp = datetime.now()
    out_dir = output_root / commodity.id / horizon_bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = 0
    for score in ranked:
        if len(saved_paths) >= n:
            break
        technique_key = score.technique
        if technique_key not in SUPPORTED_TECHNIQUES:
            logger.info(
                "%s/%s: %s not saved (composite score %.1f) -- model saving isn't implemented for this technique yet",
                commodity.id, horizon_bucket, technique_key, score.composite_score,
            )
            continue
        rank += 1
        try:
            best_params = results_by_technique[technique_key].best_params
            fitted = _fit_final_model(
                technique_key, best_params, commodity, price_series, feature_df, price_col,
                external_driver_data=external_driver_data, ext_feature_df=ext_feature_df, ext_var_cols=ext_var_cols,
            )

            bundle = {
                "commodity_id": commodity.id,
                "horizon_bucket": horizon_bucket,
                "technique": technique_key,
                "rank": rank,
                "composite_score": score.composite_score,
                "best_params": best_params,
                "trained_through": trained_through,
                "saved_at": timestamp,
                "fitted": fitted,
            }

            filename = (
                f"{commodity.id}_{horizon_bucket}_rank{rank}_{technique_key}_"
                f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            )
            path = out_dir / filename
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
            saved_paths.append(path)
            logger.info("%s/%s: saved rank %d (%s) -> %s", commodity.id, horizon_bucket, rank, technique_key, path)
        except Exception:
            logger.warning(
                "%s/%s: failed to save rank %d (%s) -- continuing without it",
                commodity.id, horizon_bucket, rank, technique_key, exc_info=True,
            )

    return saved_paths
