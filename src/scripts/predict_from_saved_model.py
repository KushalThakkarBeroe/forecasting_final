"""
Loads one of model_persistence.py's saved top-3 pickles and produces a real
forward forecast from it -- the "item 4" companion the README used to list
as not-yet-built. Never re-fits or re-searches anything; the pickle's own
fitted object/params are used exactly as saved.

Usage (CLI):
  python src/scripts/predict_from_saved_model.py --commodity copper --horizon short --rank 1 --months 3

Or programmatically:
  from predict_from_saved_model import predict_from_saved_model
  csv_path = predict_from_saved_model("copper", "short", rank=1, n_months_ahead=3)

Forecasts purely from the pickle's own frozen state. Price/driver history is
re-read from disk where a technique needs it (the commodity's own
*_ext_var.xlsx may have grown since the model was saved), but always
truncated to <= the pickle's own `trained_through` date before use, so a
prediction always reflects exactly what the saved model actually knew --
never a real value that arrived after it was trained. That's deliberate:
this answers "what would this saved snapshot forecast", not "blend the
saved model with newer real data" (the latter needs a re-fit, which is what
run_horizon.py already does on its own schedule).

Per-technique forecasting mechanics mirror run_horizon.py's own backtest
engine exactly (execution-time only, never search/backtest logic):
  - arima, sarima, ets, arima_garch, sarima_garch: the saved statsmodels
    result object's own multi-step .forecast(steps=n) -- self-contained,
    no external driver, no recursion.
  - markov_switching: markov_switching.py's own _forecast_regime_switching
    (statsmodels has no built-in multi-step .forecast() for this model
    class -- see that module's docstring for why).
  - var, vecm: the saved joint (price+drivers) result object's own
    .forecast()/.predict() -- endogenous, no separate driver projection
    needed (VAR/VECM forecast the drivers forward too, internally).
  - rf, lgbm: _common.py's own _build_future_row, recursive one step at a
    time -- same mechanism run_recursive_backtest uses at execution time.
  - arimax, sarimax: driver_projection.py's project_drivers +
    build_lagged_future_exog for future exogenous driver values, then the
    saved result object's .get_forecast(exog=...).
  - rf_ext, lgbm_ext: direct-horizon -- each requested month has its OWN
    saved model (fitted[step]), so N months ahead calls N *different*
    models once each, never recursing one model N times. A requested step
    beyond what this horizon's search actually tuned is skipped (logged),
    not interpolated or extrapolated.

Results are written to
outputs/predictions/{commodity_id}_{horizon}_rank{rank}_{technique}_{n}m_{YYYY-MM-DD_HH-MM-SS}.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "data", "features", "models", "scoring"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import CommodityConfig, PROJECT_ROOT, load_commodity  # noqa: E402
from internal_features import DATE_COLUMN_NAME, build_internal_features, derive_scaled_periods  # noqa: E402
from external_driver_loader import load_external_drivers  # noqa: E402
from external_feature_merge import assemble_features  # noqa: E402
from driver_projection import build_lagged_future_exog, project_drivers  # noqa: E402
from _common import PERIODS_PER_YEAR, _build_future_row, _build_prediction_row_ext, _period_offset  # noqa: E402
from markov_switching import _forecast_regime_switching  # noqa: E402
from model_persistence import SAVED_MODELS_ROOT, _exog_train_for_x_technique  # noqa: E402

logger = logging.getLogger(__name__)

PREDICTIONS_ROOT = PROJECT_ROOT / "outputs" / "predictions"

_STATSMODELS_FORECAST_TECHNIQUES = {"arima", "sarima", "ets", "arima_garch", "sarima_garch"}
_JOINT_TECHNIQUES = {"var", "vecm"}
_RECURSIVE_TECHNIQUES = {"rf", "lgbm"}
_EXOG_TECHNIQUES = {"arimax", "sarimax"}
_DIRECT_HORIZON_TECHNIQUES = {"rf_ext", "lgbm_ext"}


def _find_saved_pickle(commodity_id: str, horizon_bucket: str, rank: int, saved_models_root: "Path | None") -> Path:
    saved_models_root = saved_models_root if saved_models_root is not None else SAVED_MODELS_ROOT
    directory = saved_models_root / commodity_id / horizon_bucket
    # Filenames end in a sortable YYYY-MM-DD_HH-MM-SS timestamp, so the
    # alphabetically-last match is also the most recently saved one --
    # relevant if a horizon has been re-run more than once.
    matches = sorted(directory.glob(f"{commodity_id}_{horizon_bucket}_rank{rank}_*.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"predict_from_saved_model: no saved rank {rank} model for {commodity_id}/{horizon_bucket} in "
            f"{directory} -- run run_horizon()/run_batch() for this commodity/horizon first."
        )
    return matches[-1]


def _load_bundle(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _future_dates(trained_through: pd.Timestamp, n: int, periods_per_year: int) -> list[pd.Timestamp]:
    return [trained_through + _period_offset(step + 1, periods_per_year) for step in range(n)]


def _forecast_statsmodels(fitted, n: int) -> np.ndarray:
    return np.asarray(fitted.forecast(steps=n))


def _forecast_markov_switching(fitted, best_params: dict, n: int) -> np.ndarray:
    return _forecast_regime_switching(fitted, best_params["order"], best_params["k_regimes"], n)


def _forecast_joint(technique: str, fitted, n: int) -> np.ndarray:
    """VAR/VECM forecast the whole joint (price, drivers) system forward at
    once -- column 0 is always price (var_vecm.py inserts PRICE_COL first
    when it builds the joint frame, both at search and save time)."""
    if technique == "vecm":
        forecast = fitted.predict(steps=n)
    else:
        forecast = fitted.forecast(fitted.model.endog[-fitted.k_ar:], steps=n)
    return np.asarray(forecast)[:, 0]


def _forecast_recursive(fitted_bundle: dict, commodity: CommodityConfig, trained_through: pd.Timestamp, n: int) -> np.ndarray:
    """rf/lgbm: reload price history, recompute the SAME period-scaled
    lag/rolling knobs internal_features.py used at training time, and run
    the same recursive one-step-at-a-time loop run_recursive_backtest uses
    at execution time -- just starting from trained_through instead of a
    backtest window's own train_end."""
    external_driver_data = load_external_drivers(commodity)
    price_series = external_driver_data.df.set_index(DATE_COLUMN_NAME)[external_driver_data.price_column]
    price_series = price_series[price_series.index <= trained_through].dropna().sort_index()

    scaled = derive_scaled_periods(commodity)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    model = fitted_bundle["model"]
    feature_cols = fitted_bundle["feature_cols"]

    history = price_series.copy()
    preds = []
    for step_idx in range(n):
        future_date = trained_through + _period_offset(step_idx + 1, periods_per_year)
        row = _build_future_row(
            history=history, future_date=future_date, feature_cols=feature_cols,
            lag_periods=scaled["lag_periods"], roll_windows=scaled["roll_windows"],
            roll_short=scaled["roll_short"], roll_long=scaled["roll_long"],
            cum_a=scaled["cum_a"], cum_b=scaled["cum_b"], is_monthly=(commodity.frequency == "monthly"),
            long_window=scaled["long_window"], long_min_periods=scaled["long_min_periods"],
        )
        X_future = pd.DataFrame([row], columns=feature_cols)
        predicted = float(model.predict(X_future)[0])
        preds.append(predicted)
        history.loc[future_date] = predicted

    return np.asarray(preds)


def _forecast_exog(technique: str, fitted_bundle: dict, commodity: CommodityConfig, trained_through: pd.Timestamp, n: int) -> np.ndarray:
    """arimax/sarimax: future exog comes from driver_projection.py, same as
    a live (non-backtest) window would use -- as_of=trained_through so the
    projection never sees anything the saved model didn't already know.
    Falls back to the last known training exog row for any future date a
    projection doesn't reach, same fallback _make_fit_and_forecast uses."""
    model = fitted_bundle["model"]
    exog_cols = fitted_bundle["exog_cols"]
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
    future_dates = _future_dates(trained_through, n, periods_per_year)

    if not exog_cols:
        forecast = model.get_forecast(steps=n)
        return np.asarray(forecast.predicted_mean)

    external_driver_data = load_external_drivers(commodity)
    price_series = external_driver_data.df.set_index(DATE_COLUMN_NAME)[external_driver_data.price_column]
    price_series = price_series[price_series.index <= trained_through].dropna().sort_index()
    _train_series, exog_train = _exog_train_for_x_technique(price_series, commodity, external_driver_data, technique.upper())

    projection = project_drivers(commodity, external_driver_data, as_of=trained_through)
    future_exog = build_lagged_future_exog(commodity, external_driver_data, projection, future_dates, as_of=trained_through)

    exog_rows = []
    for date in future_dates:
        if date in future_exog.index and future_exog.loc[date, exog_cols].notna().all():
            exog_rows.append(future_exog.loc[date, exog_cols].values)
        else:
            exog_rows.append(exog_train.iloc[-1][exog_cols].values)

    forecast = model.get_forecast(steps=n, exog=np.array(exog_rows))
    return np.asarray(forecast.predicted_mean)


def _forecast_direct_horizon_ext(
    fitted_by_step: dict, commodity: CommodityConfig, trained_through: pd.Timestamp, n: int,
) -> "tuple[np.ndarray, list[int]]":
    """rf_ext/lgbm_ext: no recursion -- each requested step h uses its own
    saved model (fitted_by_step[h]), predicting from ONE feature row built
    exactly the way run_direct_horizon_backtest's own forecast_fn builds it
    at execution time: external-driver columns frozen at their last known
    value as of trained_through, every other feature read from the most
    recent row that's still "safe" at that step's own staleness horizon
    (train_end - h) -- see _common.py's _build_prediction_row_ext /
    _feature_safety_horizon docstrings for why that anchor is deliberately
    h periods stale, not trained_through itself: it keeps a live prediction
    consistent with how every training example for that same step h was
    itself constructed.
    """
    available_steps = sorted(fitted_by_step.keys())
    requested_steps = list(range(1, n + 1))
    usable_steps = [h for h in requested_steps if h in fitted_by_step]
    skipped_steps = [h for h in requested_steps if h not in fitted_by_step]
    if skipped_steps:
        logger.warning(
            "requested steps %s were never fit for this saved model (only %s available) -- skipping them",
            skipped_steps, available_steps,
        )
    if not usable_steps:
        raise ValueError(
            f"predict_from_saved_model: none of the requested {n} steps were fit for this saved model "
            f"(available steps: {available_steps})"
        )

    external_driver_data = load_external_drivers(commodity)
    price_series = external_driver_data.df.set_index(DATE_COLUMN_NAME)[external_driver_data.price_column]
    price_series = price_series[price_series.index <= trained_through].dropna().sort_index()
    internal_feat = build_internal_features(commodity, external_driver_data.df, external_driver_data.price_column)
    internal_feat = internal_feat[internal_feat[DATE_COLUMN_NAME] <= trained_through].reset_index(drop=True)
    assembly = assemble_features(commodity, internal_feat, external_driver_data)
    ext_feature_df = assembly.df[assembly.df[DATE_COLUMN_NAME] <= trained_through].reset_index(drop=True)
    ext_var_cols = [d.label for d in assembly.decisions if d.action in ("included", "imputed")]

    scaled = derive_scaled_periods(commodity)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]

    preds = []
    kept_steps = []
    for h in usable_steps:
        step_bundle = fitted_by_step[h]
        model, safe_feats = step_bundle["model"], step_bundle["feature_cols"]

        label_safe_cutoff = trained_through - _period_offset(h, periods_per_year)
        train_rows = ext_feature_df[
            (ext_feature_df[DATE_COLUMN_NAME] <= label_safe_cutoff) & (ext_feature_df[safe_feats].notna().all(axis=1))
        ]
        if train_rows.empty:
            logger.warning("step %d: no qualifying historical row to anchor the prediction on -- skipping", h)
            continue

        future_date = trained_through + _period_offset(h, periods_per_year)
        pred_row = _build_prediction_row_ext(
            future_date=future_date, train_end=trained_through, train_rows=train_rows,
            price_series_upto_train_end=price_series, safe_features=safe_feats, ext_var_cols=ext_var_cols,
            df_full=ext_feature_df, date_col=DATE_COLUMN_NAME,
            lag_periods=scaled["lag_periods"], roll_windows=scaled["roll_windows"],
            roll_short=scaled["roll_short"], roll_long=scaled["roll_long"],
            cum_a=scaled["cum_a"], cum_b=scaled["cum_b"], is_monthly=(commodity.frequency == "monthly"),
        )
        if pred_row.isnull().any(axis=1).values[0]:
            logger.warning("step %d: prediction row came back incomplete -- skipping", h)
            continue

        preds.append(float(model.predict(pred_row)[0]))
        kept_steps.append(h)

    if not preds:
        raise ValueError(f"predict_from_saved_model: every requested step's prediction row came back incomplete for this saved model")

    return np.asarray(preds), kept_steps


def predict_from_saved_model(
    commodity_id: str,
    horizon_bucket: str,
    rank: int,
    n_months_ahead: int,
    saved_models_root: "Path | None" = None,
    output_root: "Path | None" = None,
) -> Path:
    """
    Loads saved_models/{commodity_id}/{horizon_bucket}/'s rank-{rank} pickle
    (the most recently saved one, if it's been saved more than once) and
    forecasts n_months_ahead periods past that pickle's own trained_through
    date. Writes one CSV row per forecast step and returns its path.
    """
    if n_months_ahead < 1:
        raise ValueError(f"n_months_ahead must be >= 1, got {n_months_ahead}")
    if rank not in (1, 2, 3):
        raise ValueError(f"rank must be 1, 2, or 3, got {rank}")

    pickle_path = _find_saved_pickle(commodity_id, horizon_bucket, rank, saved_models_root)
    bundle = _load_bundle(pickle_path)
    technique = bundle["technique"]
    fitted = bundle["fitted"]
    trained_through = pd.Timestamp(bundle["trained_through"])
    commodity = load_commodity(commodity_id)
    periods_per_year = PERIODS_PER_YEAR[commodity.frequency]

    if technique in _STATSMODELS_FORECAST_TECHNIQUES:
        values = _forecast_statsmodels(fitted, n_months_ahead)
        steps_used = list(range(1, n_months_ahead + 1))
    elif technique == "markov_switching":
        values = _forecast_markov_switching(fitted, bundle["best_params"], n_months_ahead)
        steps_used = list(range(1, n_months_ahead + 1))
    elif technique in _JOINT_TECHNIQUES:
        values = _forecast_joint(technique, fitted, n_months_ahead)
        steps_used = list(range(1, n_months_ahead + 1))
    elif technique in _RECURSIVE_TECHNIQUES:
        values = _forecast_recursive(fitted, commodity, trained_through, n_months_ahead)
        steps_used = list(range(1, n_months_ahead + 1))
    elif technique in _EXOG_TECHNIQUES:
        values = _forecast_exog(technique, fitted, commodity, trained_through, n_months_ahead)
        steps_used = list(range(1, n_months_ahead + 1))
    elif technique in _DIRECT_HORIZON_TECHNIQUES:
        values, steps_used = _forecast_direct_horizon_ext(fitted, commodity, trained_through, n_months_ahead)
    else:
        raise ValueError(f"predict_from_saved_model: {technique!r} is not a supported technique for prediction")

    future_dates = [trained_through + _period_offset(h, periods_per_year) for h in steps_used]
    result_df = pd.DataFrame({
        "Commodity": commodity_id,
        "Horizon": horizon_bucket,
        "Rank": rank,
        "Technique": technique,
        "Composite Score": bundle["composite_score"],
        "Trained Through": trained_through.date(),
        "Step": steps_used,
        "Date": [d.date() for d in future_dates],
        "Predicted": values,
    })

    output_root = output_root if output_root is not None else PREDICTIONS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = output_root / f"{commodity_id}_{horizon_bucket}_rank{rank}_{technique}_{n_months_ahead}m_{timestamp}.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(
        "%s/%s rank %d (%s): wrote %d-step forecast -> %s",
        commodity_id, horizon_bucket, rank, technique, len(steps_used), out_path,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast forward from a saved top-3 model pickle.")
    parser.add_argument("--commodity", required=True, help="Commodity id, e.g. copper")
    parser.add_argument("--horizon", required=True, choices=["short", "medium", "long"])
    parser.add_argument("--rank", required=True, type=int, choices=[1, 2, 3], help="Which saved rank to use")
    parser.add_argument("--months", required=True, type=int, help="How many periods ahead to forecast")
    args = parser.parse_args()

    out_path = predict_from_saved_model(args.commodity, args.horizon, args.rank, args.months)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
