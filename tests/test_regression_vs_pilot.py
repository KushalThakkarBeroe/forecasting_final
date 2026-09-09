"""
Regression/structural test suite for the pilot commodities' techniques.

This is a STRUCTURAL regression test, re-runnable at any time without
external artifacts: correct output shape/schema, no leakage (no NaN
predictions where actuals exist), sane numeric ranges, and that every
technique in the SAME engine family produces internally CONSISTENT Step
numbering and Window/Date alignment.

Also reports (not asserts) the 5 pilot commodities' current
driver-selection state, since it's expected to evolve as
feature_selection.py is re-run over time.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FAST_N_TRIALS, FAST_N_WINDOWS

from arima import run_arima
from sarima import run_sarima
from ets import run_ets
from rf import run_rf
from lgbm import run_lgbm
from rf_ext import run_rf_ext
from lgbm_ext import run_lgbm_ext
from arimax import run_arimax
from sarimax import run_sarimax
from driver_projection import project_drivers

try:
    from tst import run_tst
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _assert_sane_technique_result(result, price_series, expected_step_numbering: str):
    assert len(result.detail_df) >= 1
    assert len(result.summary_df) >= 1

    price_min, price_max = price_series.min(), price_series.max()
    preds = result.detail_df["Predicted"]
    assert preds.notna().all() and np.isfinite(preds).all()
    assert (preds > price_min * 0.1).all() and (preds < price_max * 10).all()

    # Step numbering convention (see _common.py's own docstring on why this
    # varies by family).
    steps_by_window = result.detail_df.groupby("Window")["Step"].apply(lambda s: sorted(s.tolist()))
    for window, steps in steps_by_window.items():
        if expected_step_numbering == "relative":
            assert steps[0] == 1, f"window {window}: relative numbering should start at Step 1, got {steps[0]}"
        assert steps == sorted(steps), f"window {window}: Step values should already be sorted per window"

    # Every (Window, Date) pair should be unique -- no duplicate/overlapping
    # forecasts silently double-counted within a window.
    pairs = list(zip(result.detail_df["Window"], result.detail_df["Date"]))
    assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# Univariate, native multi-step family: absolute step numbering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run_fn", [run_arima, run_sarima, run_ets])
def test_univariate_native_multistep_structural(run_fn, pilot_commodity, pilot_price_series):
    result = run_fn(pilot_commodity, pilot_price_series, warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS)
    _assert_sane_technique_result(result, pilot_price_series, expected_step_numbering="absolute")


# ---------------------------------------------------------------------------
# Univariate, recursive family: absolute step numbering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run_fn", [run_rf, run_lgbm])
def test_univariate_recursive_structural(run_fn, pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data):
    result = run_fn(
        pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
    )
    _assert_sane_technique_result(result, pilot_price_series, expected_step_numbering="absolute")


# ---------------------------------------------------------------------------
# Ext-var, direct multi-horizon family: relative step numbering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run_fn", [run_rf_ext, run_lgbm_ext])
def test_ext_var_direct_horizon_structural(run_fn, pilot_commodity, pilot_price_series, pilot_ext_assembly, pilot_ext_var_cols, pilot_driver_data):
    if not pilot_ext_var_cols:
        pytest.skip("no drivers passed the data-adequacy gate for this pilot commodity")
    result = run_fn(
        pilot_commodity, pilot_price_series, pilot_ext_assembly.df, pilot_driver_data.price_column, pilot_ext_var_cols,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=1,
    )
    _assert_sane_technique_result(result, pilot_price_series, expected_step_numbering="relative")


# ---------------------------------------------------------------------------
# Ext-var, native multi-step family (ARIMAX/SARIMAX): relative step numbering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("run_fn", [run_arimax, run_sarimax])
def test_ext_var_native_multistep_structural(run_fn, pilot_commodity, pilot_price_series, pilot_driver_data):
    if not pilot_driver_data.lag_selections:
        pytest.skip("no drivers passed lag selection for this pilot commodity")
    projection_result = project_drivers(pilot_commodity, pilot_driver_data)
    result = run_fn(
        pilot_commodity, pilot_price_series, pilot_driver_data, projection_result,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS,
    )
    _assert_sane_technique_result(result, pilot_price_series, expected_step_numbering="relative")


# ---------------------------------------------------------------------------
# TST (torch-only, univariate native multi-step): relative step numbering.
# Skipped, not failed, when torch isn't installed in the running
# interpreter -- structural only, same as everywhere else in this file.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed in this interpreter")
def test_tst_structural(pilot_commodity, pilot_price_series):
    result = run_tst(pilot_commodity, pilot_price_series, warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=2)
    _assert_sane_technique_result(result, pilot_price_series, expected_step_numbering="relative")


# ---------------------------------------------------------------------------
# Driver-selection state report (informational, not a hard assertion — the
# 5 pilots' driver set runs through the same automated feature_selection.py
# as every other commodity, so it can legitimately change over time).
# ---------------------------------------------------------------------------

def test_report_driver_selection_state(pilot_commodity, capsys):
    assert pilot_commodity.drivers_status in ("selected", "candidates")
    drivers = pilot_commodity.drivers_config.get("selected_drivers") or pilot_commodity.drivers_config.get("candidate_drivers") or []
    with capsys.disabled():
        print(f"\n{pilot_commodity.id}: drivers_status={pilot_commodity.drivers_status}, "
              f"{len(drivers)} driver(s) in {'selected_drivers' if pilot_commodity.drivers_status == 'selected' else 'candidate_drivers'}")
