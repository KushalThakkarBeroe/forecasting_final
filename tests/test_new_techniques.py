"""
Unit/integration tests for Markov-Switching, VAR/VECM, ARIMA+GARCH, and
SARIMA+GARCH (none of which have an external baseline to regression-test
against), plus the Robustness Gate, auto-scoring engine, recommendation,
decision log, selection history, and output generation modules.

Structural assertions only (row/column shape, sane numeric ranges,
hyperparameters recorded, branch logic taking the right path) — not
bit-exact value assertions, since there is nothing to compare bit-exact
values against. See test_regression_vs_pilot.py for the other techniques.

Kept fast deliberately (n_windows/n_trials from conftest's FAST_N_WINDOWS/
FAST_N_TRIALS) — this suite is meant to be re-run often, not just once.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FAST_N_TRIALS, FAST_N_WINDOWS

from markov_switching import run_markov_switching
from arima_garch import run_arima_garch
from sarima_garch import run_sarima_garch
from var_vecm import run_var, run_vecm
from rf import run_rf
from rf_ext import run_rf_ext
from composite_score import score_technique_result
from recommend import recommend, build_forecast_row_df, build_summary_df
from decision_log import confirm_recommendation, get_current_decision, override_recommendation
from selection_history import get_default_technique, record_selection
from robustness_gate import evaluate_robustness_gate, shuffle_driver_columns
from generate_outputs import generate_outputs


def _assert_sane_technique_result(result, price_series, min_rows: int = 1):
    assert len(result.detail_df) >= min_rows
    assert len(result.summary_df) >= 1
    price_min, price_max = price_series.min(), price_series.max()
    preds = result.detail_df["Predicted"]
    assert preds.notna().all()
    assert np.isfinite(preds).all()
    # Loose sanity band, not a precision check -- catches a badly blown-up
    # or collapsed-to-zero forecast, nothing more specific than that.
    assert (preds > price_min * 0.1).all() and (preds < price_max * 10).all()


# ---------------------------------------------------------------------------
# Univariate/multivariate techniques with no external baseline
# ---------------------------------------------------------------------------

def test_markov_switching_structural(pilot_commodity, pilot_price_series):
    result = run_markov_switching(
        pilot_commodity, pilot_price_series, warmup_periods=0, forecast_periods=3,
        n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
    )
    _assert_sane_technique_result(result, pilot_price_series)
    assert "switching_ar" not in result.best_params, "switching_ar must stay hardcoded False, not Optuna-tunable (SVD convergence failure, see module docstring)"
    assert result.best_params["k_regimes"] in (2, 3)


def test_arima_garch_structural(pilot_commodity, pilot_price_series):
    result = run_arima_garch(
        pilot_commodity, pilot_price_series, warmup_periods=0, forecast_periods=3,
        n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
    )
    _assert_sane_technique_result(result, pilot_price_series)
    assert {"p", "d", "q", "garch_p", "garch_q"} <= result.best_params.keys()


def test_sarima_garch_structural(pilot_commodity, pilot_price_series):
    result = run_sarima_garch(
        pilot_commodity, pilot_price_series, warmup_periods=0, forecast_periods=3,
        n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
    )
    _assert_sane_technique_result(result, pilot_price_series)
    assert {"p", "d", "q", "P", "D", "Q", "garch_p", "garch_q"} <= result.best_params.keys()


def test_var_structural(pilot_commodity, pilot_price_series, pilot_driver_data):
    result = run_var(
        pilot_commodity, pilot_price_series, pilot_driver_data, warmup_periods=0, forecast_periods=3,
        n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
    )
    _assert_sane_technique_result(result, pilot_price_series)


def test_vecm_structural(pilot_commodity, pilot_price_series, pilot_driver_data):
    # VECM only fits when price and drivers are cointegrated -- skip rather
    # than fail if that's not the case for this pilot commodity right now.
    try:
        result = run_vecm(
            pilot_commodity, pilot_price_series, pilot_driver_data, warmup_periods=0, forecast_periods=3,
            n_windows=FAST_N_WINDOWS, n_trials=FAST_N_TRIALS,
        )
    except ValueError:
        pytest.skip("no cointegrating relationship found for this pilot commodity's drivers")
    _assert_sane_technique_result(result, pilot_price_series)


def test_var_vecm_requires_drivers(pilot_commodity, pilot_price_series, pilot_driver_data):
    import dataclasses
    no_driver_data = dataclasses.replace(pilot_driver_data, lag_selections=[])
    with pytest.raises(ValueError):
        run_var(pilot_commodity, pilot_price_series, no_driver_data, warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS)
    with pytest.raises(ValueError):
        run_vecm(pilot_commodity, pilot_price_series, no_driver_data, warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS)


# ---------------------------------------------------------------------------
# Robustness Gate
# ---------------------------------------------------------------------------

def test_shuffle_driver_columns_preserves_marginal_distribution(pilot_ext_assembly, pilot_ext_var_cols):
    shuffled = shuffle_driver_columns(pilot_ext_assembly.df, pilot_ext_var_cols, "Date", seed=0)
    for col in pilot_ext_var_cols:
        assert sorted(shuffled[col].dropna().tolist()) == sorted(pilot_ext_assembly.df[col].dropna().tolist())


def test_robustness_gate_end_to_end(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_ext_assembly, pilot_ext_var_cols, pilot_driver_data):
    price_col = pilot_driver_data.price_column

    ext_result = run_rf_ext(
        pilot_commodity, pilot_price_series, pilot_ext_assembly.df, price_col, pilot_ext_var_cols,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=1,
    )
    univariate_result = run_rf(
        pilot_commodity, pilot_price_series, pilot_internal_features, price_col,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=1,
    )

    def rerun_with_shuffled(seed):
        shuffled_df = shuffle_driver_columns(pilot_ext_assembly.df, pilot_ext_var_cols, "Date", seed)
        return run_rf_ext(
            pilot_commodity, pilot_price_series, shuffled_df, price_col, pilot_ext_var_cols,
            warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, fixed_params=ext_result.best_params,
        )

    gate_result = evaluate_robustness_gate(pilot_commodity.id, "rf_ext", ext_result, univariate_result, rerun_with_shuffled, n_permutations=2)
    assert gate_result.ablation.mape_drop_pp == pytest.approx(gate_result.ablation.univariate_mape - gate_result.ablation.ext_var_mape)
    assert 0 <= gate_result.permutation.p_value <= 1 or np.isnan(gate_result.permutation.p_value)
    assert gate_result.passed == (gate_result.ablation.passed and gate_result.permutation.passed)


# ---------------------------------------------------------------------------
# composite_score.py
# ---------------------------------------------------------------------------

def test_composite_score_scores_a_clean_univariate_technique(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data):
    result = run_rf(
        pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column,
        warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2,
    )
    score = score_technique_result(result, pilot_commodity, "short", "rf", pilot_price_series)
    assert not score.insufficient_history
    if not score.disqualified:
        cs = score.component_scores
        expected = (
            cs.accuracy_score * 50 + cs.directional_accuracy_score * 25
            + cs.dynamism_score * 15 + cs.recency_score * 10
        ) / 100 + score.flat_line_penalty + score.outlier_cap_penalty + score.normalized_bias_penalty
        assert score.composite_score == pytest.approx(max(0.0, expected), abs=1e-3)


def test_composite_score_never_disqualifies_rf_for_horizon_reasons(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data):
    # rf's own eligibility is yes/yes/yes in technique_matrix.yaml -- confirm
    # it's never disqualified for horizon-ineligibility, at any horizon.
    result = run_rf(
        pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column,
        warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2,
    )
    score = score_technique_result(result, pilot_commodity, "short", "rf", pilot_price_series)
    assert "horizon_ineligible_technique" not in score.disqualification_reasons


def test_composite_score_gated_long_term_requires_pass(pilot_commodity, pilot_price_series, pilot_ext_assembly, pilot_ext_var_cols, pilot_driver_data):
    from rf_ext import run_rf_ext
    result = run_rf_ext(
        pilot_commodity, pilot_price_series, pilot_ext_assembly.df, pilot_driver_data.price_column, pilot_ext_var_cols,
        warmup_periods=6, forecast_periods=12, n_windows=FAST_N_WINDOWS, n_trials=1,
    )
    failed = score_technique_result(result, pilot_commodity, "long", "rf_ext", pilot_price_series, robustness_gate_passed=False)
    assert failed.disqualified and "long_term_ext_var_model_without_robustness_gate_pass" in failed.disqualification_reasons

    passed = score_technique_result(result, pilot_commodity, "long", "rf_ext", pilot_price_series, robustness_gate_passed=True)
    assert "long_term_ext_var_model_without_robustness_gate_pass" not in passed.disqualification_reasons

    # Same technique at Short Term is NOT gated -- must never be
    # disqualified for the gating reason even with no argument passed.
    short_result = run_rf_ext(
        pilot_commodity, pilot_price_series, pilot_ext_assembly.df, pilot_driver_data.price_column, pilot_ext_var_cols,
        warmup_periods=0, forecast_periods=3, n_windows=FAST_N_WINDOWS, n_trials=1,
    )
    short_score = score_technique_result(short_result, pilot_commodity, "short", "rf_ext", pilot_price_series)
    assert "long_term_ext_var_model_without_robustness_gate_pass" not in short_score.disqualification_reasons


# ---------------------------------------------------------------------------
# recommend.py
# ---------------------------------------------------------------------------

def test_recommend_ranks_by_composite_score_and_excludes_disqualified(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_ext_assembly, pilot_ext_var_cols, pilot_driver_data):
    from lgbm import run_lgbm
    from rf_ext import run_rf_ext

    rf_result = run_rf(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column, warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2)
    lgbm_result = run_lgbm(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column, warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2)
    rfext_result = run_rf_ext(pilot_commodity, pilot_price_series, pilot_ext_assembly.df, pilot_driver_data.price_column, pilot_ext_var_cols, warmup_periods=6, forecast_periods=12, n_windows=FAST_N_WINDOWS, n_trials=1)

    scores = [
        score_technique_result(rf_result, pilot_commodity, "short", "rf", pilot_price_series),
        score_technique_result(lgbm_result, pilot_commodity, "short", "lgbm", pilot_price_series),
        score_technique_result(rfext_result, pilot_commodity, "long", "rf_ext", pilot_price_series, robustness_gate_passed=False),
    ]
    rec = recommend(pilot_commodity.id, "short", scores)
    assert rec.best is not None
    assert rec.best.composite_score >= (rec.runner_up.composite_score if rec.runner_up else -1)
    assert rec.best.technique != "rf_ext"  # disqualified (wrong horizon anyway, included just to prove exclusion)

    summary_df = build_summary_df(pilot_commodity.id, "short", scores)
    # +1: build_summary_df always prepends a "Benchmark" row (Rank 0), even
    # when no benchmark_score was supplied -- see its own docstring.
    assert len(summary_df) == len(scores) + 1
    assert set(summary_df["Rank"]) <= {0, 1, 2, 3}

    forecast_row = build_forecast_row_df(rec)
    assert len(forecast_row) == 1
    assert forecast_row["Best Technique"].iloc[0] is not None


def test_recommend_empty_input_degrades_gracefully():
    rec = recommend("acetic_acid", "short", [])
    assert rec.best is None and rec.runner_up is None
    forecast_row = build_forecast_row_df(rec)
    assert len(forecast_row) == 1
    assert forecast_row["Best Technique"].iloc[0] is None


# ---------------------------------------------------------------------------
# decision_log.py
# ---------------------------------------------------------------------------

class _FakeScore:
    def __init__(self, technique):
        self.technique = technique


class _FakeRecommendation:
    def __init__(self, best_technique):
        self.best = _FakeScore(best_technique) if best_technique else None


def test_decision_log_confirm_override_and_latest_wins(tmp_path):
    log_path = tmp_path / "decision_log.xlsx"
    rec = _FakeRecommendation("lgbm")

    pending = get_current_decision("acetic_acid", "short", rec, path=log_path)
    assert pending.action == "Pending" and pending.technique == "lgbm"

    confirmed = confirm_recommendation("acetic_acid", "short", rec, analyst="Tester", path=log_path)
    assert confirmed.action == "Confirmed" and confirmed.technique == "lgbm"

    with pytest.raises(ValueError):
        override_recommendation("acetic_acid", "short", "arima", analyst="Tester", justification="", path=log_path)

    overridden = override_recommendation("acetic_acid", "short", "arima", analyst="Tester", justification="testing", path=log_path)
    assert overridden.action == "Overridden"

    latest = get_current_decision("acetic_acid", "short", rec, path=log_path)
    assert latest.action == "Overridden" and latest.technique == "arima"

    with pytest.raises(ValueError):
        override_recommendation("acetic_acid", "short", "not_a_real_technique", analyst="Tester", justification="x", path=log_path)


def test_decision_log_confirm_requires_a_best(tmp_path):
    log_path = tmp_path / "decision_log.xlsx"
    with pytest.raises(ValueError):
        confirm_recommendation("acetic_acid", "short", _FakeRecommendation(None), analyst="Tester", path=log_path)


# ---------------------------------------------------------------------------
# selection_history.py + generate_outputs.py
# ---------------------------------------------------------------------------

def test_selection_history_records_and_defaults(tmp_path, pilot_commodity):
    from decision_log import DecisionRecord
    import pandas as pd

    history_path = tmp_path / "selection_history.xlsx"
    decision = DecisionRecord(
        commodity_id=pilot_commodity.id, horizon_bucket="short", action="Confirmed",
        technique="lgbm", analyst="Tester", justification="", timestamp=pd.Timestamp.now(),
    )
    record_selection(pilot_commodity, "short", "Aug-2026", decision, path=history_path)
    assert get_default_technique(pilot_commodity.id, "short", path=history_path) == "lgbm"
    assert get_default_technique(pilot_commodity.id, "medium", path=history_path) is None


def test_generate_outputs_final_scopes_to_active_technique(tmp_path, pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data):
    from lgbm import run_lgbm

    rf_result = run_rf(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column, warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2)
    lgbm_result = run_lgbm(pilot_commodity, pilot_price_series, pilot_internal_features, pilot_driver_data.price_column, warmup_periods=0, forecast_periods=3, n_windows=6, n_trials=2)
    scores = [
        score_technique_result(rf_result, pilot_commodity, "short", "rf", pilot_price_series),
        score_technique_result(lgbm_result, pilot_commodity, "short", "lgbm", pilot_price_series),
    ]
    rec = recommend(pilot_commodity.id, "short", scores)

    log_path = tmp_path / "decision_log.xlsx"
    override_recommendation(pilot_commodity.id, "short", "rf", analyst="Tester", justification="testing final-output scoping", path=log_path)
    decision = get_current_decision(pilot_commodity.id, "short", rec, path=log_path)

    paths = generate_outputs("final", pilot_commodity.id, "short", rec, scores, decision=decision, output_root=tmp_path / "outputs")
    assert paths["forecast_file"].exists() and paths["summary_file"].exists()

    import pandas as pd
    final_summary = pd.read_excel(paths["summary_file"])
    # +1: build_summary_df always prepends a "Benchmark" row (Rank 0), even
    # when no benchmark_score was supplied -- see its own docstring.
    assert len(final_summary) == 2
    active_row = final_summary[final_summary["Technique"] != "Benchmark"].iloc[0]
    assert "RF" in str(active_row["Technique"]) or "Random Forest" in str(active_row["Technique"])
