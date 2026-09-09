"""
The Robustness Gate: long-term external-variable models (RF+ext, LGBM+ext,
ARIMAX, SARIMAX, TFT ext-var mode, VAR/VECM) are eligible at Long Term only
after passing BOTH tests below (thresholds in config/scoring.yaml's
robustness_gate section, not hardcoded here).

Ablation test: does removing the driver(s) meaningfully hurt accuracy?
Compares the ext-var model's own LT backtest MAPE (already computed, e.g.
by rf_ext.run_rf_ext(..., warmup_periods=6, forecast_periods=12)) against
the matched UNIVARIATE technique's LT MAPE (rf.run_rf for the RF family,
arima.run_arima for the ARIMAX family, etc — whichever univariate sibling
corresponds). Passes if the univariate model's MAPE exceeds the ext-var
model's MAPE by more than min_mape_drop_pp.

Permutation test: is the driver's apparent contribution real signal or
noise? Standard permutation-importance test: independently shuffle each
selected driver's values (breaking any real relationship with price while
preserving each driver's own marginal distribution — see
shuffle_driver_columns), re-run the SAME ext-var technique on the shuffled
data (with best_params fixed, skipping re-tuning — this project's own
existing fixed_params passthrough, not a new mechanism), and repeat many
times to build a null distribution of MAPE. p-value = the fraction of
permuted runs that perform AS WELL OR BETTER than the real (unshuffled)
run; a low p-value means the real driver relationship rarely happens by
chance. Passes if p_value < max_p_value.

Gate-level orchestration (evaluate_robustness_gate) only handles the
STATISTICS — how to re-run a specific technique with shuffled drivers
differs by function signature (rf_ext.run_rf_ext takes feature_df;
arimax.run_arimax/var_vecm.run_var/run_vecm take external_driver_data
directly), so the caller supplies a `rerun_with_shuffled_fn(seed) ->
TechniqueResult` closure. See the module docstring's usage example for how
to build one per technique family — deliberately not a one-size-fits-all
wrapper, since forcing every technique's differently-shaped call through a
single generic wrapper would be more convoluted than the closures
themselves.

Usage example (RF+ext)
-----------------------
    from rf_ext import run_rf_ext
    ext_var_result = run_rf_ext(commodity, price_series, feature_df, price_col,
                                 ext_var_cols, warmup_periods=6, forecast_periods=12)
    univariate_result = run_rf(commodity, price_series, internal_feature_df, price_col,
                                warmup_periods=6, forecast_periods=12)

    def rerun_with_shuffled(seed):
        shuffled_feature_df = shuffle_driver_columns(feature_df, ext_var_cols, DATE_COLUMN_NAME, seed)
        return run_rf_ext(commodity, price_series, shuffled_feature_df, price_col,
                           ext_var_cols, warmup_periods=6, forecast_periods=12,
                           fixed_params=ext_var_result.best_params)

    gate_result = evaluate_robustness_gate(commodity.id, "RF+ext", ext_var_result,
                                            univariate_result, rerun_with_shuffled)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

SCORING_YAML = PROJECT_ROOT / "config" / "scoring.yaml"
DEFAULT_N_PERMUTATIONS = 30


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def shuffle_driver_columns(ext_df: pd.DataFrame, ext_var_cols: list[str], date_col: str, seed: int) -> pd.DataFrame:
    """
    Returns a copy of ext_df with ext_var_cols' VALUES independently
    permuted (each column shuffled on its own — breaks any real
    relationship with price/date and with each other, while preserving
    each driver's own marginal distribution). date_col and every other
    column are left untouched.
    """
    rng = np.random.default_rng(seed)
    shuffled = ext_df.copy()
    for col in ext_var_cols:
        if col == date_col:
            continue
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    return shuffled


def _mean_mape(summary_df: pd.DataFrame) -> float:
    values = summary_df["MAPE (%)"].dropna()
    return float(values.mean()) if len(values) else float("nan")


@dataclass
class AblationResult:
    ext_var_mape: float
    univariate_mape: float
    mape_drop_pp: float  # univariate_mape - ext_var_mape; positive = ext-var model is more accurate
    min_mape_drop_pp: float
    passed: bool


@dataclass
class PermutationResult:
    real_mape: float
    permuted_mapes: list[float] = field(default_factory=list)
    p_value: float = float("nan")
    max_p_value: float = 0.05
    passed: bool = False


@dataclass
class RobustnessGateResult:
    commodity_id: str
    technique: str
    ablation: AblationResult
    permutation: PermutationResult
    passed: bool  # both tests must pass


def run_ablation_test(ext_var_summary_df: pd.DataFrame, univariate_summary_df: pd.DataFrame, min_mape_drop_pp: "float | None" = None) -> AblationResult:
    cfg = _load_yaml(SCORING_YAML)["robustness_gate"]["ablation_test"]
    threshold = min_mape_drop_pp if min_mape_drop_pp is not None else cfg["min_mape_drop_pp"]

    ext_var_mape = _mean_mape(ext_var_summary_df)
    univariate_mape = _mean_mape(univariate_summary_df)
    mape_drop_pp = univariate_mape - ext_var_mape
    passed = bool(mape_drop_pp > threshold) if not (np.isnan(ext_var_mape) or np.isnan(univariate_mape)) else False

    return AblationResult(
        ext_var_mape=ext_var_mape, univariate_mape=univariate_mape,
        mape_drop_pp=mape_drop_pp, min_mape_drop_pp=threshold, passed=passed,
    )


def run_permutation_test(
    real_summary_df: pd.DataFrame,
    rerun_with_shuffled_fn: "Callable[[int], object]",  # seed -> TechniqueResult-like (needs .summary_df)
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    max_p_value: "float | None" = None,
    base_seed: int = 0,
) -> PermutationResult:
    cfg = _load_yaml(SCORING_YAML)["robustness_gate"]["permutation_test"]
    threshold = max_p_value if max_p_value is not None else cfg["max_p_value"]

    real_mape = _mean_mape(real_summary_df)
    permuted_mapes: list[float] = []
    for i in range(n_permutations):
        try:
            permuted_result = rerun_with_shuffled_fn(base_seed + i)
            mape = _mean_mape(permuted_result.summary_df)
            if not np.isnan(mape):
                permuted_mapes.append(mape)
        except Exception as exc:
            logger.warning("permutation %d failed to fit — skipped: %s", i, exc)

    if not permuted_mapes or np.isnan(real_mape):
        return PermutationResult(real_mape=real_mape, permuted_mapes=permuted_mapes, p_value=float("nan"), max_p_value=threshold, passed=False)

    # A permuted run "beats or ties" the real run if its MAPE is AS LOW —
    # i.e. shuffling the driver didn't hurt accuracy, suggesting the real
    # relationship could be chance. Smoothed per Davison & Hinkley (1997)
    # so p never rounds to exactly 0 with a finite permutation count.
    beats_or_ties = sum(1 for m in permuted_mapes if m <= real_mape)
    p_value = (beats_or_ties + 1) / (len(permuted_mapes) + 1)
    passed = p_value < threshold

    return PermutationResult(real_mape=real_mape, permuted_mapes=permuted_mapes, p_value=p_value, max_p_value=threshold, passed=passed)


def evaluate_robustness_gate(
    commodity_id: str,
    technique: str,
    ext_var_result,  # TechniqueResult
    univariate_result,  # TechniqueResult
    rerun_with_shuffled_fn: "Callable[[int], object]",
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> RobustnessGateResult:
    """
    Runs both tests and combines them — a technique is only eligible at
    Long Term (config/scoring.yaml's disqualification_criteria:
    long_term_ext_var_model_without_robustness_gate_pass) if BOTH the
    ablation and permutation tests pass. Short-circuits on an ablation
    failure: the gate's final passed is an AND of both tests, so once
    ablation has already failed nothing permutation returns can change the
    outcome — skip its 30 shuffled-data re-fits entirely in that case
    rather than paying for a result that can't matter.
    """
    ablation = run_ablation_test(ext_var_result.summary_df, univariate_result.summary_df)
    if not ablation.passed:
        return RobustnessGateResult(
            commodity_id=commodity_id, technique=technique,
            ablation=ablation, permutation=PermutationResult(real_mape=float("nan"), passed=False),
            passed=False,
        )
    permutation = run_permutation_test(ext_var_result.summary_df, rerun_with_shuffled_fn, n_permutations=n_permutations)

    return RobustnessGateResult(
        commodity_id=commodity_id, technique=technique,
        ablation=ablation, permutation=permutation,
        passed=ablation.passed and permutation.passed,
    )
