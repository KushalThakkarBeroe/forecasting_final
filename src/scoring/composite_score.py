"""
The auto-scoring engine. Scores ONE technique's already-computed backtest
result (a `TechniqueResult` from src/models/_common.py, produced by any
run_* model function for one commodity x horizon) into a single composite
score, per config/scoring.yaml — every weight and threshold lives there,
nothing here is hardcoded.

Terminology used throughout: an "evaluation cycle" is one backtest window
(one row of TechniqueResult.summary_df) that has real actuals to compare
against (Months Evaluated > 0).

Every evaluation cycle since config/validation.yaml's backtest_start_date
(a fixed anchor) is used for scoring, growing forward as more months pass —
not capped to a fixed recent-N window. The Accuracy component is computed
row-level (mean of every individual evaluated forecast row's Row Accuracy)
so it matches generate_consolidated_summary.py's "Accuracy by Horizon"
sheet exactly. Directional Accuracy, Forecast Dynamism, Recency Bonus, and
all three penalties are computed per-cycle over that same growing window.
Recency Bonus specifically still needs per-cycle (not row-level) accuracy
numbers to have anything to weight — see _lookback_cycles' own docstring.

Two distinct outcomes are kept separate — config/scoring.yaml's
eligibility_thresholds and disqualification_criteria are deliberately not
merged into one check:
  - `disqualified` (score forced to 0): hard fail — MAPE Accuracy < 75%
    AND DA < 55%, horizon-ineligible technique, or an LT ext-var model
    without a Robustness Gate pass.
  - `eligible_for_recommendation` (softer): MAPE Accuracy >= 85%, DA >= 55%,
    >= 3 evaluated cycles. A model can be scored and NOT disqualified yet
    still fall in the 75-85% MAPE-accuracy gap band — scored, but not
    eligible for the recommendation step to actually pick.

Two config values used below (forecast_dynamism.dynamism_reference_pct,
recency_bonus.recency_bonus_multiplier) and the outlier-cap penalty's bounds
are inferred defaults, not settled values — see the comments in
config/scoring.yaml immediately above each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from config_loader import CommodityConfig, PROJECT_ROOT

SCORING_YAML = PROJECT_ROOT / "config" / "scoring.yaml"
TECHNIQUE_MATRIX_YAML = PROJECT_ROOT / "config" / "technique_matrix.yaml"

HORIZON_BUCKETS = ("short", "medium", "long")


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_technique_eligibility(technique_key: str) -> dict:
    techniques = _load_yaml(TECHNIQUE_MATRIX_YAML)["techniques"]
    for entry in techniques:
        if entry["name"] == technique_key:
            return entry["eligibility"]
    raise ValueError(f"composite_score: technique_key {technique_key!r} not found in {TECHNIQUE_MATRIX_YAML}")


@dataclass
class ComponentScores:
    accuracy_score: float               # 0-100, row-level mean(100 - MAPE) since backtest_start_date -- see module docstring
    directional_accuracy_score: float   # 0-100, horizon-appropriate DA mean over lookback cycles
    dynamism_score: float               # 0-100, normalized forecast StdDev scaled to dynamism_reference_pct
    recency_score: float                # 0-100, recency-weighted variant of accuracy_score


@dataclass
class ScoreResult:
    commodity_id: str
    technique: str
    horizon_bucket: str
    evaluated_cycles: int
    insufficient_history: bool                 # < min_evaluated_cycles -> cannot be scored at all
    disqualified: bool                         # hard fail -> composite_score forced to 0
    disqualification_reasons: list[str] = field(default_factory=list)
    eligible_for_recommendation: bool = False
    ineligibility_reasons: list[str] = field(default_factory=list)
    component_scores: "ComponentScores | None" = None
    flat_line_penalty: float = 0.0
    outlier_cap_penalty: float = 0.0
    normalized_bias_penalty: float = 0.0
    mean_mpe_pct: float = float("nan")
    composite_score: float = 0.0
    confidence_tier: "str | None" = None        # "HIGH" | "MEDIUM" | "LOW", None if unscored/disqualified


def _lookback_cycles(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Every evaluation cycle (backtest window with real actuals) since
    backtest_start_date -- not capped to the most recent N. summary_df's own
    window range is already anchored to backtest_start_date
    (run_horizon.py's n_windows), so simply not capping here is sufficient
    to get "every cycle since the anchor date, growing forward" -- no
    separate anchor logic needed in this function.

    Still used for Directional Accuracy, Forecast Dynamism, Recency Bonus,
    and the three penalties -- NOT for the Accuracy component anymore (that's
    row-level now, straight off result.detail_df, see score_technique_result).
    Recency Bonus in particular needs these per-cycle values, not a single
    pooled number, to have anything meaningful to weight "most recent cycles
    higher" against.

    A technique that failed to fit in EVERY window (e.g. Markov-Switching
    failing to converge on every backtest window for a given commodity --
    a real statsmodels MLE failure, not a data-length issue) never adds a
    single record to _run_windowed_backtest's summary_records, so
    summary_df comes back with 0 rows AND 0 columns (pd.DataFrame([])),
    not just 0 rows -- indexing "Months Evaluated" on that raises KeyError
    instead of reaching the insufficient_history path below, which is
    already built to handle "0 evaluated cycles" correctly.
    """
    if summary_df.empty:
        return summary_df
    evaluated = summary_df[summary_df["Months Evaluated"] > 0].copy()
    return evaluated.sort_values("sort_order")


def _clip_0_100(values: "pd.Series | np.ndarray") -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)


def _dynamism_inputs(detail_df: pd.DataFrame, windows: "list[int]") -> "tuple[float, list[float]]":
    """
    Returns (mean normalized-StdDev-%, per-window normalized-StdDev-% list)
    across the given Window numbers, using EVERY recorded Predicted value in
    each window (not just rows with an Actual yet) -- Dynamism measures the
    shape of the forecast curve itself, not forecast accuracy.
    """
    per_window_pct = []
    for w in windows:
        preds = detail_df.loc[detail_df["Window"] == w, "Predicted"]
        if len(preds) < 2:
            continue
        mean_pred = preds.mean()
        if mean_pred == 0 or pd.isna(mean_pred):
            continue
        per_window_pct.append(float(preds.std() / abs(mean_pred) * 100))
    mean_pct = float(np.mean(per_window_pct)) if per_window_pct else float("nan")
    return mean_pct, per_window_pct


def score_technique_result(
    result,  # TechniqueResult (src.models._common.TechniqueResult) -- not imported to keep this module import-light
    commodity: CommodityConfig,
    horizon_bucket: str,
    technique_key: str,
    price_series: pd.Series,
    robustness_gate_passed: "bool | None" = None,
    apply_disqualification: bool = True,
) -> ScoreResult:
    """
    result: one technique's TechniqueResult for this commodity x horizon_bucket.
    horizon_bucket: "short" | "medium" | "long" -- must be supplied explicitly
        (TechniqueResult.horizon is a free-form label by convention, not
        enforced -- see _common.py's own docstring on that field).
    technique_key: matches a `name` entry in config/technique_matrix.yaml
        (e.g. "rf_ext", "arimax", "arima_garch").
    price_series: the commodity's own full price history (date-indexed) --
        used only for the outlier-cap penalty's sane-range bounds.
    robustness_gate_passed: required (non-None) only for techniques whose
        technique_matrix.yaml eligibility at this horizon is "gated" -- pass
        `RobustnessGateResult.passed` from src/scoring/robustness_gate.py.
    apply_disqualification: when False, `disqualified`/`disqualification_
        reasons` are still computed and returned, but composite_score is
        NOT forced to 0 on disqualification -- used by score_benchmark_
        result so Benchmark's Score always reflects the raw formula.
        Benchmark is a reference point, not a recommendable technique, so
        the hard-fail gate doesn't apply to its own displayed score the way
        it does for real techniques.
    """
    if horizon_bucket not in HORIZON_BUCKETS:
        raise ValueError(f"horizon_bucket must be one of {HORIZON_BUCKETS}, got {horizon_bucket!r}")

    cfg = _load_yaml(SCORING_YAML)
    eligibility = _load_technique_eligibility(technique_key)[horizon_bucket]

    disqualification_reasons: list[str] = []
    if eligibility == "no":
        disqualification_reasons.append("horizon_ineligible_technique")
    elif eligibility == "gated" and horizon_bucket == "long":
        if not robustness_gate_passed:
            disqualification_reasons.append("long_term_ext_var_model_without_robustness_gate_pass")

    cycles = _lookback_cycles(result.summary_df)
    evaluated_cycles = len(cycles)

    min_evaluated_cycles = cfg["eligibility_thresholds"]["min_evaluated_cycles"]
    insufficient_history = evaluated_cycles < min_evaluated_cycles
    if insufficient_history:
        return ScoreResult(
            commodity_id=commodity.id, technique=technique_key, horizon_bucket=horizon_bucket,
            evaluated_cycles=evaluated_cycles, insufficient_history=True,
            disqualified=bool(disqualification_reasons), disqualification_reasons=disqualification_reasons,
            eligible_for_recommendation=False, ineligibility_reasons=["insufficient_evaluated_cycles"],
            composite_score=0.0, confidence_tier=None,
        )

    # --- Accuracy (50%): row-level mean of Row Accuracy across every
    # evaluated forecast row since backtest_start_date -- matches generate_
    # consolidated_summary.py's "Accuracy by Horizon" sheet exactly (same
    # column, same plain mean, no per-window averaging step in between).
    # `cycles` (summary_df's per-window pre-averaged MAPE) is only used
    # below, for Recency Bonus's own per-cycle weighting, which needs
    # per-cycle numbers to have anything to weight.
    accuracy_score = float(np.mean(_clip_0_100(result.detail_df["Row Accuracy"].dropna().to_numpy())))

    # Per-cycle accuracy (window-level, NOT row-level) -- Recency Bonus only.
    accuracy_per_cycle = _clip_0_100(100 - cycles["MAPE (%)"].to_numpy())

    # --- Directional Accuracy (25%): horizon-appropriate method ---
    da_method = cfg["horizon_overrides"]["long_term" if horizon_bucket == "long" else "short_medium_term"]["directional_accuracy_method"]
    da_col = "Abs Directional Accuracy (%)" if da_method == "fixed_anchor" else "Directional Accuracy (%)"
    da_score = float(np.mean(_clip_0_100(cycles[da_col].to_numpy())))

    # --- Forecast Dynamism (15%) ---
    dynamism_ref_pct = cfg["forecast_dynamism"]["dynamism_reference_pct"]
    mean_stddev_pct, _per_window_stddev_pct = _dynamism_inputs(result.detail_df, cycles["Window"].tolist())
    dynamism_score = float(np.clip(mean_stddev_pct / dynamism_ref_pct * 100, 0.0, 100.0)) if not np.isnan(mean_stddev_pct) else 0.0

    # --- Recency Bonus (10%): accuracy_per_cycle re-averaged with the most
    # recent `recency_bonus_cycle_count` cycles up-weighted ---
    recency_cfg = cfg["recency_bonus"]
    n_recent = min(recency_cfg["recency_bonus_cycle_count"], evaluated_cycles)
    weights = np.ones(evaluated_cycles)
    if n_recent > 0:
        weights[-n_recent:] = recency_cfg["recency_bonus_multiplier"]
    recency_score = float(np.average(accuracy_per_cycle, weights=weights))

    component_scores = ComponentScores(
        accuracy_score=accuracy_score, directional_accuracy_score=da_score,
        dynamism_score=dynamism_score, recency_score=recency_score,
    )

    weights_cfg = cfg["criterion_weights"]
    composite_raw = (
        accuracy_score * weights_cfg["mape_accuracy_pct"]
        + da_score * weights_cfg["directional_accuracy_pct"]
        + dynamism_score * weights_cfg["forecast_dynamism_pct"]
        + recency_score * weights_cfg["recency_bonus_pct"]
    ) / 100.0

    # --- Penalties ---
    penalty_cfg = cfg["penalty_flags"]
    flat_line_penalty = 0.0
    if horizon_bucket == "long" and not np.isnan(mean_stddev_pct) and mean_stddev_pct < penalty_cfg["flat_stddev_threshold_pct"]:
        flat_line_penalty = float(penalty_cfg["flat_line_lt_penalty_points"])

    price_min, price_max = float(price_series.min()), float(price_series.max())
    lo = price_min * penalty_cfg["outlier_min_multiplier"]
    hi = price_max * penalty_cfg["outlier_max_multiplier"]
    cycle_windows = set(cycles["Window"].tolist())
    lookback_preds = result.detail_df.loc[result.detail_df["Window"].isin(cycle_windows), "Predicted"]
    outlier_cap_penalty = float(penalty_cfg["outlier_cap_penalty_points"]) if ((lookback_preds < lo) | (lookback_preds > hi)).any() else 0.0

    bias_cfg = cfg["normalized_bias_penalty"]
    mean_mpe_pct = float(cycles["MPE (%)"].mean())
    normalized_bias_penalty = float(bias_cfg["penalty_points"]) if (not np.isnan(mean_mpe_pct) and abs(mean_mpe_pct) > bias_cfg["bands"]["poor_abs_pct"]) else 0.0

    composite_score = max(cfg["composite_score_formula"]["floor"], composite_raw + flat_line_penalty + outlier_cap_penalty + normalized_bias_penalty)

    # --- Disqualification (hard fail -> score forced to 0) ---
    # Accuracy/DA must BOTH fall below threshold to disqualify (AND, not
    # OR). horizon_ineligible_technique / robustness-gate-failure above are
    # separate, unconditional disqualifiers.
    disq_cfg = cfg["disqualification_criteria"]
    accuracy_below = accuracy_score < disq_cfg["mape_accuracy_below_pct"]
    da_below = da_score < disq_cfg["directional_accuracy_below_pct"]
    if accuracy_below and da_below:
        disqualification_reasons.append("mape_accuracy_below_threshold")
        disqualification_reasons.append("directional_accuracy_below_threshold")
    disqualified = bool(disqualification_reasons)
    if disqualified and apply_disqualification:
        composite_score = 0.0

    # --- Eligibility for recommendation (softer, independent of disqualification) ---
    elig_cfg = cfg["eligibility_thresholds"]
    ineligibility_reasons: list[str] = []
    if accuracy_score < elig_cfg["min_mape_accuracy_pct"]:
        ineligibility_reasons.append("mape_accuracy_below_eligibility_threshold")
    if da_score < elig_cfg["min_directional_accuracy_pct"]:
        ineligibility_reasons.append("directional_accuracy_below_eligibility_threshold")
    eligible_for_recommendation = not disqualified and not ineligibility_reasons

    confidence_tier = None
    if not disqualified:
        tiers = cfg["confidence_tiers"]
        if composite_score >= tiers["high"]["min_score"]:
            confidence_tier = "HIGH"
        elif composite_score >= tiers["medium"]["min_score"]:
            confidence_tier = "MEDIUM"
        else:
            confidence_tier = "LOW"

    return ScoreResult(
        commodity_id=commodity.id, technique=technique_key, horizon_bucket=horizon_bucket,
        evaluated_cycles=evaluated_cycles, insufficient_history=False,
        disqualified=disqualified, disqualification_reasons=disqualification_reasons,
        eligible_for_recommendation=eligible_for_recommendation, ineligibility_reasons=ineligibility_reasons,
        component_scores=component_scores,
        flat_line_penalty=flat_line_penalty, outlier_cap_penalty=outlier_cap_penalty,
        normalized_bias_penalty=normalized_bias_penalty, mean_mpe_pct=mean_mpe_pct,
        composite_score=round(composite_score, 4), confidence_tier=confidence_tier,
    )


@dataclass
class _BenchmarkTechniqueResult:
    """Duck-types as a TechniqueResult (only .detail_df/.summary_df are
    ever read by score_technique_result) -- lets Beroe's own benchmark
    detail get scored through the exact same function real techniques use,
    without needing a real TechniqueResult (which only _run_windowed_
    backtest/_run_direct_horizon_backtest construct)."""
    detail_df: pd.DataFrame
    summary_df: pd.DataFrame


def _summary_from_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates an arbitrary row-level detail_df (Window/Predicted/Actual/
    Row MAPE (%)/Row Accuracy/Directional Accuracy/Abs Directional Accuracy
    already present -- same shape beroe_benchmark.py's st_detail/mt_detail/
    lt_detail already carry) into the minimal per-window summary shape
    _lookback_cycles/score_technique_result actually read: sort_order,
    Window, Months Evaluated, MAPE (%), MPE (%), Directional Accuracy (%),
    Abs Directional Accuracy (%). Needed only because Beroe's benchmark
    detail doesn't come from _run_windowed_backtest, so it never had a
    summary_df built for it the way every real technique's does. MPE (%)
    (signed, unlike MAPE) isn't a column on the input detail_df either --
    derived here the same way Row MAPE (%) already was, from Predicted/
    Actual directly."""
    records = []
    for window, wg in detail_df.groupby("Window"):
        evaluated = wg.dropna(subset=["Actual"])
        if evaluated.empty:
            records.append({
                "sort_order": window, "Window": window, "Months Evaluated": 0,
                "MAPE (%)": np.nan, "MPE (%)": np.nan,
                "Directional Accuracy (%)": np.nan, "Abs Directional Accuracy (%)": np.nan,
            })
            continue
        signed_pe = (evaluated["Actual"] - evaluated["Predicted"]) / evaluated["Actual"] * 100
        records.append({
            "sort_order": window, "Window": window, "Months Evaluated": len(evaluated),
            "MAPE (%)": evaluated["Row MAPE (%)"].mean(), "MPE (%)": signed_pe.mean(),
            "Directional Accuracy (%)": evaluated["Directional Accuracy"].mean() * 100,
            "Abs Directional Accuracy (%)": evaluated["Abs Directional Accuracy"].mean() * 100,
        })
    return pd.DataFrame(records).sort_values("sort_order")


def score_benchmark_result(detail_df: pd.DataFrame, commodity: CommodityConfig, horizon_bucket: str, price_series: pd.Series) -> ScoreResult:
    """
    Scores Beroe's own past forecast (the "Benchmark" row shown in Scoring
    Detail) through the exact same composite formula real techniques get --
    lets it be directly compared on equal footing, not just its raw
    accuracy. Composite Score itself is NEVER forced to 0 by the hard-
    disqualification rule (apply_disqualification=False): Benchmark is a
    reference point, not a recommendable technique, so its displayed Score
    should always reflect the raw formula, not the same hard-fail gate real
    techniques get. disqualified/disqualification_reasons are still
    computed and returned as-is.

    detail_df: one of beroe_benchmark.py's ForecastAccuracyResult.st_detail/
    mt_detail/lt_detail (matching horizon_bucket).

    The caller MUST NOT append this ScoreResult into the score_results list
    passed to recommend()/build_summary_df's ranking -- Benchmark is a
    reference point, never a candidate for "our" recommended technique.
    config/technique_matrix.yaml's "benchmark" entry exists solely so this
    call succeeds (eligibility lookup) and its own accuracy/DA numbers, not
    horizon-ineligibility or robustness-gating, drive its disqualification.
    """
    fake_result = _BenchmarkTechniqueResult(detail_df=detail_df, summary_df=_summary_from_detail(detail_df))
    return score_technique_result(
        fake_result, commodity, horizon_bucket, "benchmark", price_series,
        robustness_gate_passed=None, apply_disqualification=False,
    )
