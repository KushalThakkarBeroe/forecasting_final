"""
Narrows a commodity's candidate driver list
(config/drivers/candidates/{id}.yaml) down to a finalized selected set,
written to config/drivers/selected/{id}.yaml: data-richness tradeoff test +
RF/LightGBM importance ranking (checked for run-to-run stability) + SHAP
ranking, then a four-criteria statistical selection layer, a manual
override layer, and a final redundancy check.

Where the exact methodology isn't pinned down elsewhere, the assumptions
below are flagged explicitly, not hidden — confirm/adjust as needed via
config/feature_selection.yaml (not code):
  - "Data-richness tradeoff test" = each candidate's data coverage % must
    clear config/scoring.yaml's data_adequacy.min_data_coverage_pct (the
    same threshold Step 5 uses to gate a driver into the feature matrix)
    before it's even considered for importance ranking.
  - "RF/LightGBM importance ranking, checked for stability across runs" =
    each surviving candidate (at its own optimal lag, from
    external_driver_loader.py) is scored by RandomForestRegressor and
    LGBMRegressor feature_importances_, refit across
    feature_selection.yaml's n_stability_runs different random seeds. A
    candidate's stability score is 1 minus the (normalized) standard
    deviation of its rank across those refits — consistently-ranked
    candidates score near 1, volatile ones score near 0.
  - "SHAP ranking" = mean |SHAP value| per candidate, averaged across one
    RandomForest fit and one LightGBM fit (via shap.TreeExplainer), as a
    third, model-explanation-based signal to cross-check the two native
    importances against.
  - Final combined score = simple mean of the three normalized (sum-to-1)
    rankings (RF importance, LGBM importance, SHAP) — so combined_score is
    itself a "share of total importance" (0.15 = 15%). Candidates are
    ranked by this score among those that passed BOTH the data-richness
    gate and the minimum stability score; how many get selected from there
    is config/feature_selection.yaml's selection_mode: top_n (take the top
    max_selected_drivers, the default), threshold (take every candidate
    above a fixed % contribution), or relative_threshold (take every
    candidate above a multiple of its "fair share" 1/N — scales with how
    many candidates a commodity has, unlike a fixed percentage). See that
    file's comments for the full explanation — switching modes is
    config-only, no code change.
  - None of the above weights/thresholds/counts are asserted as final —
    they're reasonable, clearly-labeled defaults, all exposed as config so
    they can be corrected without touching this file.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config_loader import PROJECT_ROOT, CommodityConfig
from external_driver_loader import DATE_COLUMN_NAME, load_external_drivers

logger = logging.getLogger(__name__)

FEATURE_SELECTION_YAML = PROJECT_ROOT / "config" / "feature_selection.yaml"
SCORING_YAML = PROJECT_ROOT / "config" / "scoring.yaml"
SELECTED_DIR = PROJECT_ROOT / "config" / "drivers" / "selected"
CANDIDATES_DIR = PROJECT_ROOT / "config" / "drivers" / "candidates"
COMMODITIES_YAML = PROJECT_ROOT / "config" / "commodities.yaml"


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class DriverScore:
    label: str
    column: int
    unit: str
    data_coverage_pct: float
    passed_data_richness: bool
    rf_importance: float
    lgbm_importance: float
    shap_importance: float
    stability_score: float
    combined_score: float
    selected: bool


@dataclass
class FeatureSelectionResult:
    commodity_id: str
    scores: list[DriverScore]
    selected_drivers: list[dict]  # ready to write into selected/{id}.yaml's selected_drivers list


def _data_coverage_pct(series: pd.Series) -> float:
    return 100.0 * series.notna().sum() / len(series) if len(series) else 0.0


def _rank_stability(importances_by_seed: list[np.ndarray]) -> np.ndarray:
    """Per-candidate stability: 1 - (std of its rank across seeds) /
    n_candidates. Rank 1 = most important. A candidate ranked in the same
    place every seed scores near 1; one whose rank swings widely scores
    near 0."""
    n_candidates = len(importances_by_seed[0])
    ranks = np.array([pd.Series(imp).rank(ascending=False).values for imp in importances_by_seed])
    rank_std = ranks.std(axis=0)
    return np.clip(1 - (rank_std / max(n_candidates, 1)), 0, 1)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def _relative_threshold_select(scores: list[DriverScore], fs_cfg: dict) -> list[DriverScore]:
    """The relative_threshold formula itself, pulled out standalone so
    four_layer_or's criterion 1 can call the exact same logic as the
    relative_threshold selection_mode, not a re-implementation."""
    ranked = sorted(scores, key=lambda s: s.combined_score, reverse=True)
    n = len(ranked)
    fair_share = 1 / n if n else 0
    min_score = fs_cfg["relative_threshold_multiplier"] * fair_share
    return [s for s in ranked if s.combined_score >= min_score]


def _select_correlation_redundancy(scores: list[DriverScore], fs_cfg: dict, driver_data, check_redundancy: bool = True) -> list[DriverScore]:
    """Criterion 2 of four_layer_or (and selection_mode: correlation_
    redundancy on its own): keep a candidate only if |Pearson correlation
    vs price at its already-chosen best lag| exceeds price_correlation_
    threshold.

    check_redundancy: when True (default, used by standalone selection_
    mode: correlation_redundancy), also drops -- among those that passed
    the threshold above -- any that are correlated with an already-kept
    candidate above redundancy_threshold (keeping whichever of the two has
    the higher |Pearson vs price|), self-contained to this criterion alone.
    When False (four_layer_or's own call), this step is skipped entirely --
    multi-collinearity is instead checked exactly ONCE, after the 4
    criteria are OR-combined, by _apply_final_redundancy_check on the
    whole selected set. Previously this ran twice (once here, self-
    contained to criterion 2's own pool; once again at the end across the
    combined set), which could flag the same driver as "redundant" under
    two different, inconsistent comparison pools -- restructured per
    direct client feedback."""
    price_threshold = fs_cfg["price_correlation_threshold"]

    pearson_by_label = {sel.label: sel.pearson_by_lag[sel.selected_lag] for sel in driver_data.lag_selections}

    passed = [
        (s, pearson_by_label[s.label]) for s in scores
        if s.label in pearson_by_label and abs(pearson_by_label[s.label]) > price_threshold
    ]
    passed.sort(key=lambda pair: abs(pair[1]), reverse=True)

    if not check_redundancy:
        return [score for score, _pearson in passed]

    redundancy_threshold = fs_cfg["redundancy_threshold"]
    kept: list[tuple[DriverScore, float]] = []
    for score, pearson in passed:
        is_redundant = False
        for kept_score, _kept_pearson in kept:
            # raw_df (unlagged), not df (each driver individually shifted by
            # its own best lag for price correlation) -- comparing two
            # drivers after they've been shifted by DIFFERENT amounts can
            # hide a real relationship between them (confirmed: Atlantic
            # Cod's two heat-content drivers showed 0.46 lag-shifted vs 0.94
            # raw -- a client-caught, real near-duplicate this was missing).
            pair_corr = driver_data.raw_df[score.label].corr(driver_data.raw_df[kept_score.label])
            if pd.notna(pair_corr) and abs(pair_corr) > redundancy_threshold:
                is_redundant = True
                break
        if not is_redundant:
            kept.append((score, pearson))

    return [score for score, _pearson in kept]


def _select_granger(scores: list[DriverScore], fs_cfg: dict, driver_data) -> list[DriverScore]:
    """Criterion 3 of four_layer_or: Granger Causality, p < granger_
    pvalue_threshold -- does the driver's own past (at its chosen lag)
    help predict price beyond price's own past? Run on first-differenced
    series (Granger assumes stationary input, unlike Johansen). Defensive
    per-candidate: real data is messy (short overlaps, near-constant
    stretches), so a candidate that errors out is simply not selected via
    this criterion, not a hard failure for the whole commodity."""
    price_col = driver_data.price_column
    price_series = driver_data.df.set_index(DATE_COLUMN_NAME)[price_col]
    df_indexed = driver_data.df.set_index(DATE_COLUMN_NAME)
    threshold = fs_cfg["granger_pvalue_threshold"]

    chosen = []
    for s in scores:
        if s.label not in df_indexed.columns:
            continue
        try:
            pair = pd.concat([price_series.rename("price"), df_indexed[s.label].rename("driver")], axis=1).dropna()
            diffed = pair.diff().dropna()
            if len(diffed) <= 10:
                continue
            gres = grangercausalitytests(diffed[["price", "driver"]].values, maxlag=1, verbose=False)
            p_value = gres[1][0]["ssr_ftest"][1]
            if p_value < threshold:
                chosen.append(s)
        except Exception:
            continue
    return chosen


def _select_johansen(scores: list[DriverScore], fs_cfg: dict, driver_data) -> list[DriverScore]:
    """Criterion 4 of four_layer_or: Johansen Cointegration -- only
    meaningful (and only run) when BOTH price and the driver are non-
    stationary (checked via ADF), since cointegration is a long-run
    equilibrium concept between two trending series, not a bounded one.
    A driver that's stationary itself, or whose overlap with price is too
    short/messy to test, simply isn't selected via this criterion --
    that's a legitimate "not applicable" outcome, not an error."""
    price_col = driver_data.price_column
    price_series = driver_data.df.set_index(DATE_COLUMN_NAME)[price_col]
    df_indexed = driver_data.df.set_index(DATE_COLUMN_NAME)
    crit_col = fs_cfg["johansen_critical_value_column"]  # 0=90%, 1=95%, 2=99% in statsmodels' cvt table

    chosen = []
    for s in scores:
        if s.label not in df_indexed.columns:
            continue
        try:
            pair = pd.concat([price_series.rename("price"), df_indexed[s.label].rename("driver")], axis=1).dropna()
            if len(pair) <= 15:
                continue
            price_adf_p = adfuller(pair["price"])[1]
            driver_adf_p = adfuller(pair["driver"])[1]
            if price_adf_p < 0.05 or driver_adf_p < 0.05:
                continue  # one or both are stationary -- Johansen isn't meaningful here
            jres = coint_johansen(pair[["price", "driver"]].values, det_order=0, k_ar_diff=1)
            trace_stat = jres.lr1[0]
            critical_value = jres.cvt[0][crit_col]
            if trace_stat > critical_value:
                chosen.append(s)
        except Exception:
            continue
    return chosen


def _select_four_layer_or(stable_scores: list[DriverScore], richness_only_scores: list[DriverScore], fs_cfg: dict, driver_data) -> list[DriverScore]:
    """selection_mode: four_layer_or -- a driver is selected if it passes
    ANY of 4 independent criteria (OR, not AND):
      1. Feature importance (relative_threshold's own formula) -- needs
         BOTH the data-richness gate and the RF/LightGBM stability gate,
         since stability is specifically validating this criterion's own
         RF/LightGBM importance signal.
      2. Correlation vs price only (see _select_correlation_redundancy,
         check_redundancy=False here) -- needs ONLY the data-richness gate.
         Pearson correlation doesn't come from RF/LightGBM, so the
         stability gate (built to validate THAT signal) doesn't apply here.
      3. Granger Causality -- data-richness gate only, same reasoning.
      4. Johansen Cointegration -- data-richness gate only, same reasoning.
    No redundancy/multicollinearity elimination happens across (or within)
    these 4 criteria at this stage -- if 3 drivers qualify via 3 different
    criteria and happen to be >70% correlated with each other, all 3 are
    kept here. Multi-collinearity is checked exactly ONCE, after this
    OR-combination, by _apply_final_redundancy_check on the whole combined
    set -- kept as one pass rather than one per criterion, so a driver is
    never flagged "redundant" twice under two different, inconsistent
    comparison pools."""
    criterion_1 = _relative_threshold_select(stable_scores, fs_cfg)
    criterion_2 = _select_correlation_redundancy(richness_only_scores, fs_cfg, driver_data, check_redundancy=False)
    criterion_3 = _select_granger(richness_only_scores, fs_cfg, driver_data)
    criterion_4 = _select_johansen(richness_only_scores, fs_cfg, driver_data)

    seen_labels: set[str] = set()
    combined: list[DriverScore] = []
    for group in (criterion_1, criterion_2, criterion_3, criterion_4):
        for s in group:
            if s.label not in seen_labels:
                seen_labels.add(s.label)
                combined.append(s)
    return combined


def _apply_selection_mode(stable_scores: list[DriverScore], fs_cfg: dict, driver_data=None, richness_only_scores=None) -> list[DriverScore]:
    """Picks the final selected set from candidates that already passed the
    data-richness gate and the stability check, per config/
    feature_selection.yaml's selection_mode. See that file's comments for
    what each mode means — this is a pure config switch, no mode-specific
    code lives anywhere else.

    driver_data/richness_only_scores: only needed by correlation_
    redundancy and four_layer_or (the modes that need raw correlation
    data and/or a data-richness-only candidate pool, not just stable_
    scores) -- unused by top_n/threshold/relative_threshold."""
    ranked = sorted(stable_scores, key=lambda s: s.combined_score, reverse=True)
    mode = fs_cfg.get("selection_mode", "top_n")

    if mode == "top_n":
        return ranked[: fs_cfg["max_selected_drivers"]]

    if mode == "correlation_redundancy":
        return _select_correlation_redundancy(stable_scores, fs_cfg, driver_data)

    if mode == "four_layer_or":
        return _select_four_layer_or(stable_scores, richness_only_scores, fs_cfg, driver_data)

    if mode == "threshold":
        chosen = [s for s in ranked if s.combined_score >= fs_cfg["min_contribution_pct"] / 100]
    elif mode == "relative_threshold":
        chosen = _relative_threshold_select(stable_scores, fs_cfg)
    else:
        raise ValueError(f"Unknown selection_mode: {mode!r} (expected top_n, threshold, relative_threshold, correlation_redundancy, or four_layer_or)")

    cap = fs_cfg.get("max_selected_drivers_cap")
    return chosen[:cap] if cap is not None else chosen


def _apply_beroe_overrides(
    scores: list[DriverScore], chosen: list[DriverScore], drivers_config: dict, driver_entries_by_column: dict,
) -> list[DriverScore]:
    """Manual override layer, applied on top of whatever selection_mode
    already chose. Priority order: force_exclude_drivers > must_include_drivers
    > the 4 statistical criteria (or whichever selection_mode is active) >
    data-richness gate.

    must_include_drivers is a domain/fundamental judgment call, independent
    of any statistical test -- so it can add a candidate that never passed
    the data-richness gate at all (that candidate still exists in `scores`,
    just with passed_data_richness=False and zero importance/correlation
    values). "ALL" means every one of this commodity's candidates.
    force_exclude_drivers wins over everything, including must_include_drivers
    -- it flags a specific near-duplicate/redundant driver to actively drop.

    Both lists are matched against candidate_drivers[].label (the raw,
    unsanitized label in config/drivers/candidates/{id}.yaml) -- not
    DriverScore.label, which is the sanitized version used internally for
    feature-matrix column names. Leaving both fields empty in a candidate
    file (the default) makes this a no-op, i.e. driver selection is purely
    statistical (four_layer_or's 4 criteria)."""
    must_include_raw = drivers_config.get("must_include_drivers") or []
    force_exclude_raw = set(drivers_config.get("force_exclude_drivers") or [])
    if not must_include_raw and not force_exclude_raw:
        return chosen

    all_raw_labels = {e["label"] for e in drivers_config.get("candidate_drivers", [])}
    must_include_raw = all_raw_labels if must_include_raw == "ALL" else set(must_include_raw)

    raw_label_of = {s.label: driver_entries_by_column[s.column]["label"] for s in scores}

    seen_labels = {s.label for s in chosen}
    with_overrides = list(chosen)
    for s in scores:
        if s.label not in seen_labels and raw_label_of[s.label] in must_include_raw:
            seen_labels.add(s.label)
            with_overrides.append(s)

    return [s for s in with_overrides if raw_label_of[s.label] not in force_exclude_raw]


def _apply_final_redundancy_check(chosen: list[DriverScore], fs_cfg: dict, driver_data) -> list[DriverScore]:
    """Runs after four_layer_or's own 4-criteria OR selection (which
    deliberately does NOT eliminate redundancy across criteria -- a driver
    picked via Granger and a driver picked via Johansen are both kept even
    if they're 90% correlated with each other): one more pass on the FINAL
    selected set, ranking by |Pearson vs price| descending, keeping
    greedily, and dropping any driver whose correlation with an
    already-kept driver exceeds redundancy_threshold (same 0.7 default and
    same algorithm as _select_correlation_redundancy's own internal step,
    just applied across the whole selected set instead of being
    self-contained to criterion 2).

    Only ever REMOVES drivers from `chosen` -- never adds one back."""
    redundancy_threshold = fs_cfg["redundancy_threshold"]
    pearson_by_label = {sel.label: sel.pearson_by_lag[sel.selected_lag] for sel in driver_data.lag_selections}

    ranked = sorted(
        (s for s in chosen if s.label in pearson_by_label),
        key=lambda s: abs(pearson_by_label[s.label]), reverse=True,
    )
    unranked = [s for s in chosen if s.label not in pearson_by_label]

    kept: list[DriverScore] = []
    for s in ranked:
        is_redundant = False
        for kept_s in kept:
            # raw_df (unlagged), not df -- same reasoning as
            # _select_correlation_redundancy's own internal step (see that
            # function's comment): comparing two drivers after each has
            # been shifted by a different lag can hide a real relationship.
            pair_corr = driver_data.raw_df[s.label].corr(driver_data.raw_df[kept_s.label])
            if pd.notna(pair_corr) and abs(pair_corr) > redundancy_threshold:
                is_redundant = True
                break
        if not is_redundant:
            kept.append(s)

    return kept + unranked


_EXCHANGE_RATE_PATTERN = re.compile(
    r"exchange rate|exchange_rate|fx rate|usd.?inr|usd.?brl|usd.?vnd|usd.?cny|eur.?usd",
    re.IGNORECASE,
)


def _apply_exchange_rate_safety_net(
    chosen: list[DriverScore], scores: list[DriverScore], driver_data, raw_label_of: dict,
) -> list[DriverScore]:
    """Runs last (after four_layer_or, the override layer, and
    _apply_final_redundancy_check): if the commodity's FINAL selected set
    is exactly one driver, and that one driver is an
    exchange-rate type (flexible name match -- "Macro Factors — Exchange
    Rate", "Exchange Rate — Macrofactor", "USD/INR Exchange rate", etc. are
    all used inconsistently across commodities' source files, so this
    can't be one exact string), add ONE more driver: whichever remaining
    candidate has the strongest |Pearson vs price| correlation. The
    exchange-rate driver itself is NEVER removed or replaced -- it passed
    a real statistical test and stays; this only ever ADDS a second driver
    alongside it, never touches a commodity with 0 or 2+ selected drivers."""
    if len(chosen) != 1:
        return chosen
    sole_driver = chosen[0]
    if not _EXCHANGE_RATE_PATTERN.search(raw_label_of[sole_driver.label]):
        return chosen

    pearson_by_label = {sel.label: sel.pearson_by_lag[sel.selected_lag] for sel in driver_data.lag_selections}
    chosen_labels = {s.label for s in chosen}
    candidates = [
        (s, pearson_by_label[s.label]) for s in scores
        if s.label not in chosen_labels and s.label in pearson_by_label
    ]
    if not candidates:
        return chosen  # nothing else to add -- leave the sole driver as-is

    best = max(candidates, key=lambda pair: abs(pair[1]))
    return chosen + [best[0]]


def resolve_candidate_config(commodity: CommodityConfig) -> CommodityConfig:
    """Returns a commodity whose drivers_config is the full candidate pool
    (config/drivers/candidates/{id}.yaml), regardless of what
    commodity.drivers_status currently says -- for an already-'selected'
    commodity, drivers_config normally points at its FINAL, narrowed
    selected/{id}.yaml (already down to 4), not the original pool. Shared
    by select_drivers(force=True) and run_batch.py's force_reselect path,
    so both read the exact same candidate set."""
    if commodity.drivers_status == "candidates":
        return commodity
    candidates_path = CANDIDATES_DIR / f"{commodity.id}.yaml"
    if not candidates_path.is_file():
        raise ValueError(f"{commodity.id}: needs the full candidate pool at {candidates_path}, but it doesn't exist")
    return dataclasses.replace(commodity, drivers_config=_load_yaml(candidates_path))


FAST_N_STABILITY_RUNS = 2  # mirrors run_horizon.py's fast=True n_trials=2 -- a quick smoke-test value, not a real ranking


def select_drivers(commodity: CommodityConfig, force: bool = False, fast: bool = False) -> FeatureSelectionResult:
    """
    Runs the full candidates -> selected pipeline for one commodity.
    Requires commodity.drivers_status == 'candidates' — a 'selected'
    commodity's driver set is already final (selection is static once run;
    don't re-select every cycle), and should not be run through this again.

    force=True bypasses that requirement, for manual experimentation only
    (e.g. previewing what a different selection_mode/threshold would pick
    for a commodity that's already 'selected' in production) -- never used
    by run_batch.py's automatic path (_ensure_drivers_selected), which must
    keep requiring 'candidates' so a confirmed driver set is never silently
    re-picked just because someone tweaked config/feature_selection.yaml.
    When forced, reads the full candidate pool straight from config/drivers/
    candidates/{commodity.id}.yaml, NOT commodity.drivers_config -- for an
    already-'selected' commodity that field points at its FINAL, narrowed
    file (selected_drivers, already down to 4), not the original candidate
    pool. This function never writes or promotes anything either way
    (write_selected_yaml/promote_to_selected are separate, deliberate
    calls) -- force=True only ever changes what gets READ, so nothing about
    commodities.yaml or the selected/{id}.yaml file is at risk.

    fast=True: same spirit as run_horizon.py's fast=True (quick smoke-test,
    not a real result) but for THIS step specifically -- previously
    feature_selection.py had no fast path at all, so even a "fast"
    run_batch() paid the full cost here (n_stability_runs=10 RF+LightGBM
    refits, unconditionally). fast=True drops n_stability_runs to
    FAST_N_STABILITY_RUNS for this call only -- config/feature_selection.yaml
    itself is untouched, and the resulting stability scores/selected
    drivers should be treated as throwaway, same as a fast=True forecast.
    """
    if not force and commodity.drivers_status != "candidates":
        raise ValueError(
            f"{commodity.id}: drivers_status is '{commodity.drivers_status}', not 'candidates' "
            f"— selection is already final for this commodity, and shouldn't be re-run. "
            f"Pass force=True to preview a re-selection without touching commodities.yaml."
        )

    if force:
        commodity = resolve_candidate_config(commodity)

    fs_cfg = _load_yaml(FEATURE_SELECTION_YAML)
    scoring_cfg = _load_yaml(SCORING_YAML)
    min_coverage_pct = scoring_cfg["data_adequacy"]["min_data_coverage_pct"]
    n_stability_runs = FAST_N_STABILITY_RUNS if fast else fs_cfg["n_stability_runs"]
    min_stability_score = fs_cfg["min_stability_score"]
    n_estimators = fs_cfg["n_estimators"]

    driver_data = load_external_drivers(commodity)
    if not driver_data.lag_selections:
        raise ValueError(f"{commodity.id}: no candidate drivers survived lag selection (Step 1) — nothing to select from")

    price = driver_data.df.set_index(DATE_COLUMN_NAME)[driver_data.price_column]
    feature_df = driver_data.df.set_index(DATE_COLUMN_NAME)[[sel.label for sel in driver_data.lag_selections]]
    driver_entries_by_column = {e["column"]: e for e in commodity.drivers_config.get("candidate_drivers", [])}
    column_by_label = {sel.label: sel.column for sel in driver_data.lag_selections}

    # ---- Data-richness tradeoff test ----
    coverage = {label: _data_coverage_pct(feature_df[label]) for label in feature_df.columns}
    eligible_labels = [label for label in feature_df.columns if coverage[label] >= min_coverage_pct]

    if not eligible_labels:
        logger.warning("%s: no candidates passed the data-richness gate (>= %.0f%% coverage)", commodity.id, min_coverage_pct)
        scores = [
            DriverScore(
                label=label, column=column_by_label[label],
                unit=driver_entries_by_column[column_by_label[label]]["unit"],
                data_coverage_pct=coverage[label], passed_data_richness=False,
                rf_importance=0, lgbm_importance=0, shap_importance=0,
                stability_score=0, combined_score=0, selected=False,
            )
            for label in feature_df.columns
        ]
        return FeatureSelectionResult(commodity_id=commodity.id, scores=scores, selected_drivers=[])

    # Complete-case rows only, on eligible columns + price
    model_df = pd.concat([price.rename("__price__"), feature_df[eligible_labels]], axis=1).dropna()
    if len(model_df) < 10:
        raise ValueError(
            f"{commodity.id}: only {len(model_df)} complete-case rows available after the "
            f"data-richness gate — too few to fit a ranking model"
        )
    X, y = model_df[eligible_labels], model_df["__price__"]

    # ---- RF / LightGBM importance, refit across seeds for stability ----
    rf_importances, lgbm_importances = [], []
    for seed in range(n_stability_runs):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1)
        rf.fit(X, y)
        rf_importances.append(rf.feature_importances_)

        lgbm = LGBMRegressor(n_estimators=n_estimators, random_state=seed, verbosity=-1, n_jobs=1)
        lgbm.fit(X, y)
        lgbm_importances.append(lgbm.feature_importances_)

    rf_norm = _normalize(np.mean(rf_importances, axis=0))
    lgbm_norm = _normalize(np.mean(lgbm_importances, axis=0))
    stability = (_rank_stability(rf_importances) + _rank_stability(lgbm_importances)) / 2

    # ---- SHAP ranking: mean |SHAP| from one RF fit + one LGBM fit ----
    # n_jobs=1 on both (see the loop above's comment): removes core-count-
    # dependent thread summation order as a source of cross-machine
    # divergence in the importance ranking that drives driver selection.
    rf_final = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=1).fit(X, y)
    lgbm_final = LGBMRegressor(n_estimators=n_estimators, random_state=0, verbosity=-1, n_jobs=1).fit(X, y)
    shap_rf = np.abs(shap.TreeExplainer(rf_final).shap_values(X)).mean(axis=0)
    shap_lgbm = np.abs(shap.TreeExplainer(lgbm_final).shap_values(X)).mean(axis=0)
    shap_norm = _normalize((_normalize(shap_rf) + _normalize(shap_lgbm)) / 2)

    combined = (rf_norm + lgbm_norm + shap_norm) / 3

    scores = []
    for i, label in enumerate(eligible_labels):
        col = column_by_label[label]
        scores.append(DriverScore(
            label=label, column=col, unit=driver_entries_by_column[col]["unit"],
            data_coverage_pct=coverage[label], passed_data_richness=True,
            rf_importance=float(rf_norm[i]), lgbm_importance=float(lgbm_norm[i]),
            shap_importance=float(shap_norm[i]), stability_score=float(stability[i]),
            combined_score=float(combined[i]), selected=False,
        ))
    for label in feature_df.columns:
        if label not in eligible_labels:
            col = column_by_label[label]
            scores.append(DriverScore(
                label=label, column=col, unit=driver_entries_by_column[col]["unit"],
                data_coverage_pct=coverage[label], passed_data_richness=False,
                rf_importance=0, lgbm_importance=0, shap_importance=0,
                stability_score=0, combined_score=0, selected=False,
            ))

    # ---- Rank eligible-and-stable candidates, apply the configured selection mode ----
    stable_scores = [s for s in scores if s.passed_data_richness and s.stability_score >= min_stability_score]
    # Data-richness only (no stability requirement) -- correlation_redundancy
    # and four_layer_or's criteria 2-4 use this pool instead of stable_scores,
    # since the stability gate specifically validates the RF/LightGBM
    # importance signal (criterion 1's own basis), not Pearson correlation/
    # Granger/Johansen, which don't come from RF/LightGBM at all.
    richness_only_scores = [s for s in scores if s.passed_data_richness]
    chosen = _apply_selection_mode(stable_scores, fs_cfg, driver_data, richness_only_scores)
    chosen = _apply_beroe_overrides(scores, chosen, commodity.drivers_config, driver_entries_by_column)
    chosen = _apply_final_redundancy_check(chosen, fs_cfg, driver_data)
    raw_label_of = {s.label: driver_entries_by_column[s.column]["label"] for s in scores}
    chosen = _apply_exchange_rate_safety_net(chosen, scores, driver_data, raw_label_of)
    chosen_labels = {s.label for s in chosen}
    for s in scores:
        s.selected = s.label in chosen_labels

    scores.sort(key=lambda s: s.combined_score, reverse=True)

    selected_drivers_yaml = [
        {"column": s.column, "label": driver_entries_by_column[s.column]["label"], "unit": s.unit}
        for s in chosen
    ]

    if not chosen:
        logger.warning(
            "%s: 0 drivers selected under selection_mode=%r — check config/feature_selection.yaml's thresholds",
            commodity.id, fs_cfg.get("selection_mode", "top_n"),
        )
    elif fs_cfg.get("selection_mode", "top_n") == "top_n" and len(chosen) < fs_cfg["max_selected_drivers"]:
        logger.warning(
            "%s: only %d/%d drivers selected (fewer candidates passed data-richness + stability gates than max_selected_drivers)",
            commodity.id, len(chosen), fs_cfg["max_selected_drivers"],
        )

    return FeatureSelectionResult(commodity_id=commodity.id, scores=scores, selected_drivers=selected_drivers_yaml)


def write_selected_yaml(commodity: CommodityConfig, result: FeatureSelectionResult) -> Path:
    """Writes config/drivers/selected/{id}.yaml from a FeatureSelectionResult
    — overwrites any existing file at that path. Selection is static once
    run: this is meant to be called once per commodity, not every cycle.
    Carries over data_file/sheet_name/header_rows/date_column/price_column/
    lag_search_range/feedstock_margin_spread unchanged from the candidates
    config that fed this selection."""
    cfg = commodity.drivers_config
    path = SELECTED_DIR / f"{commodity.id}.yaml"

    header = [
        "# FINALIZED driver set — written by src/features/feature_selection.py",
        "# (data-richness tradeoff test + RF/LightGBM importance ranking, checked",
        "# for stability across runs, + SHAP ranking, + a four-criteria",
        "# statistical selection layer). Re-run after adjusting",
        "# config/feature_selection.yaml if the selection needs to change.",
        "#",
        "# Lag: dynamic per driver, not fixed — see src/data/external_driver_loader.py.",
        "",
    ]
    body = {
        "commodity_id": commodity.id,
        "source": "feature_selection.py (Step 2 automated selection)",
        "data_file": cfg["data_file"],
        "sheet_name": cfg["sheet_name"],
        "header_rows": cfg["header_rows"],
        "date_column": cfg["date_column"],
        "price_column": cfg["price_column"],
        "lag_search_range": cfg["lag_search_range"],
        "selected_drivers": result.selected_drivers,
        "feedstock_margin_spread": cfg.get("feedstock_margin_spread", {"enabled": False, "feedstock_driver_column": None}),
    }
    path.write_text("\n".join(header) + yaml.dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("%s: wrote %s (%d drivers selected)", commodity.id, path, len(result.selected_drivers))
    return path


def promote_to_selected(commodity_id: str) -> None:
    """Flips a commodity's drivers_status: candidates -> selected in
    commodities.yaml and points drivers_config at selected/{id}.yaml.
    Requires write_selected_yaml to have already been called for this
    commodity (the file it points to must exist)."""
    selected_path = SELECTED_DIR / f"{commodity_id}.yaml"
    if not selected_path.is_file():
        raise FileNotFoundError(f"{selected_path} does not exist — call write_selected_yaml first")

    with open(COMMODITIES_YAML, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    found = False
    for entry in doc["commodities"]:
        if entry["id"] == commodity_id:
            entry["drivers_status"] = "selected"
            entry["drivers_config"] = f"config/drivers/selected/{commodity_id}.yaml"
            found = True
            break
    if not found:
        raise ValueError(f"{commodity_id} not found in {COMMODITIES_YAML}")

    with open(COMMODITIES_YAML, encoding="utf-8") as fh:
        header_lines = []
        for line in fh:
            if line.startswith("#") or not line.strip():
                header_lines.append(line.rstrip("\n"))
            else:
                break
        # Strip trailing blank lines so the separator below doesn't compound
        # across repeated calls (each call re-reads its own prior output).
        while header_lines and not header_lines[-1].strip():
            header_lines.pop()

    body = yaml.dump({"commodities": doc["commodities"]}, sort_keys=False, allow_unicode=True)
    COMMODITIES_YAML.write_text("\n".join(header_lines) + "\n\n" + body, encoding="utf-8")
    logger.info("%s: promoted to drivers_status=selected in commodities.yaml", commodity_id)
