"""
Runs every eligible technique for ONE commodity x ONE horizon bucket
through feature engineering, modeling, scoring, and output generation.
(Raw ingestion + EDA/driver-selection happen once per batch, in
run_batch.py, not repeated per horizon.)

Execution order matters: univariate techniques run before their gated
ext-var counterparts (MATCHED_UNIVARIATE below) so the Robustness Gate's
ablation test always has a same-horizon univariate TechniqueResult to
compare against by the time a gated technique needs one.

TST/TFT (torch-only) are imported lazily and skipped — not failed — when
torch isn't installed in the running interpreter, so a missing torch
import degrades to "skip these two techniques, run everything else" rather
than crashing the whole horizon run.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "data", "features", "models", "scoring"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import CommodityConfig, PROJECT_ROOT, load_backtest_start_date  # noqa: E402
from external_driver_loader import DATE_COLUMN_NAME, ExternalDriverData, load_external_drivers  # noqa: E402
from internal_features import build_internal_features  # noqa: E402
from external_feature_merge import assemble_features  # noqa: E402
from driver_projection import DriverProjectionResult, project_drivers  # noqa: E402

from arima import run_arima  # noqa: E402
from sarima import run_sarima  # noqa: E402
from ets import run_ets  # noqa: E402
from rf import run_rf  # noqa: E402
from lgbm import run_lgbm  # noqa: E402
from markov_switching import run_markov_switching  # noqa: E402
from arima_garch import run_arima_garch  # noqa: E402
from sarima_garch import run_sarima_garch  # noqa: E402
from rf_ext import run_rf_ext  # noqa: E402
from lgbm_ext import run_lgbm_ext  # noqa: E402
from arimax import run_arimax  # noqa: E402
from sarimax import run_sarimax  # noqa: E402
from var_vecm import run_var, run_vecm  # noqa: E402

# tst.py / tft.py moved to modeling/archive/models_deep_learning_disabled/
# -- deliberately not importable, so tst/tft/tft_ext can never run even if
# torch gets installed on some machine later (move the two files back to
# src/models/ to re-enable). The ImportError below is expected and
# permanent, not an environment gap to fix.
try:
    from tst import run_tst
    from tft import run_tft
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from robustness_gate import evaluate_robustness_gate, shuffle_driver_columns  # noqa: E402
from composite_score import score_technique_result, score_benchmark_result  # noqa: E402
from recommend import recommend  # noqa: E402
from decision_log import DECISION_LOG_XLSX, get_current_decision  # noqa: E402
from selection_history import SELECTION_HISTORY_XLSX, record_selection  # noqa: E402
from model_persistence import save_top_n_models  # noqa: E402
from beroe_benchmark import compute_beroe_benchmark, write_beroe_benchmark_xlsx  # noqa: E402

logger = logging.getLogger(__name__)

TECHNIQUE_MATRIX_YAML = PROJECT_ROOT / "config" / "technique_matrix.yaml"
# Maps this run's horizon_bucket to the matching ForecastAccuracyResult
# detail attribute (benchmark_accuracy.py's st_detail/mt_detail/lt_detail),
# same mapping generate_consolidated_summary.py uses for its own Data -
# ST/MT/LT sheets, so the Summary File's Benchmark Accuracy (%) column
# always agrees with what the consolidated summary shows.
HORIZON_TO_BEROE_DETAIL_ATTR = {"short": "st_detail", "medium": "mt_detail", "long": "lt_detail"}

# Runs before its gated ext-var counterpart so a same-horizon univariate
# TechniqueResult always exists by the time the Robustness Gate needs one
# for the ablation test. var/vecm have no direct univariate sibling (they're
# inherently multivariate) -- ARIMA is used as their baseline, a judgment
# call.
EXECUTION_ORDER = [
    "arima", "sarima", "ets", "rf", "lgbm", "markov_switching", "arima_garch", "sarima_garch", "tst", "tft",
    "rf_ext", "lgbm_ext", "arimax", "sarimax", "tft_ext", "var", "vecm",
]
MATCHED_UNIVARIATE = {"rf_ext": "rf", "lgbm_ext": "lgbm", "arimax": "arima", "sarimax": "sarima", "tft_ext": "tft", "var": "arima", "vecm": "arima"}
# lstm, tst_ext: in technique_matrix.yaml but no module exists to run them.


def _load_eligibility_map() -> dict[str, dict[str, str]]:
    techniques = yaml.safe_load(TECHNIQUE_MATRIX_YAML.read_text(encoding="utf-8"))["techniques"]
    return {t["name"]: t["eligibility"] for t in techniques}


@dataclass
class RunContext:
    commodity: CommodityConfig
    price_series: pd.Series
    price_col: str
    internal_feat: pd.DataFrame
    ext_feature_df: pd.DataFrame
    ext_var_cols: list[str]
    external_driver_data: ExternalDriverData
    driver_projection_result: DriverProjectionResult
    warmup_periods: int
    forecast_periods: int
    horizon_label: str
    n_windows: int
    n_trials: "int | None"   # None -> each technique keeps its own tuned default
    n_permutations: int

    def kw(self, include_n_trials: bool = True) -> dict:
        d = {"n_windows": self.n_windows, "horizon_label": self.horizon_label}
        if include_n_trials and self.n_trials is not None:
            d["n_trials"] = self.n_trials
        return d


def _run_arima(ctx): return run_arima(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_sarima(ctx): return run_sarima(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_ets(ctx): return run_ets(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_rf(ctx): return run_rf(ctx.commodity, ctx.price_series, ctx.internal_feat, ctx.price_col, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_lgbm(ctx): return run_lgbm(ctx.commodity, ctx.price_series, ctx.internal_feat, ctx.price_col, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_markov_switching(ctx): return run_markov_switching(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_arima_garch(ctx): return run_arima_garch(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_sarima_garch(ctx): return run_sarima_garch(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_tst(ctx): return run_tst(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_tft_univariate(ctx): return run_tft(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, external_driver_data=None, **ctx.kw())
def _run_rf_ext(ctx): return run_rf_ext(ctx.commodity, ctx.price_series, ctx.ext_feature_df, ctx.price_col, ctx.ext_var_cols, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_lgbm_ext(ctx): return run_lgbm_ext(ctx.commodity, ctx.price_series, ctx.ext_feature_df, ctx.price_col, ctx.ext_var_cols, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
# arimax/sarimax use pmdarima.auto_arima internally, not Optuna -- no n_trials kwarg exists on either.
def _run_arimax(ctx): return run_arimax(ctx.commodity, ctx.price_series, ctx.external_driver_data, ctx.driver_projection_result, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw(include_n_trials=False))
def _run_sarimax(ctx): return run_sarimax(ctx.commodity, ctx.price_series, ctx.external_driver_data, ctx.driver_projection_result, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw(include_n_trials=False))
def _run_tft_ext(ctx): return run_tft(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, external_driver_data=ctx.external_driver_data, **ctx.kw())
def _run_var(ctx): return run_var(ctx.commodity, ctx.price_series, ctx.external_driver_data, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())
def _run_vecm(ctx): return run_vecm(ctx.commodity, ctx.price_series, ctx.external_driver_data, ctx.warmup_periods, ctx.forecast_periods, **ctx.kw())


def _technique_runners() -> dict:
    runners = {
        "arima": _run_arima, "sarima": _run_sarima, "ets": _run_ets, "rf": _run_rf, "lgbm": _run_lgbm,
        "markov_switching": _run_markov_switching, "arima_garch": _run_arima_garch, "sarima_garch": _run_sarima_garch,
        "rf_ext": _run_rf_ext, "lgbm_ext": _run_lgbm_ext, "arimax": _run_arimax, "sarimax": _run_sarimax,
        "var": _run_var, "vecm": _run_vecm,
    }
    if TORCH_AVAILABLE:
        runners.update({"tst": _run_tst, "tft": _run_tft_univariate, "tft_ext": _run_tft_ext})
    return runners


def _make_rerun_with_shuffled(technique_key: str, ctx: RunContext, best_params: dict):
    """
    Robustness Gate (Step 10) permutation-test closure, dispatched per
    technique family -- see robustness_gate.py's own module docstring for
    why this can't be one generic wrapper (each family's run_* signature
    shape differs). Shuffles only the TRAINING-time driver values (feature_df
    for rf_ext/lgbm_ext; external_driver_data.df -- the lagged columns
    actually fit on -- for the exog-native families); driver_projection_result
    (arimax/sarimax's FUTURE exog beyond history) is deliberately left real,
    since the permutation test asks whether the driver's in-sample
    contribution is genuine signal, not whether the future projection itself
    is noise.
    """
    if technique_key in ("rf_ext", "lgbm_ext"):
        run_fn = run_rf_ext if technique_key == "rf_ext" else run_lgbm_ext

        def rerun(seed):
            shuffled_df = shuffle_driver_columns(ctx.ext_feature_df, ctx.ext_var_cols, DATE_COLUMN_NAME, seed)
            return run_fn(
                ctx.commodity, ctx.price_series, shuffled_df, ctx.price_col, ctx.ext_var_cols,
                ctx.warmup_periods, ctx.forecast_periods, n_windows=ctx.n_windows,
                fixed_params=best_params, horizon_label=ctx.horizon_label,
            )
        return rerun

    if technique_key in ("arimax", "sarimax", "var", "vecm", "tft_ext"):
        driver_labels = [sel.label for sel in ctx.external_driver_data.lag_selections]

        def rerun(seed):
            shuffled_df = shuffle_driver_columns(ctx.external_driver_data.df, driver_labels, DATE_COLUMN_NAME, seed)
            shuffled_data = dataclasses.replace(ctx.external_driver_data, df=shuffled_df)
            if technique_key == "arimax":
                return run_arimax(ctx.commodity, ctx.price_series, shuffled_data, ctx.driver_projection_result, ctx.warmup_periods, ctx.forecast_periods, n_windows=ctx.n_windows, horizon_label=ctx.horizon_label)
            if technique_key == "sarimax":
                return run_sarimax(ctx.commodity, ctx.price_series, shuffled_data, ctx.driver_projection_result, ctx.warmup_periods, ctx.forecast_periods, n_windows=ctx.n_windows, horizon_label=ctx.horizon_label)
            if technique_key == "var":
                return run_var(ctx.commodity, ctx.price_series, shuffled_data, ctx.warmup_periods, ctx.forecast_periods, n_windows=ctx.n_windows, fixed_params=best_params, horizon_label=ctx.horizon_label)
            if technique_key == "vecm":
                return run_vecm(ctx.commodity, ctx.price_series, shuffled_data, ctx.warmup_periods, ctx.forecast_periods, n_windows=ctx.n_windows, fixed_params=best_params, horizon_label=ctx.horizon_label)
            return run_tft(ctx.commodity, ctx.price_series, ctx.warmup_periods, ctx.forecast_periods, external_driver_data=shuffled_data, n_windows=ctx.n_windows, fixed_params=best_params, horizon_label=ctx.horizon_label)
        return rerun

    raise ValueError(f"_make_rerun_with_shuffled: no shuffled-rerun closure defined for {technique_key!r}")


@dataclass
class HorizonRunResult:
    commodity_id: str
    horizon_bucket: str
    results_by_technique: dict          # technique_key -> TechniqueResult
    score_results: list                 # list[composite_score.ScoreResult]
    skipped_techniques: dict            # technique_key -> reason string
    recommendation: object              # recommend.RecommendationResult
    decision: object                    # decision_log.DecisionRecord (active decision used for the final output)
    output_paths: dict                  # {"review": {...}, "final": {...}}
    beroe_benchmark: object              # beroe_benchmark.ForecastAccuracyResult, or None (no mapping / no confirmed technique yet)
    benchmark_score: object = None       # composite_score.ScoreResult for Benchmark itself, or None -- NEVER part of score_results/recommend()
    runtime_by_technique: dict = None    # technique_key -> wall-clock seconds for that fit, for the client timing report


def run_horizon(
    commodity: CommodityConfig,
    horizon_bucket: str,
    forecast_cycle: "str | None" = None,
    fast: bool = False,
    output_root: Path = None,
    decision_log_path: Path = DECISION_LOG_XLSX,
    selection_history_path: Path = SELECTION_HISTORY_XLSX,
    saved_models_root: "Path | None" = None,
) -> HorizonRunResult:
    """
    Runs feature engineering, modeling, scoring, and output generation for
    one commodity x horizon_bucket ("short"/"medium"/"long"). Raw ingestion,
    EDA, and driver selection are assumed already done -- run_batch.py
    handles those once per batch, not per horizon, since driver selection
    is static (don't re-run per horizon or even per cycle once it's
    happened).

    fast=True scales n_windows/n_trials/n_permutations down uniformly for
    smoke-testing -- production batch runs should leave it False so every
    technique gets its own properly-tuned defaults.

    output_root/decision_log_path/selection_history_path/saved_models_root
    all default to their real production locations -- override every one of
    them (not just output_root) when running a throwaway/verification call
    (e.g. fast=True) that shouldn't touch real state. A prior fast=True
    verification run that redirected only output_root still leaked real
    pickles into saved_models/ AND overwrote a real technique_selection_
    history.xlsx row, precisely because those two weren't also redirected --
    see git history/README Section 8 for the incident this parameter exists
    to prevent from recurring.
    """
    from generate_outputs import generate_outputs  # local import: pipeline/ imports pipeline/, avoid a cycle at module load time

    if output_root is None:
        output_root = PROJECT_ROOT / "outputs"

    n_trials = 2 if fast else None
    n_permutations = 2 if fast else 30
    forecast_cycle = forecast_cycle or pd.Timestamp.now().strftime("%b-%Y")

    external_driver_data = load_external_drivers(commodity)
    price_series = external_driver_data.df.set_index(DATE_COLUMN_NAME)[external_driver_data.price_column]

    if fast:
        n_windows = 4
    else:
        # Fixed anchor (config/validation.yaml), not a fixed count: every
        # actual period from backtest_start_date through the latest
        # available data becomes one backtest window, so this grows every
        # cycle instead of staying pinned at the same recent-N periods.
        backtest_start_date = load_backtest_start_date()
        n_windows = int((price_series.index >= backtest_start_date).sum())
        if n_windows < 1:
            raise ValueError(
                f"{commodity.id}: no price data on/after backtest_start_date "
                f"({backtest_start_date.date()}) in config/validation.yaml"
            )

    internal_feat = build_internal_features(commodity, external_driver_data.df, external_driver_data.price_column)
    assembly = assemble_features(commodity, internal_feat, external_driver_data)
    ext_var_cols = [d.label for d in assembly.decisions if d.action in ("included", "imputed")]
    driver_projection_result = project_drivers(commodity, external_driver_data)

    m_start, m_end = getattr(commodity.horizon_periods, horizon_bucket)
    ctx = RunContext(
        commodity=commodity, price_series=price_series, price_col=external_driver_data.price_column,
        internal_feat=internal_feat, ext_feature_df=assembly.df, ext_var_cols=ext_var_cols,
        external_driver_data=external_driver_data, driver_projection_result=driver_projection_result,
        warmup_periods=m_start - 1, forecast_periods=m_end - m_start + 1, horizon_label=horizon_bucket,
        n_windows=n_windows, n_trials=n_trials, n_permutations=n_permutations,
    )

    eligibility_map = _load_eligibility_map()
    runners = _technique_runners()

    results_by_technique: dict = {}
    score_results: list = []
    skipped: dict = {}
    runtime_by_technique: dict = {}

    for technique_key in EXECUTION_ORDER:
        eligibility = eligibility_map[technique_key][horizon_bucket]
        if eligibility == "no":
            skipped[technique_key] = "horizon_ineligible (technique_matrix.yaml)"
            continue
        if technique_key not in runners:
            skipped[technique_key] = "not runnable in this environment (no module, or torch unavailable)"
            continue

        fit_start = time.perf_counter()
        try:
            result = runners[technique_key](ctx)
        except Exception as exc:
            logger.warning("%s/%s/%s: fit failed — %s", commodity.id, horizon_bucket, technique_key, exc)
            skipped[technique_key] = f"fit_failed: {type(exc).__name__}: {exc}"
            continue
        results_by_technique[technique_key] = result

        robustness_gate_passed = None
        if eligibility == "gated" and horizon_bucket == "long":
            univariate_result = results_by_technique.get(MATCHED_UNIVARIATE.get(technique_key))
            if univariate_result is None:
                logger.warning("%s/%s/%s: gated at Long Term but no matched univariate result to ablation-test against — treating as gate failure", commodity.id, horizon_bucket, technique_key)
                robustness_gate_passed = False
            else:
                try:
                    rerun_fn = _make_rerun_with_shuffled(technique_key, ctx, result.best_params)
                    gate_result = evaluate_robustness_gate(
                        commodity.id, technique_key, result, univariate_result, rerun_fn, n_permutations=ctx.n_permutations,
                    )
                    robustness_gate_passed = gate_result.passed
                except Exception as exc:
                    logger.warning("%s/%s/%s: robustness gate evaluation failed — %s (treating as gate failure)", commodity.id, horizon_bucket, technique_key, exc)
                    robustness_gate_passed = False

        # Stopped here, not right after the fit -- for gated Long Term
        # techniques (RF+ext, LGBM+ext, ARIMAX, SARIMAX, TFT+ext, VAR/VECM)
        # this includes the robustness gate's own ablation+permutation cost.
        # The old placement (right after the fit) under-reported these 6
        # techniques' true runtime, most visibly on VAR/VECM, whose own fit
        # is genuinely sub-second while its gate overhead can run minutes.
        runtime_by_technique[technique_key] = time.perf_counter() - fit_start

        score = score_technique_result(result, commodity, horizon_bucket, technique_key, price_series, robustness_gate_passed=robustness_gate_passed)
        score_results.append(score)

    # Standalone, non-blocking: runs only after every technique above has
    # already been fit and scored, entirely separate from the loop and the
    # shared backtest engine those numbers came from. A failure here can
    # only ever affect what gets saved to saved_models/, never the scores,
    # recommendation, or anything else this run produces.
    try:
        save_top_n_models(
            commodity, horizon_bucket, results_by_technique, score_results, price_series, internal_feat, external_driver_data.price_column,
            external_driver_data=external_driver_data, ext_feature_df=assembly.df, ext_var_cols=ext_var_cols,
            output_root=saved_models_root,
        )
    except Exception:
        logger.warning("%s/%s: save_top_n_models failed entirely — continuing without saved models", commodity.id, horizon_bucket, exc_info=True)

    recommendation = recommend(commodity.id, horizon_bucket, score_results)
    decision = get_current_decision(commodity.id, horizon_bucket, recommendation, path=decision_log_path)

    # Doesn't depend on which technique is recommended, or on horizon_bucket
    # at all (Beroe's own forecast vintages were never split by horizon the
    # way our own runs are) -- just the commodity's own real price series,
    # already available as `price_series` above. Recomputed identically on
    # every horizon_bucket call; cheap (no model fitting), so not cached.
    # Computed BEFORE generate_outputs so its Summary File's "Benchmark
    # Accuracy (%)" column can actually be populated instead of always NaN.
    beroe_benchmark = None
    benchmark_accuracy = None
    benchmark_score = None
    try:
        beroe_benchmark = compute_beroe_benchmark(commodity.id, price_series)
        if beroe_benchmark is not None:
            benchmark_path = output_root / commodity.id / horizon_bucket / "beroe_benchmark.xlsx"
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            write_beroe_benchmark_xlsx(beroe_benchmark, benchmark_path)
            detail_attr = HORIZON_TO_BEROE_DETAIL_ATTR[horizon_bucket]
            detail_df = getattr(beroe_benchmark, detail_attr)
            if detail_df is not None and not detail_df.empty:
                benchmark_accuracy = float(detail_df["Row Accuracy"].mean())
                try:
                    benchmark_score = score_benchmark_result(detail_df, commodity, horizon_bucket, price_series)
                except Exception:
                    logger.exception("%s/%s: scoring Benchmark itself failed — skipped (non-blocking)", commodity.id, horizon_bucket)
                    benchmark_score = None
    except Exception:
        logger.exception("%s/%s: Beroe benchmark failed — skipped (non-blocking)", commodity.id, horizon_bucket)
        beroe_benchmark = None

    output_paths = {
        "review": generate_outputs(
            "review", commodity.id, horizon_bucket, recommendation, score_results,
            benchmark_accuracy=benchmark_accuracy, region=commodity.region, output_root=output_root,
            benchmark_score=benchmark_score,
        ),
    }
    if decision.technique is not None:
        output_paths["final"] = generate_outputs(
            "final", commodity.id, horizon_bucket, recommendation, score_results, decision=decision,
            benchmark_accuracy=benchmark_accuracy, region=commodity.region, output_root=output_root,
            benchmark_score=benchmark_score,
        )
        record_selection(commodity, horizon_bucket, forecast_cycle, decision, path=selection_history_path)
    else:
        logger.warning("%s/%s: no resolved technique (nothing eligible, nothing overridden) — skipping final output + selection history", commodity.id, horizon_bucket)

    return HorizonRunResult(
        commodity_id=commodity.id, horizon_bucket=horizon_bucket, results_by_technique=results_by_technique,
        score_results=score_results, skipped_techniques=skipped, recommendation=recommendation,
        decision=decision, output_paths=output_paths, beroe_benchmark=beroe_benchmark, benchmark_score=benchmark_score,
        runtime_by_technique=runtime_by_technique,
    )
