"""
The top-level entry point: loops every commodity in config/commodities.yaml
x every horizon bucket through the full pipeline, never letting one
commodity/horizon's failure stop the rest — the same non-blocking
convention src/data/eda.py already uses per-commodity, extended here to
every step.

Adding a commodity is purely a config change (commodities.yaml + its
driver yaml) — this module never branches on a commodity's identity.

No scheduler here — run_batch() is a plain function a person or a future
cron/Task Scheduler entry calls, not a daemon.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "data", "features", "scoring"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import CommodityConfig, PROJECT_ROOT, load_commodities  # noqa: E402
from eda import run_eda_batch  # noqa: E402
from feature_selection import promote_to_selected, resolve_candidate_config, select_drivers, write_selected_yaml  # noqa: E402
from decision_log import DECISION_LOG_XLSX  # noqa: E402
from selection_history import SELECTION_HISTORY_XLSX  # noqa: E402

from run_horizon import HorizonRunResult, run_horizon  # noqa: E402
from generate_consolidated_summary import write_consolidated_summary  # noqa: E402

logger = logging.getLogger(__name__)

HORIZON_BUCKETS = ("short", "medium", "long")
LOGS_ROOT = PROJECT_ROOT / "logs"


def _setup_run_log_file(logs_root: Path) -> "tuple[Path, logging.Handler]":
    """One log file per run_batch() call, timestamped the same way eda.py's
    run_eda_batch already timestamps its own run folders (logs/{run_
    timestamp}/run_batch.log) -- so a run's full diagnostic trail (every
    logger.info/warning/exception already scattered through src/, currently
    thrown away the moment the console closes) survives afterward. Added
    as an EXTRA handler on the root logger, not a replacement for whatever
    console handler the caller already has (CLI's __main__ block still
    calls logging.basicConfig for console output) -- this only adds
    persistence, it doesn't change what shows up on screen. Caller must
    remove the returned handler when done (run_batch does this in a
    finally block) so repeated run_batch() calls in one process don't pile
    up duplicate file handlers."""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = logs_root / run_timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_batch.log"

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
        # Ensures INFO-level messages actually reach this handler even when
        # the caller never set up logging.basicConfig themselves (e.g. a
        # programmatic run_batch() call from a script/notebook, not the
        # CLI __main__ path) -- Python's root logger defaults to WARNING,
        # which would otherwise silently drop every logger.info() call.
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return log_path, handler


def _ensure_drivers_selected(commodities: list[CommodityConfig], force_reselect: bool = False, fast: bool = False) -> list[CommodityConfig]:
    """
    The driver-selection half of ingestion. A commodity at drivers_status=
    'candidates' (nothing is ever permanently frozen — even a 'selected'
    commodity can be moved back to 'candidates') gets run through
    feature_selection.py once; 'selected' commodities are left untouched
    (selection is static once run) UNLESS force_reselect=True.

    force_reselect: a deliberate, explicit opt-in to re-run selection for
    EVERY commodity in this batch, even ones already 'selected' -- e.g.
    after changing config/feature_selection.yaml's selection_mode/
    threshold and wanting it to actually take effect. Off by default: this
    is a real commit (writes selected/{id}.yaml, promotes drivers_status),
    not the read-only force=True preview select_drivers() itself offers --
    it exists so a genuine "re-select everything with today's settings"
    run doesn't require hand-editing commodities.yaml back to 'candidates'
    first, while an ordinary run_batch() call still leaves confirmed driver
    sets untouched.

    fast: same fast=True this function's caller (run_batch) already accepts
    for the forecasting step -- passed straight through to select_drivers,
    which previously had no fast path at all (driver selection always paid
    its full n_stability_runs=10 cost regardless of run_batch(fast=True)).
    Now a fast=True batch is fast end-to-end, not just for forecasting.

    Reloads commodities.yaml only if anything actually changed, so callers
    always see up-to-date drivers_config for the commodities that just got
    (re-)promoted -- but still scoped to the SAME commodities the caller
    passed in. (Bug fixed here: this used to return load_commodities()
    unfiltered whenever anything changed, silently expanding a filtered
    batch -- e.g. run_batch(commodities=[soybean, steel_scrap]) -- into a
    run over all commodities in commodities.yaml the moment either one got
    promoted. Caught when acetic_acid's outputs showed up in a run that was
    only ever supposed to touch soybean/steel_scrap.)
    """
    changed = False
    for commodity in commodities:
        needs_selection = force_reselect or commodity.drivers_status == "candidates"
        if not needs_selection:
            continue
        try:
            logger.info(
                "%s: running feature_selection.py (drivers_status=%s, force_reselect=%s, fast=%s)",
                commodity.id, commodity.drivers_status, force_reselect, fast,
            )
            effective = resolve_candidate_config(commodity) if force_reselect else commodity
            result = select_drivers(effective, force=force_reselect, fast=fast)
            write_selected_yaml(effective, result)
            promote_to_selected(commodity.id)
            changed = True
        except Exception:
            logger.exception("%s: driver selection failed — commodity stays on its current drivers_status (ext-var models skipped for it this run)", commodity.id)
    if not changed:
        return commodities
    reloaded_by_id = {c.id: c for c in load_commodities()}
    return [reloaded_by_id[c.id] for c in commodities]


@dataclass
class BatchResult:
    results: dict = field(default_factory=dict)   # (commodity_id, horizon_bucket) -> HorizonRunResult
    errors: dict = field(default_factory=dict)     # (commodity_id, horizon_bucket) -> error message
    eda_flags_path: "Path | None" = None
    consolidated_summary_path: "Path | None" = None
    log_path: "Path | None" = None


def _run_one_horizon_job(
    commodity: CommodityConfig, horizon_bucket: str, forecast_cycle, fast: bool,
    output_root: Path, decision_log_path: Path, selection_history_path: Path, saved_models_root: "Path | None",
) -> tuple:
    """The unit of work for parallel=True. Must be a
    plain top-level function (not a closure/method) -- ProcessPoolExecutor
    pickles a reference to it to send to each worker process, and only
    top-level functions are picklable that way. Deliberately mirrors the
    sequential loop's own try/except in run_batch() exactly (same error
    stringification), so results/errors come back identical either way --
    parallel=True changes WHERE this runs, never WHAT it computes. Runs in
    a freshly-spawned interpreter; this module's own top-of-file sys.path
    setup (_SRC = Path(__file__).resolve().parents[1], self-locating) reruns
    automatically on import in that fresh process, so no separate path setup
    is needed here."""
    key = (commodity.id, horizon_bucket)
    try:
        result = run_horizon(
            commodity, horizon_bucket, forecast_cycle=forecast_cycle, fast=fast,
            output_root=output_root, decision_log_path=decision_log_path,
            selection_history_path=selection_history_path, saved_models_root=saved_models_root,
        )
        return key, result, None
    except Exception as exc:
        return key, None, f"{type(exc).__name__}: {exc}"


def run_batch(
    commodities: "list[CommodityConfig] | None" = None,
    horizons: tuple = HORIZON_BUCKETS,
    fast: bool = False,
    run_eda: bool = True,
    output_root: "Path | None" = None,
    eda_root: "Path | None" = None,
    logs_root: "Path | None" = None,
    write_log_file: bool = True,
    forecast_cycle: "str | None" = None,
    decision_log_path: Path = DECISION_LOG_XLSX,
    selection_history_path: Path = SELECTION_HISTORY_XLSX,
    saved_models_root: "Path | None" = None,
    force_reselect: bool = False,
    parallel: bool = False,
    max_workers: "int | None" = None,
) -> BatchResult:
    """
    parallel/max_workers: parallel=False (default) is a plain sequential
    commodity x horizon loop. parallel=True flattens every
    (commodity, horizon) pair from this call into ONE pool of jobs (not a
    separate pool per commodity -- two commodities' horizons and one
    commodity's three horizons are equally independent of each other, so
    there's no reason to nest pools) and runs up to max_workers of them at
    once via ProcessPoolExecutor. max_workers is ignored entirely when
    parallel=False. Driver selection (_ensure_drivers_selected, below)
    always stays sequential regardless of this flag -- a commodity's
    horizons can't start until its own driver selection has already been
    written, so that step is a prerequisite pass, not part of the
    parallelized pool.
    """
    commodities = commodities if commodities is not None else load_commodities()
    output_root = output_root or (PROJECT_ROOT / "outputs")
    eda_root = eda_root or (PROJECT_ROOT / "eda")
    logs_root = logs_root or LOGS_ROOT
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    log_handler = None
    if write_log_file:
        log_path, log_handler = _setup_run_log_file(logs_root)
        logger.info("run_batch starting: %d commodities, horizons=%s, fast=%s", len(commodities), horizons, fast)

    try:
        batch_result = BatchResult(log_path=log_path if write_log_file else None)

        if run_eda:
            try:
                batch_result.eda_flags_path = run_eda_batch(commodities, eda_root)
            except Exception:
                # run_eda_for_commodity already catches per-commodity, and
                # run_eda_batch guards per-commodity again -- this is the last
                # line of defense in case the batch-level wiring itself breaks
                # (e.g. can't create eda_root). EDA is diagnostic, never blocking.
                logger.exception("EDA batch failed entirely — continuing without it")

        commodities = _ensure_drivers_selected(commodities, force_reselect=force_reselect, fast=fast)

        if not parallel:
            for commodity in commodities:
                for horizon_bucket in horizons:
                    key = (commodity.id, horizon_bucket)
                    try:
                        batch_result.results[key] = run_horizon(
                            commodity, horizon_bucket, forecast_cycle=forecast_cycle, fast=fast,
                            output_root=output_root, decision_log_path=decision_log_path,
                            selection_history_path=selection_history_path, saved_models_root=saved_models_root,
                        )
                    except Exception as exc:
                        logger.exception("%s/%s: run_horizon failed entirely — skipped", commodity.id, horizon_bucket)
                        batch_result.errors[key] = f"{type(exc).__name__}: {exc}"
        else:
            # Flattened pool: every (commodity, horizon) pair from this call
            # is one job, regardless of which commodity it belongs to -- see
            # _run_one_horizon_job's docstring and run_batch's own docstring
            # above for why this isn't nested per-commodity pools.
            jobs = [(commodity, horizon_bucket) for commodity in commodities for horizon_bucket in horizons]
            logger.info("run_batch: dispatching %d (commodity, horizon) jobs across up to %d parallel workers", len(jobs), max_workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_one_horizon_job, commodity, horizon_bucket, forecast_cycle, fast,
                        output_root, decision_log_path, selection_history_path, saved_models_root,
                    ): (commodity.id, horizon_bucket)
                    for commodity, horizon_bucket in jobs
                }
                for future in as_completed(futures):
                    key, result, error = future.result()
                    if error is not None:
                        logger.error("%s/%s: run_horizon failed entirely — skipped: %s", key[0], key[1], error)
                        batch_result.errors[key] = error
                    else:
                        batch_result.results[key] = result

        if batch_result.results:
            try:
                summary_path = output_root / "consolidated_summary.xlsx"
                write_consolidated_summary(batch_result.results, summary_path)
                batch_result.consolidated_summary_path = summary_path
            except Exception:
                # Cross-commodity summary is a convenience artifact built from
                # already-computed results -- never let it take down a batch
                # that otherwise succeeded (same non-blocking convention as EDA).
                logger.exception("Consolidated summary generation failed — skipped (non-blocking)")

        logger.info("run_batch complete: %d commodity x horizon runs, %d errors", len(batch_result.results), len(batch_result.errors))
        return batch_result
    finally:
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_batch()
    logger.info("log file: %s", result.log_path)
    for key, err in result.errors.items():
        logger.error("%s: %s", key, err)
