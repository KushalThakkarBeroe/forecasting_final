"""
Builds a single consolidated cross-commodity workbook -- NOT a replacement
for run_horizon.py's per-commodity/per-horizon review_*.xlsx/final_*.xlsx
files (those stay exactly as they are); this is an additional artifact
generated once, after a batch of run_horizon() calls, that stacks
everything together.

Sheets produced:
  Data - All Horizons
    Row-level detail across every commodity, every horizon, every technique
    (display name), PLUS "Benchmark" rows (Beroe's own forecast scored
    against our real price -- see beroe_benchmark.py's module docstring).
    One combined sheet, distinguished by the "Horizon" column
    ("short"/"medium"/"long"), same values that column already carried on
    technique rows; Benchmark rows didn't carry a Horizon value upstream,
    so it's set explicitly here while building this sheet. Same column
    shape as HorizonRunResult's own TechniqueResult.detail_df, with Model
    renamed to its display name and an "Accuracy (%)" column added (= Row
    Accuracy).

  Driver Selection, Scoring Detail, Technique Runtime
    See build_driver_selection_sheet / build_scoring_detail_sheet /
    build_runtime_sheet below.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "scoring", "data", "features", "models"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import PROJECT_ROOT, load_commodities, load_commodity  # noqa: E402
from external_driver_loader import load_external_drivers  # noqa: E402
from internal_features import DATE_COLUMN_NAME, build_internal_features  # noqa: E402
from external_feature_merge import assemble_features  # noqa: E402
from _common import drop_multicollinear_features, select_important_internal_features  # noqa: E402
from recommend import build_summary_df  # noqa: E402

logger = logging.getLogger(__name__)

TECHNIQUE_MATRIX_YAML = PROJECT_ROOT / "config" / "technique_matrix.yaml"
CANDIDATES_DIR = PROJECT_ROOT / "config" / "drivers" / "candidates"
SELECTED_DIR = PROJECT_ROOT / "config" / "drivers" / "selected"
DATA_SHEET_NAME = "Data - All Horizons"
HORIZON_TO_DETAIL_ATTR = {"short": "st_detail", "medium": "mt_detail", "long": "lt_detail"}

DATA_SHEET_COLUMNS = [
    "commodity_name", "Region", "sort_order", "Model", "Horizon", "Window", "Train End", "Forecast Start",
    "Step", "Date", "Predicted", "Actual", "Directional Accuracy", "Abs Directional Accuracy",
    "Flag", "Abs Error", "APE (%)", "Accuracy (%)", "Row MAE", "Row MSE", "Row RMSE",
    "Row MAPE (%)", "Row Accuracy",
]


def _load_technique_display_names() -> dict[str, str]:
    techniques = yaml.safe_load(TECHNIQUE_MATRIX_YAML.read_text(encoding="utf-8"))["techniques"]
    return {t["name"]: t.get("display_name", t["name"]) for t in techniques}


def _load_region_lookup() -> dict[str, "str | None"]:
    """commodity_id -> region (config/commodities.yaml's own `region` field,
    e.g. 'APAC' for acetic_acid) -- may be None for a commodity scaffold_
    config.py couldn't derive a region for."""
    return {c.id: c.region for c in load_commodities()}


def _multicollinearity_kept_features(cid: str) -> dict:
    """Which of internal_features.py's engineered columns actually survive
    BOTH filtering stages _common.py applies before a model ever sees these
    columns -- Item 1's multicollinearity filter (config/multicollinearity.
    yaml), then Item [cumulative-importance] filter (config/
    internal_feature_importance.yaml) -- replayed here purely for
    reporting, same two calls rf.py/lgbm.py/rf_ext.py/lgbm_ext.py/
    model_persistence.py actually make. Computed TWICE, deliberately kept
    separate rather than reconciled into one answer, because the two paths
    can genuinely disagree:
      - "univariate": both filters run on internal features alone
        (internal_feat, price_col) -- what rf.py/lgbm.py actually use.
      - "ext_aware": both filters run on internal features + selected
        drivers merged (assemble_features' output) -- what rf_ext.py/
        lgbm_ext.py actually use. Selected drivers themselves are never
        dropped or ranked by either filter (ext_var_cols is passed
        through both), but the row set correlations/importances are
        computed over differs (assemble_features' own dropna), so which
        internal features survive can differ from the univariate case
        even though the same thresholds and formulas are used.
    A commodity with no selected drivers naturally gets identical results
    in both, since assemble_features falls back to internal_features_df
    unchanged when there's nothing to merge -- not a bug, just means there
    was nothing for the ext-aware path to differ on.

    Non-blocking: any failure returns an all-empty result for this
    commodity rather than breaking the whole consolidated summary.
    """
    try:
        commodity = load_commodity(cid)
        external_driver_data = load_external_drivers(commodity)
        price_col = external_driver_data.price_column
        internal_feat = build_internal_features(commodity, external_driver_data.df, price_col)

        internal_cols = [c for c in internal_feat.columns if c not in (DATE_COLUMN_NAME, price_col)]
        univariate_kept = drop_multicollinear_features(internal_feat, internal_cols, price_col)
        univariate_kept = select_important_internal_features(internal_feat, univariate_kept, price_col)

        assembly = assemble_features(commodity, internal_feat, external_driver_data)
        ext_var_cols = [d.label for d in assembly.decisions if d.action in ("included", "imputed")]
        ext_feature_cols = [c for c in assembly.df.columns if c not in (DATE_COLUMN_NAME, price_col)]
        ext_kept = drop_multicollinear_features(assembly.df, ext_feature_cols, price_col, ext_var_cols=ext_var_cols)
        ext_kept = select_important_internal_features(assembly.df, ext_kept, price_col, ext_var_cols=ext_var_cols)
        ext_kept_internal = [c for c in ext_kept if c not in ext_var_cols]

        return {"total": len(internal_cols), "univariate_kept": univariate_kept, "ext_kept": ext_kept_internal}
    except Exception:
        logger.warning("%s: could not recompute kept internal features for the Driver Selection sheet", cid, exc_info=True)
        return {"total": 0, "univariate_kept": [], "ext_kept": []}


def build_driver_selection_sheet(commodity_ids: "list[str]") -> pd.DataFrame:
    """One row per commodity: total candidate driver count + names, and
    selected driver count + names, read straight from config/drivers/
    candidates/{id}.yaml and selected/{id}.yaml. Scoped to whichever
    commodity_ids are passed in -- write_consolidated_summary calls this
    with every commodity currently accumulated in the file (all commodities
    trained so far, not just this run's), so it stays a full picture of
    every already-trained commodity, same as the Data sheets.

    Region sits right after Commodity, matching the Data - All Horizons and
    Scoring Detail sheets -- same region_lookup those two use.

    The last five columns report both internal-feature filtering stages
    (multicollinearity, then cumulative-importance -- see
    _multicollinearity_kept_features) -- Total Internal Features is the
    same ~63-column count for every monthly commodity (fewer for
    quarterly, since the 12 is_month flags are monthly-only) before any
    filtering; the Univariate/Ext-Aware pairs show what's actually left
    after BOTH stages, side by side since the two can genuinely differ."""
    region_lookup = _load_region_lookup()
    rows = []
    for cid in sorted(commodity_ids):
        cand_path = CANDIDATES_DIR / f"{cid}.yaml"
        sel_path = SELECTED_DIR / f"{cid}.yaml"

        cand_drivers = []
        if cand_path.is_file():
            cand_cfg = yaml.safe_load(cand_path.read_text(encoding="utf-8")) or {}
            cand_drivers = [d["label"] for d in cand_cfg.get("candidate_drivers", [])]

        sel_drivers = []
        if sel_path.is_file():
            sel_cfg = yaml.safe_load(sel_path.read_text(encoding="utf-8")) or {}
            sel_drivers = [d["label"] for d in sel_cfg.get("selected_drivers", [])]

        mc = _multicollinearity_kept_features(cid)

        rows.append({
            "Commodity": cid,
            "Region": region_lookup.get(cid),
            "Total Candidate Drivers": len(cand_drivers),
            "Candidate Driver Names": "; ".join(cand_drivers),
            "Selected Driver Count": len(sel_drivers),
            "Selected Driver Names": "; ".join(sel_drivers),
            "Total Internal Features": mc["total"],
            "Internal Features Kept (Univariate) Count": len(mc["univariate_kept"]),
            "Internal Features Kept (Univariate) Names": "; ".join(mc["univariate_kept"]),
            "Internal Features Kept (Ext-Aware) Count": len(mc["ext_kept"]),
            "Internal Features Kept (Ext-Aware) Names": "; ".join(mc["ext_kept"]),
        })
    return pd.DataFrame(rows)


def build_runtime_sheet(batch_results: dict) -> pd.DataFrame:
    """One row per commodity x horizon x technique, how long that
    technique's fit actually took (wall-clock seconds), for the
    client-facing timing report. Sourced from
    HorizonRunResult.runtime_by_technique, captured in run_horizon.py
    around each runners[technique_key](ctx) call -- covers the core model
    fit only, not the Robustness Gate's extra permutation-test refits for
    gated Long-Term techniques (a separate, well-understood added cost)."""
    names = _load_technique_display_names()
    region_lookup = _load_region_lookup()
    rows = []

    for (commodity_id, horizon_bucket), run_result in batch_results.items():
        if horizon_bucket not in HORIZON_TO_DETAIL_ATTR:
            continue
        region = region_lookup.get(commodity_id)
        runtimes = getattr(run_result, "runtime_by_technique", None) or {}
        for technique_key, seconds in runtimes.items():
            rows.append({
                "Commodity": commodity_id,
                "Region": region,
                "Horizon": horizon_bucket,
                "Model": names.get(technique_key, technique_key),
                "Runtime (seconds)": round(seconds, 3),
            })

    if not rows:
        return pd.DataFrame(columns=["Commodity", "Region", "Horizon", "Model", "Runtime (seconds)"])
    return pd.DataFrame(rows)


def build_data_sheet(batch_results: dict) -> pd.DataFrame:
    """batch_results: BatchResult.results, {(commodity_id, horizon_bucket): HorizonRunResult}.
    Returns one combined DataFrame, every horizon stacked together --
    distinguish horizon via the "Horizon" column."""
    names = _load_technique_display_names()
    region_lookup = _load_region_lookup()
    parts: list = []

    for (commodity_id, horizon_bucket), run_result in batch_results.items():
        if horizon_bucket not in HORIZON_TO_DETAIL_ATTR:
            continue
        region = region_lookup.get(commodity_id)

        for technique_key, technique_result in run_result.results_by_technique.items():
            detail_df = technique_result.detail_df
            if detail_df is None or detail_df.empty:
                continue
            df = detail_df.copy()
            df["Model"] = names.get(technique_key, technique_key)
            df["Accuracy (%)"] = df["Row Accuracy"]
            df["Region"] = region
            df["Horizon"] = horizon_bucket
            parts.append(df)

        beroe_result = run_result.beroe_benchmark
        if beroe_result is not None:
            detail_attr = HORIZON_TO_DETAIL_ATTR[horizon_bucket]
            bench_df = getattr(beroe_result, detail_attr)
            if bench_df is not None and not bench_df.empty:
                df = bench_df.copy()
                df["Accuracy (%)"] = df["Row Accuracy"]
                df["Region"] = region
                # Benchmark rows don't carry a Horizon value upstream
                # (beroe_benchmark.py's row-builder never sets one, since
                # Beroe's own vintages aren't split by horizon the way our
                # runs are) -- set it explicitly here, now that we know
                # which horizon_bucket this detail_attr came from.
                df["Horizon"] = horizon_bucket
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=DATA_SHEET_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    cols = [c for c in DATA_SHEET_COLUMNS if c in combined.columns]
    return combined[cols]


SCORING_DETAIL_COLUMN_MAP = {
    "Commodity": "Commodity", "Region": "Region", "Horizon": "Horizon",
    "Technique": "Method", "Composite Score": "Score", "Rank": "Rank",
    "MAPE Accuracy %": "mape (100-MAPE)", "Directional Accuracy %": "Directional Accuracy %",
    "Forecast Dynamism": "Forecast Dynamism", "Recency Score": "Recency Score",
    "Flat-line Penalty": "Flat-line Penalty", "Outlier Penalty": "Outlier Penalty",
}


def build_scoring_detail_sheet(batch_results: dict) -> pd.DataFrame:
    """One row per commodity x horizon x technique, trimmed down to this
    sheet's own 12-column shape (Commodity..Outlier Penalty) via
    SCORING_DETAIL_COLUMN_MAP -- deliberately narrower than recommend.
    build_summary_df's full per-commodity review_summary_file.xlsx schema
    (Eligible Y/N, Disqualification/Ineligibility Reason, Normalized Bias
    Penalty, Mean MPE, Evaluated Cycles, Insufficient History, Benchmark
    Accuracy -- those stay internal-review-only, in the per-commodity files,
    not this cross-commodity sheet). Reuses build_summary_df verbatim for
    the values themselves (same function that builds the per-commodity
    file) rather than re-deriving them, then selects/renames down."""
    region_lookup = _load_region_lookup()
    parts = []
    for (commodity_id, horizon_bucket), run_result in batch_results.items():
        if horizon_bucket not in HORIZON_TO_DETAIL_ATTR:
            continue
        if not run_result.score_results:
            continue
        benchmark_accuracy = None
        if run_result.beroe_benchmark is not None:
            detail_attr = HORIZON_TO_DETAIL_ATTR[horizon_bucket]
            bench_df = getattr(run_result.beroe_benchmark, detail_attr)
            if bench_df is not None and not bench_df.empty:
                benchmark_accuracy = bench_df["Row Accuracy"].mean()
        parts.append(build_summary_df(
            commodity_id=commodity_id, horizon_bucket=horizon_bucket,
            score_results=run_result.score_results, benchmark_accuracy=benchmark_accuracy,
            region=region_lookup.get(commodity_id), benchmark_score=run_result.benchmark_score,
        ))
    if not parts:
        return pd.DataFrame()
    full_df = pd.concat(parts, ignore_index=True)
    return full_df[list(SCORING_DETAIL_COLUMN_MAP.keys())].rename(columns=SCORING_DETAIL_COLUMN_MAP)


def _merge_scoring_detail_with_existing(scoring_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Same accumulate-and-replace-per-commodity pattern as _merge_with_
    existing, keyed on this sheet's own 'Commodity' column (not
    'commodity_name' -- build_summary_df uses a different naming convention)."""
    if not Path(output_path).is_file():
        return scoring_df
    try:
        existing_df = pd.read_excel(output_path, sheet_name="Scoring Detail")
    except Exception:
        logger.warning("Scoring Detail: couldn't read existing sheet from %s -- treating as no prior data", output_path)
        return scoring_df
    new_commodity_ids = set(scoring_df["Commodity"].unique()) if not scoring_df.empty else set()
    kept = existing_df[~existing_df["Commodity"].isin(new_commodity_ids)]
    return pd.concat([kept, scoring_df], ignore_index=True)


def _merge_runtime_with_existing(runtime_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Same accumulate-and-replace-per-commodity pattern, for the
    Technique Runtime sheet."""
    if not Path(output_path).is_file():
        return runtime_df
    try:
        existing_df = pd.read_excel(output_path, sheet_name="Technique Runtime")
    except Exception:
        logger.warning("Technique Runtime: couldn't read existing sheet from %s -- treating as no prior data", output_path)
        return runtime_df
    new_commodity_ids = set(runtime_df["Commodity"].unique()) if not runtime_df.empty else set()
    kept = existing_df[~existing_df["Commodity"].isin(new_commodity_ids)]
    return pd.concat([kept, runtime_df], ignore_index=True)


def _merge_with_existing(data_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Accumulates this batch's Data - All Horizons rows into whatever's
    already at output_path, instead of overwriting it: a commodity being
    rerun REPLACES its old rows (no stale duplicates piling up), every
    other commodity's rows are left untouched."""
    if not Path(output_path).is_file():
        return data_df

    new_commodity_ids = set(data_df["commodity_name"].unique())
    try:
        existing_df = pd.read_excel(output_path, sheet_name=DATA_SHEET_NAME)
    except Exception:
        logger.warning("%s: couldn't read existing sheet from %s -- treating as no prior data", DATA_SHEET_NAME, output_path)
        return data_df

    kept = existing_df[~existing_df["commodity_name"].isin(new_commodity_ids)]
    return pd.concat([kept, data_df], ignore_index=True)


def _round_for_display(df: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    """Rounds every numeric (float) column to `decimals` places -- applied
    only at the final write step, never during computation, so any
    aggregate derived from Data - All Horizons is always computed from
    full-precision values, not from already-rounded ones (which would
    introduce a small, avoidable extra rounding error)."""
    if df.empty:
        return df
    float_cols = df.select_dtypes(include="float").columns
    return df.assign(**{c: df[c].round(decimals) for c in float_cols})


def write_consolidated_summary(batch_results: dict, output_path: Path) -> None:
    data_df = build_data_sheet(batch_results)
    data_df = _merge_with_existing(data_df, output_path)

    # Every commodity currently accumulated in the file (not just this run's)
    # -- same set the Data sheet reflects, so Driver Selection always shows
    # every already-trained commodity, kept in sync.
    all_commodity_ids = data_df["commodity_name"].unique()
    driver_selection_df = build_driver_selection_sheet(sorted(all_commodity_ids))

    scoring_detail_df = build_scoring_detail_sheet(batch_results)
    scoring_detail_df = _merge_scoring_detail_with_existing(scoring_detail_df, output_path)

    runtime_df = build_runtime_sheet(batch_results)
    runtime_df = _merge_runtime_with_existing(runtime_df, output_path)

    data_df = _round_for_display(data_df)
    scoring_detail_df = _round_for_display(scoring_detail_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data_df.to_excel(writer, sheet_name=DATA_SHEET_NAME, index=False)
        driver_selection_df.to_excel(writer, sheet_name="Driver Selection", index=False)
        scoring_detail_df.to_excel(writer, sheet_name="Scoring Detail", index=False)
        runtime_df.to_excel(writer, sheet_name="Technique Runtime", index=False)

    # Widen the Driver Selection columns -- the driver-name lists are long,
    # unreadable at Excel's default column width.
    from openpyxl import load_workbook
    wb = load_workbook(output_path)
    ws = wb["Driver Selection"]
    for col, width in (
        ("A", 16), ("B", 16), ("C", 22), ("D", 90), ("E", 20), ("F", 60),
        ("G", 22), ("H", 22), ("I", 90), ("J", 22), ("K", 90),
    ):
        ws.column_dimensions[col].width = width
    ws2 = wb["Scoring Detail"]
    for col, width in (("A", 16), ("B", 16), ("C", 12), ("D", 26), ("F", 45)):
        ws2.column_dimensions[col].width = width
    ws3 = wb["Technique Runtime"]
    for col, width in (("A", 16), ("B", 16), ("C", 12), ("D", 26), ("E", 18)):
        ws3.column_dimensions[col].width = width
    wb.save(output_path)
