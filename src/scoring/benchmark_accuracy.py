"""
Benchmark accuracy: computes forecast accuracy from a forecast-vintage-
history workbook.

Input shape: one row per target month ("Forecasted Period"), one column per
forecast-vintage date (e.g. 'Feb-2025', ..., 'Apr-2026'). For a target
month at/before a vintage's own cutoff, that column holds the realized
actual; beyond the cutoff, it holds what was predicted at that time.
Comparing an old vintage's predictions against what a later vintage shows
as the realized actual for the same months is what this module computes.

Scope note: this module computes the accuracy dataframes only — it does not
write a master Excel workbook. That's generate_outputs.py's job, kept
separate so output-writing logic lives in one place.

Not currently wired into commodities.yaml — callers pass a file path
explicitly (see compute_forecast_accuracy's signature). Wire it in as e.g.
a `forecast_history_file` field alongside `data.file` once every
commodity's forecast-vintage-history file is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_COLUMN_NAME = "Forecasted Period"


def _build_sort_order_map(predicted_cols: list[str]) -> dict[str, int]:
    sorted_cols = sorted(predicted_cols, key=lambda x: pd.to_datetime(x, format="%b-%Y"))
    return {col: i + 1 for i, col in enumerate(sorted_cols)}


def _row_metrics(actual: float, predicted: float) -> dict:
    actual = float(actual)
    predicted = float(predicted)
    abs_error = abs(actual - predicted)
    mse = (actual - predicted) ** 2
    ape = abs((actual - predicted) / actual) * 100
    # Floored at 0, matching src/models/_common.py's _row_metrics (real
    # techniques) -- without this floor, a forecast miss > 100% (ape > 100)
    # produces a NEGATIVE Row Accuracy, which real techniques never show
    # and which silently drags down every mean that includes it (e.g. the
    # "Accuracy by Horizon" pivot, Scoring Detail's Benchmark accuracy).
    accuracy = max(0.0, 100 - ape)
    return {
        "Row MAE": round(abs_error, 4),
        "Row MSE": round(mse, 4),
        "Row RMSE": round(np.sqrt(mse), 4),
        "Row MAPE (%)": round(ape, 4),
        "Row Accuracy": round(accuracy, 4),
    }


def _build_benchmark_rows(
    df: pd.DataFrame,
    actual_col: str,
    actual_cutoff,
    predicted_cols: list[str],
    commodity_name: str,
    sort_order_map: dict[str, int],
) -> pd.DataFrame:
    """Row-level detail: for each forecast vintage, the overlap between its
    predicted window and realized actuals. Computes both DA methods for
    every row — Directional Accuracy = Rolling MoM (baseline shifts to the
    previous row's prediction each step), Abs Directional Accuracy = Fixed
    Anchor (baseline is always the vintage's own cycle-start actual). Which
    one a horizon uses is a config/scoring.yaml concern (horizon_overrides),
    not decided here — both are always computed."""
    actual_cutoff = pd.Timestamp(actual_cutoff)
    detail_dfs = []

    for pred_col in predicted_cols:
        pred_start = pd.Timestamp(pd.to_datetime(pred_col, format="%b-%Y"))
        baseline_fp = pred_start - pd.DateOffset(months=1)

        baseline_row = df[df[DATE_COLUMN_NAME] == baseline_fp]
        if baseline_row.empty:
            continue
        # pd.to_numeric coerces a garbage entry (e.g. a stray typo like "s"
        # where a price belongs) to NaN instead of leaving it as a string --
        # without this, a single bad cell anywhere in the source file poisons
        # this column's dtype to "object", which crashes .round() much later
        # for EVERY vintage, not just the one row the bad value is actually
        # in. Treating a garbage value as missing (same as a blank cell) is
        # the same non-blocking, per-row-skip behavior already given to
        # genuinely missing data below.
        baseline_value = pd.to_numeric(pd.Series(baseline_row[actual_col].values[:1]), errors="coerce").iloc[0]
        if pd.isna(baseline_value):
            continue

        mask = (df[DATE_COLUMN_NAME] >= pred_start) & (df[DATE_COLUMN_NAME] <= actual_cutoff)
        overlap_df = df[mask][[DATE_COLUMN_NAME, pred_col, actual_col]].copy()
        overlap_df.columns = [DATE_COLUMN_NAME, "Predicted", "Actual"]
        overlap_df["Predicted"] = pd.to_numeric(overlap_df["Predicted"], errors="coerce")
        overlap_df["Actual"] = pd.to_numeric(overlap_df["Actual"], errors="coerce")
        overlap_df["Predicted From"] = pred_col
        overlap_df = overlap_df.dropna(subset=["Predicted", "Actual"]).reset_index(drop=True)
        if overlap_df.empty:
            continue

        dir_indicators = []
        abs_dir_indicators = []
        for i in range(len(overlap_df)):
            predicted = float(overlap_df.loc[i, "Predicted"])
            actual = float(overlap_df.loc[i, "Actual"])

            baseline_mom = float(baseline_value) if i == 0 else float(overlap_df.loc[i - 1, "Predicted"])
            pred_dir_mom = 1 if predicted > baseline_mom else -1
            act_dir_mom = 1 if actual > baseline_mom else -1
            dir_indicators.append(1 if pred_dir_mom == act_dir_mom else 0)

            baseline_abs = float(baseline_value)
            pred_dir_abs = 1 if predicted > baseline_abs else -1
            act_dir_abs = 1 if actual > baseline_abs else -1
            abs_dir_indicators.append(1 if pred_dir_abs == act_dir_abs else 0)

        overlap_df["Directional Accuracy"] = pd.array(dir_indicators, dtype="Int64")
        overlap_df["Abs Directional Accuracy"] = pd.array(abs_dir_indicators, dtype="Int64")
        overlap_df["Abs Error"] = (overlap_df["Actual"] - overlap_df["Predicted"]).abs()
        overlap_df["APE (%)"] = (overlap_df["Abs Error"] / overlap_df["Actual"].abs()) * 100

        row_m_df = pd.DataFrame(
            overlap_df.apply(lambda r: _row_metrics(r["Actual"], r["Predicted"]), axis=1).tolist()
        )
        for col in row_m_df.columns:
            overlap_df[col] = row_m_df[col].values

        detail_dfs.append(overlap_df)

    if not detail_dfs:
        return pd.DataFrame()

    all_detail = pd.concat(detail_dfs, ignore_index=True)
    all_detail.rename(columns={DATE_COLUMN_NAME: "Date"}, inplace=True)
    all_detail["Date"] = all_detail["Date"].dt.strftime("%b-%Y")
    all_detail["Predicted"] = all_detail["Predicted"].round(4)
    all_detail["Actual"] = all_detail["Actual"].round(4)
    all_detail["Abs Error"] = all_detail["Abs Error"].round(4)
    all_detail["APE (%)"] = all_detail["APE (%)"].round(4)

    all_detail.insert(0, "commodity_name", commodity_name)
    all_detail.insert(1, "sort_order", all_detail["Predicted From"].map(sort_order_map))

    col_order = [
        "commodity_name", "sort_order", "Predicted From", "Date",
        "Predicted", "Actual", "Abs Error", "APE (%)",
        "Directional Accuracy", "Abs Directional Accuracy",
        "Row MAE", "Row MSE", "Row RMSE", "Row MAPE (%)", "Row Accuracy",
    ]
    return all_detail[[c for c in col_order if c in all_detail.columns]].reset_index(drop=True)


def _build_range_table(
    benchmark_df: pd.DataFrame,
    commodity_name: str,
    actual_cutoff,
    cycles_start,
    cycles_end,
    offset: int,
    n_months: int,
    m_start: int,
    sort_order_map: dict[str, int],
) -> pd.DataFrame:
    """Wide table: one row per forecast cycle, one column-group per M-step
    (M{m_start}..M{m_start+n_months-1}), sliced from benchmark_df by
    offset/n_months. offset/n_months/m_start are the horizon knobs — e.g.
    Short Term = offset 0, n_months 3, m_start 1 (M1-M3)."""
    if benchmark_df.empty:
        return pd.DataFrame()

    actual_cutoff = pd.Timestamp(actual_cutoff)
    cycles_start = pd.Timestamp(cycles_start)
    cycles_end = pd.Timestamp(cycles_end)

    st = benchmark_df.copy()
    st["Date_dt"] = pd.to_datetime(st["Date"], format="%b-%Y")
    st["Predicted From_dt"] = pd.to_datetime(st["Predicted From"], format="%b-%Y")

    df_filtered = st[(st["Predicted From_dt"] >= cycles_start) & (st["Predicted From_dt"] <= cycles_end)]

    records = []
    for cycle_dt in sorted(df_filtered["Predicted From_dt"].unique()):
        df_cycle = df_filtered[df_filtered["Predicted From_dt"] == cycle_dt].sort_values("Date_dt")
        df_cycle = df_cycle.iloc[offset: offset + n_months]
        df_eval = df_cycle[(df_cycle["Date_dt"] <= actual_cutoff) & (df_cycle["Actual"].notna())]

        months_available = len(df_eval)
        if months_available == 0:
            continue

        cycle_label = cycle_dt.strftime("%b-%Y")
        month_cols = {}
        for i in range(n_months):
            m_label = m_start + i
            if i < len(df_eval):
                r = df_eval.iloc[i]
                month_cols[f"M{m_label} Date"] = r["Date"]
                month_cols[f"M{m_label} Accuracy"] = round(float(r["Row Accuracy"]), 2)
                month_cols[f"M{m_label} MAPE (%)"] = round(float(r["Row MAPE (%)"]), 2)
                month_cols[f"M{m_label} DA"] = int(r["Directional Accuracy"]) if pd.notna(r["Directional Accuracy"]) else None
                month_cols[f"M{m_label} Abs DA"] = int(r["Abs Directional Accuracy"]) if pd.notna(r["Abs Directional Accuracy"]) else None
            else:
                month_cols[f"M{m_label} Date"] = "—"
                month_cols[f"M{m_label} Accuracy"] = None
                month_cols[f"M{m_label} MAPE (%)"] = None
                month_cols[f"M{m_label} DA"] = None
                month_cols[f"M{m_label} Abs DA"] = None

        records.append({
            "commodity_name": commodity_name,
            "sort_order": sort_order_map.get(cycle_label),
            "Predicted From": cycle_label,
            "Months Available": f"{months_available}/{n_months}M",
            **month_cols,
            "Avg Accuracy (%)": round(df_eval["Row Accuracy"].mean(), 2),
            "Avg MAPE (%)": round(df_eval["Row MAPE (%)"].mean(), 2),
            "Avg DA (%)": round(df_eval["Directional Accuracy"].mean() * 100, 1),
            "Avg Abs DA (%)": round(df_eval["Abs Directional Accuracy"].mean() * 100, 1),
            "Avg MAE": round(df_eval["Row MAE"].mean(), 2),
            "Avg RMSE": round(df_eval["Row RMSE"].mean(), 2),
        })

    return pd.DataFrame(records)


def _build_horizon_detail(
    benchmark_df: pd.DataFrame,
    commodity_name: str,
    cycles_start,
    cycles_end,
    offset: int,
    n_months: int,
    m_start: int,
    sort_order_map: dict[str, int],
) -> pd.DataFrame:
    """Row-level detail sliced to one horizon (ST/MT/LT/LongerT) -- same
    benchmark_df as _build_range_table, but keeps one row per (cycle,
    target month) instead of collapsing each cycle into one wide row.
    Produces the 4 row-level sheets (ST_Detail/MT_Detail/LT_Detail/
    LongerT_Detail) alongside _build_range_table's 4 wide sheets
    (Short_Term/Medium_Term/Long_Term/Longer_Term)."""
    if benchmark_df.empty:
        return pd.DataFrame()

    cycles_start = pd.Timestamp(cycles_start)
    cycles_end = pd.Timestamp(cycles_end)

    bm = benchmark_df.copy()
    bm["Date_dt"] = pd.to_datetime(bm["Date"], format="%b-%Y")
    bm["Predicted From_dt"] = pd.to_datetime(bm["Predicted From"], format="%b-%Y")
    bm_filtered = bm[(bm["Predicted From_dt"] >= cycles_start) & (bm["Predicted From_dt"] <= cycles_end)]

    records = []
    for cycle_dt in sorted(bm_filtered["Predicted From_dt"].unique()):
        df_cycle = bm_filtered[bm_filtered["Predicted From_dt"] == cycle_dt].sort_values("Date_dt").reset_index(drop=True)
        df_horizon = df_cycle.iloc[offset: offset + n_months].reset_index(drop=True)
        if df_horizon.empty:
            continue

        train_end_str = (cycle_dt - pd.DateOffset(months=1)).strftime("%b-%Y")
        forecast_start = cycle_dt.strftime("%b-%Y")
        window_num = sort_order_map.get(forecast_start)

        for i, row in df_horizon.iterrows():
            step = m_start + i
            actual, pred = row["Actual"], row["Predicted"]
            accuracy_pct = round(max(0.0, 100 - float(row["APE (%)"])), 4) if pd.notna(row.get("APE (%)")) else np.nan
            records.append({
                "commodity_name": commodity_name, "sort_order": window_num, "Model": "Benchmark",
                "Window": window_num, "Train End": train_end_str, "Forecast Start": forecast_start,
                "Step": step, "Date": row["Date"],
                "Predicted": round(float(pred), 4) if pd.notna(pred) else np.nan,
                "Actual": round(float(actual), 4) if pd.notna(actual) else np.nan,
                "Directional Accuracy": row.get("Directional Accuracy", np.nan),
                "Abs Directional Accuracy": row.get("Abs Directional Accuracy", np.nan),
                "Flag": "Actual" if pd.notna(actual) else "No Actual Yet",
                "Abs Error": row.get("Abs Error", np.nan), "APE (%)": row.get("APE (%)", np.nan),
                "Accuracy (%)": accuracy_pct, "Row MAE": row.get("Row MAE", np.nan),
                "Row MSE": row.get("Row MSE", np.nan), "Row RMSE": row.get("Row RMSE", np.nan),
                "Row MAPE (%)": row.get("Row MAPE (%)", np.nan), "Row Accuracy": row.get("Row Accuracy", np.nan),
            })

    if not records:
        return pd.DataFrame()

    col_order = [
        "commodity_name", "sort_order", "Model", "Window", "Train End", "Forecast Start", "Step", "Date",
        "Predicted", "Actual", "Directional Accuracy", "Abs Directional Accuracy", "Flag",
        "Abs Error", "APE (%)", "Accuracy (%)", "Row MAE", "Row MSE", "Row RMSE", "Row MAPE (%)", "Row Accuracy",
    ]
    result = pd.DataFrame(records)
    return result[[c for c in col_order if c in result.columns]].reset_index(drop=True)


@dataclass
class ForecastAccuracyResult:
    benchmark: pd.DataFrame       # row-level detail, all cycles ("Benchmark_Summary" sheet)
    short_term: pd.DataFrame      # wide table M1-M3 (rolling-MoM DA is 'Avg DA (%)')
    medium_term: pd.DataFrame     # wide table M4-M6
    long_term: pd.DataFrame       # wide table M7-M18 (fixed-anchor DA is 'Avg Abs DA (%)')
    longer_term: pd.DataFrame     # wide table M4-M18
    st_detail: pd.DataFrame       # row-level detail, M1-M3 only
    mt_detail: pd.DataFrame       # row-level detail, M4-M6 only
    lt_detail: pd.DataFrame       # row-level detail, M7-M18 only
    longer_detail: pd.DataFrame   # row-level detail, M4-M18 only


def compute_forecast_accuracy(
    df: pd.DataFrame,
    actual_col: str,
    actual_cutoff: str,
    predicted_cols: list[str],
    commodity_name: str,
    short_term_cycles_start: str = "2025-01-01",
    short_term_cycles_end: str = "2026-03-01",
    short_term_forecast_months: int = 3,
    medium_term_cycles_start: str = "2025-01-01",
    medium_term_cycles_end: str = "2026-03-01",
    medium_term_forecast_months: int = 3,
    medium_term_offset: int = 3,
    long_term_cycles_start: str = "2025-01-01",
    long_term_cycles_end: str = "2026-03-01",
    long_term_forecast_months: int = 12,
    long_term_offset: int = 6,
    longer_term_cycles_start: str = "2025-01-01",
    longer_term_cycles_end: str = "2026-03-01",
) -> ForecastAccuracyResult:
    """
    df must have DATE_COLUMN_NAME ("Forecasted Period") + one column per
    forecast vintage in predicted_cols (format 'Mon-YYYY', e.g. 'Feb-2025')
    + actual_col (the latest/current vintage, used as ground truth for
    everything up to actual_cutoff).
    """
    sort_order_map = _build_sort_order_map(predicted_cols)

    benchmark_df = _build_benchmark_rows(
        df=df, actual_col=actual_col, actual_cutoff=actual_cutoff,
        predicted_cols=predicted_cols, commodity_name=commodity_name,
        sort_order_map=sort_order_map,
    )
    if benchmark_df.empty:
        return ForecastAccuracyResult(*(pd.DataFrame() for _ in range(9)))

    short_term_df = _build_range_table(
        benchmark_df, commodity_name, actual_cutoff,
        short_term_cycles_start, short_term_cycles_end,
        offset=0, n_months=short_term_forecast_months, m_start=1,
        sort_order_map=sort_order_map,
    )
    medium_term_df = _build_range_table(
        benchmark_df, commodity_name, actual_cutoff,
        medium_term_cycles_start, medium_term_cycles_end,
        offset=medium_term_offset, n_months=medium_term_forecast_months, m_start=medium_term_offset + 1,
        sort_order_map=sort_order_map,
    )
    long_term_df = _build_range_table(
        benchmark_df, commodity_name, actual_cutoff,
        long_term_cycles_start, long_term_cycles_end,
        offset=long_term_offset, n_months=long_term_forecast_months, m_start=long_term_offset + 1,
        sort_order_map=sort_order_map,
    )
    longer_term_df = _build_range_table(
        benchmark_df, commodity_name, actual_cutoff,
        longer_term_cycles_start, longer_term_cycles_end,
        offset=3, n_months=15, m_start=4,
        sort_order_map=sort_order_map,
    )

    st_detail_df = _build_horizon_detail(
        benchmark_df, commodity_name, short_term_cycles_start, short_term_cycles_end,
        offset=0, n_months=short_term_forecast_months, m_start=1, sort_order_map=sort_order_map,
    )
    mt_detail_df = _build_horizon_detail(
        benchmark_df, commodity_name, medium_term_cycles_start, medium_term_cycles_end,
        offset=medium_term_offset, n_months=medium_term_forecast_months, m_start=medium_term_offset + 1,
        sort_order_map=sort_order_map,
    )
    lt_detail_df = _build_horizon_detail(
        benchmark_df, commodity_name, long_term_cycles_start, long_term_cycles_end,
        offset=long_term_offset, n_months=long_term_forecast_months, m_start=long_term_offset + 1,
        sort_order_map=sort_order_map,
    )
    longer_detail_df = _build_horizon_detail(
        benchmark_df, commodity_name, longer_term_cycles_start, longer_term_cycles_end,
        offset=3, n_months=15, m_start=4, sort_order_map=sort_order_map,
    )

    return ForecastAccuracyResult(
        benchmark=benchmark_df,
        short_term=short_term_df,
        medium_term=medium_term_df,
        long_term=long_term_df,
        longer_term=longer_term_df,
        st_detail=st_detail_df,
        mt_detail=mt_detail_df,
        lt_detail=lt_detail_df,
        longer_detail=longer_detail_df,
    )
