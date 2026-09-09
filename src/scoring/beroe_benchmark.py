"""
Scores Beroe's own past forecasts against our real observed price, fed into
forecast_accuracy/compute_forecast_accuracy unmodified.

Direction matters here: a 'Benchmark' row and a same-window 'Random Forest'
row share the exact same Actual value (our own real price) -- only
Predicted differs (Beroe's forecast vs. our model's forecast). So Benchmark
is Beroe's forecast scored against OUR real price, using the SAME ground
truth every real technique's row already uses -- not "our forecast vs
Beroe's actual".

Consequence: this computation needs neither a specific technique's
matrix_df nor which technique is currently recommended -- it only needs the
commodity's own real price series (external_driver_loader's price_series,
already used as ground truth for every real technique) and Beroe's own
forecast-vintage history (load_beroe_forecast_vintages). Same result
regardless of which horizon_bucket triggered the call, since Beroe's
vintages were never split by horizon the way our own runs are -- each one
already spans as many months out as Beroe forecasted.
"""

from __future__ import annotations

import pandas as pd

from beroe_actuals_loader import BeroeBenchmarkError, load_beroe_forecast_vintages
from benchmark_accuracy import ForecastAccuracyResult, compute_forecast_accuracy

OUR_ACTUAL_COL = "Our Actual"


def compute_beroe_benchmark(commodity_id: str, price_series: "pd.Series") -> "ForecastAccuracyResult | None":
    """
    price_series: the commodity's own real price history (same series every
    real technique is scored against) -- external_driver_loader's
    price_series, indexed by date.

    Returns None (non-fatal) if this commodity has no Beroe benchmark file
    yet (data/benchmark_history/{commodity_id}/) -- matches this project's
    per-commodity non-blocking convention.
    """
    try:
        forecast_df = load_beroe_forecast_vintages(commodity_id)
    except BeroeBenchmarkError:
        return None

    if forecast_df.empty:
        return None

    predicted_cols = [c for c in forecast_df.columns if c != "Forecasted Period"]
    if not predicted_cols:
        return None

    price_lookup = price_series.copy()
    price_lookup.index = pd.DatetimeIndex(price_lookup.index)

    merged = forecast_df.copy()
    merged["Forecasted Period"] = pd.to_datetime(merged["Forecasted Period"])
    merged[OUR_ACTUAL_COL] = merged["Forecasted Period"].map(price_lookup)

    vintage_dates = pd.to_datetime(predicted_cols, format="%b-%Y")
    cycles_start, cycles_end = vintage_dates.min(), vintage_dates.max()
    # Our own latest real price date -- the natural cutoff for OUR ground
    # truth (Beroe's own vintage cutoffs are irrelevant here; we're scoring
    # Beroe's forecast against what WE know really happened).
    actual_cutoff = price_lookup.index.max()

    return compute_forecast_accuracy(
        df=merged,
        actual_col=OUR_ACTUAL_COL,
        actual_cutoff=actual_cutoff,
        predicted_cols=predicted_cols,
        commodity_name=commodity_id,
        short_term_cycles_start=cycles_start, short_term_cycles_end=cycles_end,
        medium_term_cycles_start=cycles_start, medium_term_cycles_end=cycles_end,
        long_term_cycles_start=cycles_start, long_term_cycles_end=cycles_end,
        longer_term_cycles_start=cycles_start, longer_term_cycles_end=cycles_end,
    )


def write_beroe_benchmark_xlsx(result: ForecastAccuracyResult, output_path) -> None:
    """9 sheets, one file per commodity x horizon (matches this pipeline's
    outputs/{commodity_id}/{horizon}/ layout). A sheet is skipped when
    empty."""
    sheets = [
        ("Benchmark_Summary", result.benchmark),
        ("Short_Term", result.short_term),
        ("Medium_Term", result.medium_term),
        ("Long_Term", result.long_term),
        ("Longer_Term", result.longer_term),
        ("ST_Detail", result.st_detail),
        ("MT_Detail", result.mt_detail),
        ("LT_Detail", result.lt_detail),
        ("LongerT_Detail", result.longer_detail),
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets:
            if not sheet_df.empty:
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
