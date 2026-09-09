"""
Loads Beroe's real actual prices per commodity from
data/benchmark_history/{commodity_id}/{commodity_id}_benchmark.xlsx --
mirrors data/external/{commodity_id}/{commodity_id}_ext_var.xlsx exactly
(one file per commodity, commodity_id is the folder name, no separate
mapping config to keep in sync). Onboarding a new commodity's Beroe data
is the same motion as onboarding its price/driver data: drop the file in
the folder, nothing else to edit.

Each file's one sheet is a forecast-vintage-history workbook: a header row
(row position NOT assumed fixed -- see _find_header_row) of one column per
forecast-vintage date (descending, most-recent first), then one row per
target month. For a target month at/before a vintage's own cutoff, every
vintage column agrees on the same realized value. We only need that
agreed-upon real value, so the most-recent vintage column is read as
ground truth, up through its own cutoff (one month before the vintage's own
label -- a reporting-lag convention, not the same month).

Two functions read this same shape for two different purposes:
  load_beroe_actuals            -- pre-cutoff portion of the latest vintage
                                    (real settled price) -- used to source
                                    price for data/external/{id}/_ext_var.xlsx
                                    when no independent price file exists.
  load_beroe_forecast_vintages  -- post-cutoff portion of EVERY vintage
                                    (Beroe's own past forecasts) -- used by
                                    beroe_benchmark.py to score Beroe's
                                    forecast accuracy against our own real
                                    price: Predicted = Beroe's forecast,
                                    Actual = our own real price (the same
                                    Actual every real technique's row uses)
                                    -- NOT "our forecast vs Beroe's actual".
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_history"


class BeroeBenchmarkError(Exception):
    """Raised when a commodity has no Beroe benchmark file, or its sheet doesn't match the expected shape."""


@dataclass(frozen=True)
class BeroeActuals:
    commodity_id: str
    actual_series: pd.Series      # index: Timestamp (target month), values: real settled price
    actual_cutoff: pd.Timestamp   # last month actual_series is real data, not Beroe's own forecast


def _find_header_row(file_path: Path, sheet_name: str, max_scan_rows: int = 15) -> int:
    """Scans the first max_scan_rows rows for the vintage-column header row
    -- the row where columns 2+ are date-typed -- instead of assuming a
    fixed row position. Returns its 0-based index, usable directly as
    pandas' `header=`."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    for row_idx in range(len(raw)):
        row_values = raw.iloc[row_idx, 1:]
        date_typed = sum(isinstance(v, (pd.Timestamp, datetime.datetime)) for v in row_values)
        if date_typed >= 2:  # 2+ date-typed cells rules out a stray single date in a metadata row
            return row_idx
    raise BeroeBenchmarkError(
        f"{file_path}::{sheet_name}: couldn't find a vintage-column header row in the first {max_scan_rows} rows"
    )


def _load_raw_sheet(commodity_id: str, benchmark_dir: Path) -> "tuple[pd.DataFrame, list]":
    """Resolves commodity_id -> its one benchmark file/sheet, auto-detects
    the header row, and returns (raw df, vintage_cols) -- shared by
    load_beroe_actuals (keeps the pre-cutoff/actual portion of each column)
    and load_beroe_forecast_vintages (keeps the post-cutoff/forecast
    portion) since both read the exact same underlying sheet."""
    commodity_dir = benchmark_dir / commodity_id
    if not commodity_dir.is_dir():
        raise BeroeBenchmarkError(f"No Beroe benchmark data for '{commodity_id}' -- expected a folder at {commodity_dir}")

    xlsx_files = [f for f in sorted(commodity_dir.glob("*.xlsx")) if not f.name.startswith("~$")]
    if not xlsx_files:
        raise BeroeBenchmarkError(f"{commodity_id}: no .xlsx file found in {commodity_dir}")
    if len(xlsx_files) > 1:
        raise BeroeBenchmarkError(
            f"{commodity_id}: expected exactly one .xlsx file in {commodity_dir}, "
            f"found {len(xlsx_files)}: {[f.name for f in xlsx_files]}"
        )
    file_path = xlsx_files[0]

    sheet_names = pd.ExcelFile(file_path).sheet_names
    if len(sheet_names) != 1:
        raise BeroeBenchmarkError(
            f"{commodity_id}: expected exactly one sheet in {file_path.name}, found {len(sheet_names)}: {sheet_names}"
        )
    sheet_name = sheet_names[0]

    header_row = _find_header_row(file_path, sheet_name)
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

    vintage_cols = [c for c in raw.columns[1:] if isinstance(c, (pd.Timestamp, datetime.datetime))]
    if not vintage_cols:
        raise BeroeBenchmarkError(f"{file_path.name}::{sheet_name}: no forecast-vintage columns found")

    return raw, vintage_cols


def load_beroe_actuals(commodity_id: str, benchmark_dir: Path = BENCHMARK_DIR) -> BeroeActuals:
    raw, vintage_cols = _load_raw_sheet(commodity_id, benchmark_dir)
    date_col = raw.columns[0]

    latest_vintage_raw = max(vintage_cols)  # leftmost column = most recent vintage = ground truth
    latest_vintage_col = pd.Timestamp(latest_vintage_raw)
    actual_cutoff = latest_vintage_col - pd.DateOffset(months=1)

    series = raw.set_index(date_col)[latest_vintage_raw].dropna()
    series.index = pd.DatetimeIndex(series.index)
    # Beyond the cutoff, this column holds Beroe's own forecast (or an
    # unconfirmed flash figure for its own vintage month), not a settled
    # actual -- drop those rows so nothing downstream can mistake it for
    # ground truth.
    series = series[series.index <= actual_cutoff]

    return BeroeActuals(commodity_id=commodity_id, actual_series=series, actual_cutoff=actual_cutoff)


def load_beroe_forecast_vintages(commodity_id: str, benchmark_dir: Path = BENCHMARK_DIR) -> pd.DataFrame:
    """The mirror image of load_beroe_actuals: for EACH vintage column,
    keeps only the portion AFTER that column's own cutoff -- i.e. Beroe's
    own forecast, made from that vintage point, for target months >= the
    vintage's own label. (Cutoff = vintage_date - 1 month, matching
    load_beroe_actuals' convention, so a vintage's own forecast starts
    exactly at its own labeled month.)

    Returns a df shaped like TechniqueResult.matrix_df: DATE_COLUMN_NAME
    ("Forecasted Period") + one column per vintage (named 'Mon-YYYY',
    matching benchmark_accuracy.py's predicted_cols convention), values =
    Beroe's forecast, NaN where that vintage hadn't started forecasting yet.
    """
    raw, vintage_cols = _load_raw_sheet(commodity_id, benchmark_dir)
    date_col = raw.columns[0]
    indexed = raw.set_index(date_col)
    indexed.index = pd.DatetimeIndex(indexed.index)

    out = {"Forecasted Period": indexed.index}
    for vcol in vintage_cols:
        vdate = pd.Timestamp(vcol)
        series = indexed[vcol]
        forecast_only = series.where(indexed.index >= vdate)
        out[vdate.strftime("%b-%Y")] = forecast_only.values

    return pd.DataFrame(out)
