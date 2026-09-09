"""
The technique-selection-history file: per Commodity × Region × Forecast
Cycle, outputs/technique_selection_history.xlsx records which technique was
confirmed. The next cycle defaults to that same technique unless a full
refresh (driver reselection from scratch) is explicitly requested.

This is a DIFFERENT grain from decision_log.py's audit trail (Commodity x
Horizon, "what did an analyst just decide and why"): this file adds Region
and Forecast Cycle, and exists to answer a forward-looking question next
cycle — "what did we run last time, so we default to it again" — not to
re-litigate this cycle's justification. In practice this module is fed BY
decision_log.py's resolved decision (see record_selection's `decision`
param) at the point generate_outputs.py writes the final output for a
cycle; it is not an independent decision source.

`forecast_cycle` is caller-supplied (e.g. a "2026-08" or "Aug-2026" label),
not auto-generated — there's no scheduler yet, so nothing in this codebase
yet knows what "the current cycle" is. Append-only, same convention as
decision_log.py, for the same reason: the full history of what was
selected each cycle has to survive across runs, not just the latest row.
"""

from __future__ import annotations

import pandas as pd

from config_loader import CommodityConfig, PROJECT_ROOT

SELECTION_HISTORY_XLSX = PROJECT_ROOT / "outputs" / "technique_selection_history.xlsx"

SELECTION_HISTORY_COLUMNS = [
    "Commodity", "Region", "Horizon", "Forecast Cycle", "Technique", "Action", "Analyst", "Timestamp",
]


def load_selection_history(path=SELECTION_HISTORY_XLSX) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SELECTION_HISTORY_COLUMNS)
    df = pd.read_excel(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def record_selection(
    commodity: CommodityConfig, horizon_bucket: str, forecast_cycle: str, decision, path=SELECTION_HISTORY_XLSX,
) -> pd.DataFrame:
    """
    decision: a decision_log.DecisionRecord (the resolved active decision
    for this commodity x horizon — see decision_log.get_current_decision).
    Records nothing and raises if decision.technique is None (nothing was
    ever confirmed, overridden, or auto-recommended for this cycle — there
    is no technique to remember for next time).

    REPLACES any existing row for the same (Commodity, Region, Horizon,
    Forecast Cycle) rather than appending a new one -- forecast_cycle
    defaults to the current calendar month (run_horizon.py), so every
    run_batch() call within the same month naturally shares one cycle key.
    Without this, re-running the same cycle (a real, ordinary thing to do —
    a data fix, a scoring correction, a rerun after an interrupted batch)
    silently piled up duplicate rows for that one cycle instead of updating
    it, which is what "record what THIS cycle selected" actually means.
    """
    if decision.technique is None:
        raise ValueError(
            f"record_selection: {commodity.id}/{horizon_bucket}/{forecast_cycle} has no resolved "
            f"technique (decision.technique is None) — nothing to record."
        )

    history_df = load_selection_history(path)
    same_cycle = (
        (history_df["Commodity"] == commodity.id) & (history_df["Horizon"] == horizon_bucket)
        & (history_df["Forecast Cycle"] == forecast_cycle)
    )
    kept = history_df[~same_cycle]
    row = {
        "Commodity": commodity.id, "Region": commodity.region, "Horizon": horizon_bucket,
        "Forecast Cycle": forecast_cycle, "Technique": decision.technique, "Action": decision.action,
        "Analyst": decision.analyst, "Timestamp": pd.Timestamp.now(),
    }
    history_df = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_excel(path, index=False)
    return history_df


def get_default_technique(commodity_id: str, horizon_bucket: str, path=SELECTION_HISTORY_XLSX) -> "str | None":
    """
    The technique the LAST recorded forecast cycle used for this commodity
    x horizon — what the next cycle should default to unless a full
    refresh is explicitly requested. Returns None if nothing's ever been
    recorded. Rows are ordered by Timestamp (when they were
    actually run), not by the forecast_cycle label's own text, since cycle
    labels are free-form caller-supplied strings with no guaranteed sort
    order.
    """
    history_df = load_selection_history(path)
    matches = history_df[(history_df["Commodity"] == commodity_id) & (history_df["Horizon"] == horizon_bucket)]
    if matches.empty:
        return None
    return matches.sort_values("Timestamp").iloc[-1]["Technique"]
