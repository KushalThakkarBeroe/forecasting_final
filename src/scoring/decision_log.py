"""
The Decision Log & analyst override: the one deliberate, explicit
checkpoint in an otherwise fully unattended pipeline. Steps 1-12 (ingestion
through scoring/recommendation) run fully unattended; this is the natural,
expected point where an analyst steps in to confirm or override — not a
hard blocker that halts run_batch.py, but not skippable either: the final
output always reflects whatever the Decision Log currently holds
(auto-defaulted to the top score until an analyst acts, updated whenever
they do).

Unlike composite_score.py/recommend.py (pure dataframe-in, dataframe-out,
no file I/O), this module IS an I/O module by necessity: an audit trail
has to survive across separate `run_batch.py` runs, or "confirm/override
with audit trail" is meaningless — an analyst's decision made today must
still be there next time the pipeline runs. Storage is a single
append-only Excel workbook, `outputs/decision_log.xlsx` (sibling to
`outputs/technique_selection_history.xlsx` — a DIFFERENT file: "what
technique did we default to last cycle", not this module's "what did an
analyst actually decide, and why, with a timestamp").

Every confirm/override call appends a new row — existing rows are never
edited or deleted, so the full history of who-decided-what-when is always
recoverable, not just the latest state. "Current" status for a commodity x
horizon is always "the most recent row for that pair," resolved by
get_current_decision/resolve_active_technique below.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml

from config_loader import PROJECT_ROOT

DECISION_LOG_XLSX = PROJECT_ROOT / "outputs" / "decision_log.xlsx"
TECHNIQUE_MATRIX_YAML = PROJECT_ROOT / "config" / "technique_matrix.yaml"

DECISION_LOG_COLUMNS = [
    "Commodity", "Horizon", "Action", "Technique", "Analyst", "Justification", "Timestamp",
]


def _known_technique_keys() -> set[str]:
    techniques = yaml.safe_load(TECHNIQUE_MATRIX_YAML.read_text(encoding="utf-8"))["techniques"]
    return {t["name"] for t in techniques}


@dataclass
class DecisionRecord:
    commodity_id: str
    horizon_bucket: str
    action: str          # "Confirmed" | "Overridden" | "Pending" (no analyst action yet)
    technique: "str | None"
    analyst: "str | None"
    justification: str
    timestamp: "pd.Timestamp | None"


def load_decision_log(path=DECISION_LOG_XLSX) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DECISION_LOG_COLUMNS)
    df = pd.read_excel(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def _append_row(row: dict, path=DECISION_LOG_XLSX) -> pd.DataFrame:
    log_df = load_decision_log(path)
    log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_excel(path, index=False)
    return log_df


def confirm_recommendation(
    commodity_id: str, horizon_bucket: str, recommendation, analyst: str,
    justification: str = "", path=DECISION_LOG_XLSX,
) -> DecisionRecord:
    """
    Analyst agrees with recommend()'s top pick. Requires
    recommendation.best is not None (nothing to confirm if no technique was
    eligible — use override_recommendation to force one through with
    justification, or leave it Pending).
    """
    if recommendation.best is None:
        raise ValueError(
            f"confirm_recommendation: {commodity_id}/{horizon_bucket} has no eligible "
            f"'best' technique to confirm (recommend() returned none) — use "
            f"override_recommendation if the analyst wants to force a specific technique."
        )
    if not analyst or not analyst.strip():
        raise ValueError("confirm_recommendation: analyst name is required.")

    technique = recommendation.best.technique
    timestamp = pd.Timestamp.now()
    _append_row({
        "Commodity": commodity_id, "Horizon": horizon_bucket, "Action": "Confirmed",
        "Technique": technique, "Analyst": analyst.strip(), "Justification": justification.strip(),
        "Timestamp": timestamp,
    }, path=path)
    return DecisionRecord(
        commodity_id=commodity_id, horizon_bucket=horizon_bucket, action="Confirmed",
        technique=technique, analyst=analyst.strip(), justification=justification.strip(), timestamp=timestamp,
    )


def override_recommendation(
    commodity_id: str, horizon_bucket: str, technique: str, analyst: str, justification: str,
    path=DECISION_LOG_XLSX,
) -> DecisionRecord:
    """
    Analyst picks a technique OTHER than (or despite) recommend()'s top
    pick — e.g. every technique was disqualified but the analyst wants to
    proceed with one on judgment, or the top score doesn't match their own
    read of the market. Unlike confirm, justification is REQUIRED.
    """
    if not analyst or not analyst.strip():
        raise ValueError("override_recommendation: analyst name is required.")
    if not justification or not justification.strip():
        raise ValueError("override_recommendation: justification is required for an override.")
    known = _known_technique_keys()
    if technique not in known:
        raise ValueError(f"override_recommendation: technique {technique!r} not found in {TECHNIQUE_MATRIX_YAML} (known: {sorted(known)})")

    timestamp = pd.Timestamp.now()
    _append_row({
        "Commodity": commodity_id, "Horizon": horizon_bucket, "Action": "Overridden",
        "Technique": technique, "Analyst": analyst.strip(), "Justification": justification.strip(),
        "Timestamp": timestamp,
    }, path=path)
    return DecisionRecord(
        commodity_id=commodity_id, horizon_bucket=horizon_bucket, action="Overridden",
        technique=technique, analyst=analyst.strip(), justification=justification.strip(), timestamp=timestamp,
    )


def get_current_decision(
    commodity_id: str, horizon_bucket: str, recommendation, log_df: "pd.DataFrame | None" = None, path=DECISION_LOG_XLSX,
) -> DecisionRecord:
    """
    The active decision for this commodity x horizon: the most recent
    Confirmed/Overridden row if one exists, else falls back to the
    auto-default (recommendation.best, action="Pending") — auto-defaulted
    to the top score until an analyst acts. log_df may be passed in (e.g.
    already loaded once for a batch run) to avoid re-reading the workbook
    per commodity x horizon.
    """
    if log_df is None:
        log_df = load_decision_log(path)

    matches = log_df[(log_df["Commodity"] == commodity_id) & (log_df["Horizon"] == horizon_bucket)]
    if not matches.empty:
        latest = matches.sort_values("Timestamp").iloc[-1]
        return DecisionRecord(
            commodity_id=commodity_id, horizon_bucket=horizon_bucket, action=latest["Action"],
            technique=latest["Technique"], analyst=latest["Analyst"], justification=latest["Justification"] or "",
            timestamp=latest["Timestamp"],
        )

    if recommendation.best is not None:
        return DecisionRecord(
            commodity_id=commodity_id, horizon_bucket=horizon_bucket, action="Pending",
            technique=recommendation.best.technique, analyst=None, justification="", timestamp=None,
        )
    return DecisionRecord(
        commodity_id=commodity_id, horizon_bucket=horizon_bucket, action="Pending",
        technique=None, analyst=None, justification="", timestamp=None,
    )


def resolve_active_technique(commodity_id: str, horizon_bucket: str, recommendation, log_df: "pd.DataFrame | None" = None, path=DECISION_LOG_XLSX) -> "str | None":
    """
    The single seam generate_outputs.py (stage="final") needs: which
    technique's forecast to actually publish for this commodity x horizon.
    Returns None only when there is neither a logged decision nor an
    eligible recommendation to fall back to.
    """
    return get_current_decision(commodity_id, horizon_bucket, recommendation, log_df=log_df, path=path).technique


def decision_log_status_str(decision: DecisionRecord) -> str:
    """Builds the Forecast File's 'Decision Log Status' text:
    'Confirmed/Overridden ... by <analyst> on <date>', or the pending-
    default text when nothing has been logged yet."""
    if decision.action == "Pending":
        if decision.technique is None:
            return "Pending — no eligible technique to default to"
        return "Pending (auto-defaulted to top score; awaiting analyst review)"
    date_str = decision.timestamp.strftime("%d-%b-%Y") if decision.timestamp is not None else "unknown date"
    return f"{decision.action} by {decision.analyst} on {date_str}"
