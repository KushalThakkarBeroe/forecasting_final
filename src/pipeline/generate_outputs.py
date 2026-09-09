"""
Final output file generation, plus the writing half of the internal review
output recommend.py builds the dataframes for.

Two-stage output design: Steps 1-12 (ingestion through scoring/
recommendation) run fully unattended. Step 12 produces an *internal*
review copy of the Forecast + Summary files — every eligible technique,
plus benchmark accuracy — for an analyst to sanity-check. Step 13 (Decision
Log) is the point an analyst confirms or overrides the top pick — not a
hard blocker, but Step 14's *final*, client-facing output always reflects
whatever the Decision Log currently holds. `generate_outputs()` takes a
`stage` parameter ('review' vs 'final') rather than being single-purpose.

Both stages use the SAME two schemas (Forecast File, Summary File) via
recommend.py's build_forecast_row_df/build_summary_df — they differ only
in WHICH rows/values populate them:
  - stage="review": Forecast File shows the raw Best + Runner-Up
    (recommend()'s own top score); Summary File lists EVERY technique that
    was scored, plus the internal-review-only columns (Eligible For
    Recommendation, benchmark accuracy, etc — see recommend.py's docstring).
  - stage="final": Forecast File's "Best" block is overwritten with
    whatever src/scoring/decision_log.py resolved as the ACTIVE technique
    (which may differ from the raw top score if an analyst overrode it) and
    "Decision Log Status" shows the real Confirmed/Overridden/Pending
    state; Summary File is scoped down to that one technique's row only.

Written to outputs/{commodity_id}/{horizon_bucket}/ as separate
review_*.xlsx / final_*.xlsx workbooks. stage="final" additionally writes a
never-overwritten timestamped copy of both files to that same folder's
history/ subdirectory, on every call -- see _write_history_copy's own
docstring for why only stage="final" gets this.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))

from config_loader import PROJECT_ROOT
from recommend import HORIZON_DISPLAY, _display_name, _load_technique_display_names, build_forecast_row_df, build_summary_df
from decision_log import decision_log_status_str

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def _output_dir(commodity_id: str, horizon_bucket: str, output_root: Path) -> Path:
    d = output_root / commodity_id / horizon_bucket
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_history_copy(out_dir: Path, forecast_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Snapshot copy for comparing runs over time -- final_forecast_file.xlsx/
    final_summary_file.xlsx at their normal fixed path stay exactly as they
    are (the Decision Log, and anyone pointing at "the current file", keep
    working unchanged); this writes an ADDITIONAL, never-overwritten copy per
    run into history/, so an older run's actual published numbers are still
    there to open and compare against a newer one. Both files from the same
    call share one timestamp, so a forecast/summary pair from the same run
    are easy to match up. Not applied to review_*.xlsx (see generate_review_
    output) or consolidated_summary.xlsx -- see README's "Comparing runs
    over time" section for why the scope stops here."""
    history_dir = out_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    forecast_df.to_excel(history_dir / f"final_forecast_file_{timestamp}.xlsx", index=False)
    summary_df.to_excel(history_dir / f"final_summary_file_{timestamp}.xlsx", index=False)


def generate_review_output(
    commodity_id: str,
    horizon_bucket: str,
    recommendation,
    score_results: list,
    benchmark_accuracy: "float | None" = None,
    region: "str | None" = None,
    output_root: Path = OUTPUTS_ROOT,
    benchmark_score: "object | None" = None,
) -> dict[str, Path]:
    """Step 12's internal review workbooks. score_results should include
    every technique run for this commodity x horizon, disqualified ones
    included — build_summary_df lists all of them.

    benchmark_score: optional composite_score.ScoreResult for Benchmark
    itself (composite_score.score_benchmark_result) — see build_summary_df's
    own docstring. Never part of score_results / recommendation."""
    forecast_df = build_forecast_row_df(recommendation)
    summary_df = build_summary_df(commodity_id, horizon_bucket, score_results, benchmark_accuracy, region, benchmark_score=benchmark_score)

    out_dir = _output_dir(commodity_id, horizon_bucket, output_root)
    forecast_path = out_dir / "review_forecast_file.xlsx"
    summary_path = out_dir / "review_summary_file.xlsx"
    forecast_df.to_excel(forecast_path, index=False)
    summary_df.to_excel(summary_path, index=False)
    return {"forecast_file": forecast_path, "summary_file": summary_path}


def _final_forecast_row(commodity_id: str, horizon_bucket: str, recommendation, decision, active_score) -> pd.DataFrame:
    row = build_forecast_row_df(recommendation).iloc[0].to_dict()
    names = _load_technique_display_names()

    row["Best Technique"] = _display_name(decision.technique, names)
    if active_score is not None and active_score.component_scores is not None:
        cs = active_score.component_scores
        row["Best - MAPE Accuracy %"] = cs.accuracy_score
        row["Best - Directional Accuracy %"] = cs.directional_accuracy_score
        row["Best - Forecast Dynamism"] = cs.dynamism_score
        row["Best - Recency Score"] = cs.recency_score
        row["Best - Composite Score"] = active_score.composite_score
        row["Confidence Tier"] = active_score.confidence_tier
    else:
        # Analyst overrode to a technique that was never actually scored
        # for this commodity x horizon (override_recommendation only
        # validates the technique KEY against technique_matrix.yaml, not
        # that it was run) -- publish the decision, not fabricated metrics.
        row["Best - MAPE Accuracy %"] = np.nan
        row["Best - Directional Accuracy %"] = np.nan
        row["Best - Forecast Dynamism"] = np.nan
        row["Best - Recency Score"] = np.nan
        row["Best - Composite Score"] = np.nan
        row["Confidence Tier"] = None

    row["Decision Log Status"] = decision_log_status_str(decision)
    return pd.DataFrame([row])


def generate_final_output(
    commodity_id: str,
    horizon_bucket: str,
    recommendation,
    score_results: list,
    decision,  # decision_log.DecisionRecord, from decision_log.get_current_decision
    benchmark_accuracy: "float | None" = None,
    region: "str | None" = None,
    output_root: Path = OUTPUTS_ROOT,
    benchmark_score: "object | None" = None,
) -> dict[str, Path]:
    """
    Client-facing final workbooks, scoped to decision.technique (whatever
    src/scoring/decision_log.py resolved as active for this commodity x
    horizon — Confirmed, Overridden, or the Step 12 Pending auto-default),
    plus a Benchmark row (see build_summary_df's docstring) so the client
    can see the chosen technique against Benchmark directly, not just the
    one row. Raises if nothing has been resolved at all (decision.technique
    is None) -- there is nothing to publish to a client in that case.
    """
    if decision.technique is None:
        raise ValueError(
            f"generate_final_output: {commodity_id}/{horizon_bucket} has no resolved technique to "
            f"publish (decision.technique is None — no eligible Step 12 recommendation and no "
            f"analyst override on record)."
        )

    active_score = next((r for r in score_results if r.technique == decision.technique), None)
    forecast_df = _final_forecast_row(commodity_id, horizon_bucket, recommendation, decision, active_score)

    if active_score is not None:
        summary_df = build_summary_df(commodity_id, horizon_bucket, [active_score], benchmark_accuracy, region, benchmark_score=benchmark_score)
    else:
        names = _load_technique_display_names()
        summary_df = pd.DataFrame([{
            "Commodity": commodity_id, "Region": region, "Horizon": HORIZON_DISPLAY[horizon_bucket],
            "Technique": _display_name(decision.technique, names),
            # "Not Scored", not "N/A" -- pandas' default read_excel na_values
            # treats the literal string "N/A" as a null sentinel, which would
            # silently turn this into NaN for any downstream reader.
            "Eligible (Y/N)": "Not Scored", "Disqualification Reason": "Not scored for this commodity x horizon (recorded via analyst override only).",
            "MAPE Accuracy %": np.nan, "Directional Accuracy %": np.nan, "Forecast Dynamism": np.nan,
            "Recency Score": np.nan, "Flat-line Penalty": 0.0, "Outlier Penalty": 0.0,
            "Composite Score": np.nan, "Rank": None,
            "Benchmark Accuracy (%)": benchmark_accuracy if benchmark_accuracy is not None else np.nan,
        }])

    out_dir = _output_dir(commodity_id, horizon_bucket, output_root)
    forecast_path = out_dir / "final_forecast_file.xlsx"
    summary_path = out_dir / "final_summary_file.xlsx"
    forecast_df.to_excel(forecast_path, index=False)
    summary_df.to_excel(summary_path, index=False)
    _write_history_copy(out_dir, forecast_df, summary_df)
    return {"forecast_file": forecast_path, "summary_file": summary_path}


def generate_outputs(
    stage: str,
    commodity_id: str,
    horizon_bucket: str,
    recommendation,
    score_results: list,
    decision=None,  # required for stage="final"
    benchmark_accuracy: "float | None" = None,
    region: "str | None" = None,
    output_root: Path = OUTPUTS_ROOT,
    benchmark_score: "object | None" = None,
) -> dict[str, Path]:
    """Single entry point, dispatching on stage ("review" vs "final").

    benchmark_accuracy: ONE number (Beroe's own mean Row Accuracy for this
    commodity x horizon, from beroe_benchmark.py), applied to every row of
    the Summary File -- not per-technique, see build_summary_df's docstring.
    region: config/commodities.yaml's own `region` field for this commodity.
    benchmark_score: optional composite_score.ScoreResult for Benchmark
    itself (composite_score.score_benchmark_result) -- see build_summary_df's
    docstring. Never part of score_results / recommendation.
    """
    if stage == "review":
        return generate_review_output(
            commodity_id, horizon_bucket, recommendation, score_results,
            benchmark_accuracy=benchmark_accuracy, region=region, output_root=output_root,
            benchmark_score=benchmark_score,
        )
    elif stage == "final":
        if decision is None:
            raise ValueError("generate_outputs: stage='final' requires `decision` (decision_log.get_current_decision(...)).")
        return generate_final_output(
            commodity_id, horizon_bucket, recommendation, score_results, decision,
            benchmark_accuracy=benchmark_accuracy, region=region, output_root=output_root,
            benchmark_score=benchmark_score,
        )
    else:
        raise ValueError(f"generate_outputs: stage must be 'review' or 'final', got {stage!r}")
