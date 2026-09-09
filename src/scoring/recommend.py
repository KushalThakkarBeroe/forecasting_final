"""
Model recommendation, confidence tiering, and internal review output.
Consumes the `ScoreResult` objects composite_score.py already produced for
every technique at one commodity x horizon, and picks best + runner-up.

Scope note: this module produces the RECOMMENDATION and the Forecast/
Summary File DATAFRAMES only. Writing them to the internal-review Excel
workbook is `src/pipeline/generate_outputs.py` (stage="review") — kept
separate so the file-writing/output-path design lives in one place,
together with the final-stage output.

Two deliberate additions beyond the base output schema, both
internal-review-only (never part of the client-facing final file):
  1. "Eligible For Recommendation (Y/N)" alongside "Eligible (Y/N)" —
     "Eligible (Y/N)" reflects `disqualified` (the hard-fail outcome), while
     a separate, softer `eligibility_thresholds` check (85%/55%/3 cycles) is
     deliberately NOT merged into disqualification (see composite_score.py's
     docstring). Surfacing both, rather than silently collapsing them,
     preserves that distinction here.
  2. "Normalized Bias Penalty" / "Mean MPE (%)" — already baked into
     Composite Score either way; these two columns just make it auditable
     for the analyst reviewing this output.
  3. "Benchmark Accuracy (%)" — OPTIONAL, populated only when the caller has
     a benchmark_accuracy.py result to pass in; NaN otherwise, not
     fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from config_loader import PROJECT_ROOT

TECHNIQUE_MATRIX_YAML = PROJECT_ROOT / "config" / "technique_matrix.yaml"

HORIZON_DISPLAY = {"short": "Short Term", "medium": "Medium Term", "long": "Long Term"}


def _load_technique_display_names() -> dict[str, str]:
    techniques = yaml.safe_load(TECHNIQUE_MATRIX_YAML.read_text(encoding="utf-8"))["techniques"]
    return {t["name"]: t.get("display_name", t["name"]) for t in techniques}


def _display_name(technique_key: str, names: dict[str, str]) -> str:
    return names.get(technique_key, technique_key)


@dataclass
class RecommendationResult:
    commodity_id: str
    horizon_bucket: str
    n_eligible: int
    best: "object | None"        # ScoreResult, or None if nothing was eligible
    runner_up: "object | None"   # ScoreResult, or None if fewer than 2 eligible
    key_reason: str
    decision_log_status: str


def _generate_key_reason(best, runner_up, n_eligible: int, names: dict[str, str]) -> str:
    cs = best.component_scores
    parts = [
        f"Highest composite score ({best.composite_score:.1f}, {best.confidence_tier}) "
        f"among {n_eligible} eligible technique{'s' if n_eligible != 1 else ''}."
    ]

    strengths = []
    if cs.accuracy_score >= 90:
        strengths.append(f"strong MAPE accuracy ({cs.accuracy_score:.1f}%)")
    if cs.directional_accuracy_score >= 70:
        strengths.append(f"strong directional accuracy ({cs.directional_accuracy_score:.1f}%)")
    if cs.dynamism_score >= 70:
        strengths.append(f"healthy forecast dynamism ({cs.dynamism_score:.1f})")
    if strengths:
        parts.append("Driven by " + ", ".join(strengths) + ".")

    penalty_notes = []
    if best.flat_line_penalty != 0:
        penalty_notes.append(f"flat-line penalty ({best.flat_line_penalty:+.0f})")
    if best.outlier_cap_penalty != 0:
        penalty_notes.append(f"outlier-cap penalty ({best.outlier_cap_penalty:+.0f})")
    if best.normalized_bias_penalty != 0:
        penalty_notes.append(f"normalized-bias penalty ({best.normalized_bias_penalty:+.0f}, mean MPE {best.mean_mpe_pct:.1f}%)")
    if penalty_notes:
        parts.append("Note: " + ", ".join(penalty_notes) + " applied.")

    if runner_up is not None:
        margin = best.composite_score - runner_up.composite_score
        parts.append(
            f"Runner-up ({_display_name(runner_up.technique, names)}, "
            f"{runner_up.composite_score:.1f}) trails by {margin:.1f} points."
        )

    return " ".join(parts)


def recommend(commodity_id: str, horizon_bucket: str, score_results: "list") -> RecommendationResult:
    """
    score_results: every technique's ScoreResult (composite_score.py) for
    this commodity x horizon_bucket — include disqualified/insufficient-
    history ones too, they're simply excluded from best/runner-up here (and
    still show up in build_summary_df).
    """
    names = _load_technique_display_names()
    eligible = sorted(
        (r for r in score_results if r.eligible_for_recommendation),
        key=lambda r: r.composite_score, reverse=True,
    )
    n_eligible = len(eligible)
    best = eligible[0] if n_eligible >= 1 else None
    runner_up = eligible[1] if n_eligible >= 2 else None

    if best is None:
        key_reason = (
            "No technique met the eligibility thresholds for this commodity x horizon "
            "(MAPE Accuracy, Directional Accuracy, or evaluated-cycle minimums) — "
            "see the Summary File's Eligible For Recommendation / Disqualification Reason columns."
        )
        decision_log_status = "Pending — no eligible technique to default to"
    else:
        key_reason = _generate_key_reason(best, runner_up, n_eligible, names)
        decision_log_status = "Pending (auto-defaulted to top score; awaiting analyst review)"

    return RecommendationResult(
        commodity_id=commodity_id, horizon_bucket=horizon_bucket, n_eligible=n_eligible,
        best=best, runner_up=runner_up, key_reason=key_reason, decision_log_status=decision_log_status,
    )


def build_summary_df(
    commodity_id: str,
    horizon_bucket: str,
    score_results: "list",
    benchmark_accuracy: "float | None" = None,
    region: "str | None" = None,
    benchmark_score: "object | None" = None,
) -> pd.DataFrame:
    """One row per technique, the Summary File schema + the 4
    internal-review-only additions documented in this module's docstring.

    benchmark_accuracy: ONE number (mean Row Accuracy of Beroe's own
    forecast, scored against our real price -- see beroe_benchmark.py's
    module docstring), the same value on every row. Not per-technique:
    Benchmark isn't "this technique's own benchmark", it's Beroe's own
    accuracy, independent of which of our techniques a given row describes
    -- matches how it's already shown in generate_consolidated_summary.py's
    Accuracy by Horizon sheet (one Benchmark column, not one per technique).

    benchmark_score: optional composite_score.ScoreResult for Benchmark
    itself (composite_score.score_benchmark_result) -- when given, appends
    ONE extra "Benchmark" row using the SAME composite formula and hard-
    disqualification rule real techniques get, so Benchmark can be compared
    on equal footing rather than by raw accuracy alone. Always Rank 0 (a
    fixed sentinel, not part of the 1/2/3/... sequence -- Benchmark wasn't
    "a technique tried", it's the reference everything else is measured
    against) and "Eligible For Recommendation (Y/N)" is always "N" --
    deliberately never a candidate in recommend()'s best/runner-up pool.
    """
    names = _load_technique_display_names()
    ranked = sorted(score_results, key=lambda r: r.composite_score, reverse=True)
    # Sequential 1..N by position, not pandas' .rank(method="min") -- ties in
    # composite_score (e.g. several disqualified techniques all forced to 0)
    # must NOT share a rank number. sorted() is stable, so ties keep
    # whatever order score_results was given in; each still gets its own
    # unique rank.
    rank_series = range(1, len(ranked) + 1)

    rows = []
    # Benchmark row goes FIRST (Rank 0, ahead of Rank 1's technique), not
    # appended after -- it's the reference point every technique below is
    # measured against, so it reads top-to-bottom as Benchmark -> best ->
    # worst rather than best -> worst -> Benchmark. Always added, even when
    # benchmark_score is None (this commodity has no Beroe benchmark data
    # at all, e.g. aluminum/ferrochrome/gold) -- every commodity x horizon
    # block stays structurally consistent (Benchmark at Rank 0, then 1..N),
    # with every other column simply blank (NaN) rather than the row being
    # silently skipped.
    bcs = benchmark_score.component_scores if benchmark_score is not None else None
    rows.append({
        "Commodity": commodity_id,
        "Region": region,
        "Horizon": HORIZON_DISPLAY[horizon_bucket],
        "Technique": "Benchmark",
        "Eligible (Y/N)": ("N" if benchmark_score.disqualified else "Y") if benchmark_score is not None else None,
        "Disqualification Reason": ("; ".join(benchmark_score.disqualification_reasons) if benchmark_score.disqualified else "") if benchmark_score is not None else None,
        "MAPE Accuracy %": bcs.accuracy_score if bcs else np.nan,
        "Directional Accuracy %": bcs.directional_accuracy_score if bcs else np.nan,
        "Forecast Dynamism": bcs.dynamism_score if bcs else np.nan,
        "Recency Score": bcs.recency_score if bcs else np.nan,
        "Flat-line Penalty": benchmark_score.flat_line_penalty if benchmark_score is not None else np.nan,
        "Outlier Penalty": benchmark_score.outlier_cap_penalty if benchmark_score is not None else np.nan,
        "Composite Score": benchmark_score.composite_score if benchmark_score is not None else np.nan,
        "Rank": 0,
        "Eligible For Recommendation (Y/N)": "N",
        "Ineligibility Reason": "not_a_recommendable_technique",
        "Normalized Bias Penalty": benchmark_score.normalized_bias_penalty if benchmark_score is not None else np.nan,
        "Mean MPE (%)": benchmark_score.mean_mpe_pct if benchmark_score is not None else np.nan,
        "Evaluated Cycles": benchmark_score.evaluated_cycles if benchmark_score is not None else np.nan,
        "Insufficient History (Y/N)": ("Y" if benchmark_score.insufficient_history else "N") if benchmark_score is not None else None,
        "Benchmark Accuracy (%)": benchmark_accuracy if benchmark_accuracy is not None else np.nan,
    })

    for r, rank in zip(ranked, rank_series):
        cs = r.component_scores
        rows.append({
            "Commodity": commodity_id,
            "Region": region,
            "Horizon": HORIZON_DISPLAY[horizon_bucket],
            "Technique": _display_name(r.technique, names),
            "Eligible (Y/N)": "N" if r.disqualified else "Y",
            "Disqualification Reason": "; ".join(r.disqualification_reasons) if r.disqualified else "",
            "MAPE Accuracy %": cs.accuracy_score if cs else np.nan,
            "Directional Accuracy %": cs.directional_accuracy_score if cs else np.nan,
            "Forecast Dynamism": cs.dynamism_score if cs else np.nan,
            "Recency Score": cs.recency_score if cs else np.nan,
            "Flat-line Penalty": r.flat_line_penalty,
            "Outlier Penalty": r.outlier_cap_penalty,
            "Composite Score": r.composite_score,
            "Rank": int(rank),
            # -- internal-review-only additions (see module docstring) --
            "Eligible For Recommendation (Y/N)": "Y" if r.eligible_for_recommendation else "N",
            "Ineligibility Reason": "; ".join(r.ineligibility_reasons) if r.ineligibility_reasons else "",
            "Normalized Bias Penalty": r.normalized_bias_penalty,
            "Mean MPE (%)": r.mean_mpe_pct,
            "Evaluated Cycles": r.evaluated_cycles,
            "Insufficient History (Y/N)": "Y" if r.insufficient_history else "N",
            "Benchmark Accuracy (%)": benchmark_accuracy if benchmark_accuracy is not None else np.nan,
        })

    return pd.DataFrame(rows)


def build_forecast_row_df(recommendation: RecommendationResult) -> pd.DataFrame:
    """Exactly one row, the Forecast File schema."""
    names = _load_technique_display_names()

    def _tech_block(prefix: str, r) -> dict:
        if r is None:
            return {
                f"{prefix} Technique": None, f"{prefix} - MAPE Accuracy %": np.nan,
                f"{prefix} - Directional Accuracy %": np.nan, f"{prefix} - Forecast Dynamism": np.nan,
                f"{prefix} - Recency Score": np.nan, f"{prefix} - Composite Score": np.nan,
            }
        cs = r.component_scores
        return {
            f"{prefix} Technique": _display_name(r.technique, names),
            f"{prefix} - MAPE Accuracy %": cs.accuracy_score,
            f"{prefix} - Directional Accuracy %": cs.directional_accuracy_score,
            f"{prefix} - Forecast Dynamism": cs.dynamism_score,
            f"{prefix} - Recency Score": cs.recency_score,
            f"{prefix} - Composite Score": r.composite_score,
        }

    row = {
        "Commodity": recommendation.commodity_id,
        "Horizon": HORIZON_DISPLAY[recommendation.horizon_bucket],
        **_tech_block("Best", recommendation.best),
        **_tech_block("Runner-Up", recommendation.runner_up),
        "Confidence Tier": recommendation.best.confidence_tier if recommendation.best else None,
        "Key Reason": recommendation.key_reason,
        "Decision Log Status": recommendation.decision_log_status,
    }
    return pd.DataFrame([row])
