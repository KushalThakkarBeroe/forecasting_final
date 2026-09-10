"""
One consolidated workbook of forward forecasts across many commodities,
many horizons, and (by default) all 3 saved ranks per commodity x horizon
-- the multi-commodity companion to predict_from_saved_model.py's
single-call CLI. Reuses that module's exact per-technique forecasting
dispatch (_forecast_from_bundle) rather than duplicating any forecasting
logic; this file is pure orchestration (which commodities, which horizons,
which ranks, which pickles) plus the workbook layout.

Usage (CLI):
  python src/scripts/generate_consolidated_predictions.py --commodities copper,wheat --horizons short,medium
  python src/scripts/generate_consolidated_predictions.py --all --horizons short,medium,long
  python src/scripts/generate_consolidated_predictions.py --commodities copper --horizons short --ranks 1

Or programmatically:
  from generate_consolidated_predictions import generate_consolidated_predictions
  xlsx_path = generate_consolidated_predictions(["copper", "wheat"], ["short", "medium"])
  xlsx_path = generate_consolidated_predictions(["copper"], ["short"], ranks=[1])

--commodities omitted (or --all) means every commodity that actually HAS at
least one saved model on disk (discovered from saved_models/'s own
subfolders) -- not every commodity in commodities.yaml, most of which have
never been trained at all.

Per (commodity, horizon) pair, forecasts out to that horizon's own upper
bound from config/horizon_defaults.yaml (short=3, medium=6, long=18 for a
monthly commodity) via `ranks` -- defaults to all 3 (the client sees the
top pick and both alternates side by side, never just one number with no
way to judge how much the techniques agree), but can be narrowed to
specific rank(s) via --ranks (CLI) / ranks= (programmatic) when that's
genuinely what's wanted. Rows are then trimmed to that horizon's own
(m_start, m_end) range (e.g. medium keeps only steps 4-6, discarding the
1-3 that were computed only because most techniques can't be asked to
start partway through their own forecast) -- this is what keeps a
commodity's short/medium/long rows from covering the same calendar month
twice with two different techniques' numbers.

Non-blocking throughout, same pattern as every other batch-shaped function
in this pipeline: a commodity/horizon/rank with no saved pickle, or whose
forecast computation fails outright, is logged and skipped -- one bad
combination never aborts the whole consolidated run.

Written to outputs/predictions/consolidated_predictions_{YYYY-MM-DD_HH-MM-SS}.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "data", "features", "models", "scoring"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import PROJECT_ROOT, load_commodity  # noqa: E402
from _common import PERIODS_PER_YEAR, _period_offset  # noqa: E402
from recommend import _load_technique_display_names  # noqa: E402
from model_persistence import SAVED_MODELS_ROOT  # noqa: E402
from predict_from_saved_model import _find_saved_pickle, _load_bundle, _forecast_from_bundle  # noqa: E402

logger = logging.getLogger(__name__)

PREDICTIONS_ROOT = PROJECT_ROOT / "outputs" / "predictions"
HORIZON_DEFAULTS_YAML = PROJECT_ROOT / "config" / "horizon_defaults.yaml"
HORIZON_BUCKETS = ("short", "medium", "long")
RANKS = (1, 2, 3)


def _horizon_ranges_for(frequency: str) -> dict:
    cfg = yaml.safe_load(HORIZON_DEFAULTS_YAML.read_text(encoding="utf-8"))
    return cfg[frequency]


def _discover_commodities_with_saved_models(saved_models_root: Path) -> list[str]:
    if not saved_models_root.is_dir():
        return []
    return sorted(p.name for p in saved_models_root.iterdir() if p.is_dir())


def generate_consolidated_predictions(
    commodity_ids: "list[str] | None" = None,
    horizons: "list[str] | tuple[str, ...]" = HORIZON_BUCKETS,
    ranks: "list[int] | tuple[int, ...]" = RANKS,
    saved_models_root: "Path | None" = None,
    output_root: "Path | None" = None,
) -> Path:
    """
    commodity_ids: None discovers every commodity with at least one saved
    model (scans saved_models/'s own subfolders) rather than every
    commodity in commodities.yaml.
    horizons: which horizon buckets to include -- defaults to all three,
    but the client can ask for just one or two.
    ranks: which saved rank(s) to include -- defaults to all 3 (1, 2, 3);
    narrow to e.g. [1] for just the top pick.
    """
    saved_models_root = saved_models_root if saved_models_root is not None else SAVED_MODELS_ROOT
    if commodity_ids is None:
        commodity_ids = _discover_commodities_with_saved_models(saved_models_root)
        if not commodity_ids:
            raise ValueError(f"generate_consolidated_predictions: no commodity has any saved model under {saved_models_root}")

    unknown_horizons = set(horizons) - set(HORIZON_BUCKETS)
    if unknown_horizons:
        raise ValueError(f"generate_consolidated_predictions: unknown horizon(s) {sorted(unknown_horizons)}, must be from {HORIZON_BUCKETS}")

    unknown_ranks = set(ranks) - set(RANKS)
    if unknown_ranks:
        raise ValueError(f"generate_consolidated_predictions: unknown rank(s) {sorted(unknown_ranks)}, must be from {RANKS}")

    display_names = _load_technique_display_names()
    generated_at = datetime.now()
    rows = []

    for cid in sorted(commodity_ids):
        try:
            commodity = load_commodity(cid)
        except Exception:
            logger.warning("%s: could not load commodity config -- skipping entirely", cid, exc_info=True)
            continue

        periods_per_year = PERIODS_PER_YEAR[commodity.frequency]
        horizon_ranges = _horizon_ranges_for(commodity.frequency)

        for horizon_bucket in horizons:
            m_start, m_end = horizon_ranges[horizon_bucket]

            for rank in ranks:
                try:
                    pickle_path = _find_saved_pickle(cid, horizon_bucket, rank, saved_models_root)
                except FileNotFoundError:
                    logger.info("%s/%s rank %d: no saved model -- skipping", cid, horizon_bucket, rank)
                    continue

                try:
                    bundle = _load_bundle(pickle_path)
                    steps_used, values = _forecast_from_bundle(bundle, commodity, m_end)
                except Exception:
                    logger.warning("%s/%s rank %d (%s): forecast failed -- skipping", cid, horizon_bucket, rank, pickle_path.name, exc_info=True)
                    continue

                trained_through = pd.Timestamp(bundle["trained_through"])
                technique = bundle["technique"]

                for step, value in zip(steps_used, values):
                    if step < m_start:
                        continue  # computed only because most techniques forecast 1..n_end as one block; not this horizon's own range
                    rows.append({
                        "Commodity": cid,
                        "Region": commodity.region,
                        "Horizon": horizon_bucket,
                        "Rank": rank,
                        "Technique": display_names.get(technique, technique),
                        "Composite Score": round(float(bundle["composite_score"]), 4),
                        "Model Pickle File": pickle_path.name,
                        "Trained Through": trained_through.date(),
                        "Forecast Step": step,
                        "Forecast Date": (trained_through + _period_offset(step, periods_per_year)).date(),
                        "Predicted Value": value,
                        "Generated At": generated_at,
                    })

    if not rows:
        raise ValueError("generate_consolidated_predictions: no forecastable (commodity, horizon, rank) combination produced any rows")

    result_df = pd.DataFrame(rows)

    output_root = output_root if output_root is not None else PREDICTIONS_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"consolidated_predictions_{generated_at.strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    result_df.to_excel(out_path, index=False)
    logger.info("wrote %d rows across %d commodity/horizon/rank combinations -> %s", len(result_df), result_df.groupby(["Commodity", "Horizon", "Rank"]).ngroups, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one consolidated forecast workbook across commodities/horizons/ranks.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--commodities", help="Comma-separated commodity ids, e.g. copper,wheat")
    group.add_argument("--all", action="store_true", help="Every commodity that has at least one saved model")
    parser.add_argument("--horizons", default="short,medium,long", help="Comma-separated horizons (default: all three)")
    parser.add_argument("--ranks", default="1,2,3", help="Comma-separated saved ranks to include (default: all 3)")
    args = parser.parse_args()

    commodity_ids = args.commodities.split(",") if args.commodities else None
    horizons = tuple(args.horizons.split(","))
    ranks = tuple(int(r) for r in args.ranks.split(","))

    out_path = generate_consolidated_predictions(commodity_ids, horizons, ranks)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
