"""
Runs ONLY the driver-selection step for one or more commodities -- the
same candidates -> selected pipeline run_batch.py's own _ensure_drivers_
selected() runs automatically as a first step, pulled out standalone here
so it can be run on its own, without also kicking off the full (expensive)
training/backtesting run_batch() always proceeds into right behind it.

Does not modify run_batch.py or its existing behavior at all -- run_batch()
still does exactly what it always did (select-then-train in one call for
any commodity still at drivers_status: candidates). This is a separate,
additive tool for the specific case where a human wants a checkpoint
between "selection just ran" and "training starts": run this, open the
freshly-written config/drivers/selected/{id}.yaml, hand-edit it if needed
(add/discard a driver -- see README Section 7), and only THEN run training
separately (run_batch()/run_horizon()) whenever ready.

Usage (CLI):
  python src/scripts/run_driver_selection.py --commodities acetic_acid,pvc
  python src/scripts/run_driver_selection.py --all-candidates
  python src/scripts/run_driver_selection.py --commodities copper --force

Or programmatically:
  from run_driver_selection import run_driver_selection
  results = run_driver_selection(["acetic_acid", "pvc"])

For each commodity: requires drivers_status == "candidates" (same
requirement select_drivers() itself enforces) unless --force is passed,
which mirrors select_drivers()'s own force=True semantics -- a deliberate,
explicit re-selection for a commodity that's already "selected", e.g.
after changing config/feature_selection.yaml and wanting to see what it
would pick now. Non-blocking across multiple commodities: one commodity's
failure is logged and skipped, never stops the rest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "data", "features"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

from config_loader import load_commodities, load_commodity  # noqa: E402
from feature_selection import promote_to_selected, resolve_candidate_config, select_drivers, write_selected_yaml  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class DriverSelectionOutcome:
    commodity_id: str
    status: str  # "selected" | "skipped" | "error"
    selected_driver_names: "list[str] | None" = None
    selected_yaml_path: "Path | None" = None
    detail: "str | None" = None  # skip reason or error message


def _discover_candidates_commodities() -> list[str]:
    return [c.id for c in load_commodities() if c.drivers_status == "candidates"]


def run_driver_selection(
    commodity_ids: "list[str] | None" = None,
    force: bool = False,
    fast: bool = False,
) -> list[DriverSelectionOutcome]:
    """
    commodity_ids: None discovers every commodity currently at
    drivers_status: candidates (same set run_batch() would auto-select for
    right now). Selection only, nothing beyond it -- no feature
    engineering, no model fitting, no scoring, no output files.
    """
    if commodity_ids is None:
        commodity_ids = _discover_candidates_commodities()
        if not commodity_ids:
            logger.info("run_driver_selection: no commodity is currently at drivers_status: candidates -- nothing to do")
            return []

    outcomes = []
    for cid in commodity_ids:
        try:
            commodity = load_commodity(cid)
        except Exception as exc:
            logger.warning("%s: could not load commodity config -- skipped", cid, exc_info=True)
            outcomes.append(DriverSelectionOutcome(cid, "error", detail=f"{type(exc).__name__}: {exc}"))
            continue

        if not force and commodity.drivers_status != "candidates":
            logger.info("%s: drivers_status is %r, not 'candidates' -- skipped (pass force=True to re-select anyway)", cid, commodity.drivers_status)
            outcomes.append(DriverSelectionOutcome(cid, "skipped", detail=f"drivers_status is {commodity.drivers_status!r}, not 'candidates'"))
            continue

        try:
            effective = resolve_candidate_config(commodity) if force else commodity
            result = select_drivers(effective, force=force, fast=fast)
            path = write_selected_yaml(effective, result)
            promote_to_selected(cid)
            names = [d["label"] for d in result.selected_drivers]
            logger.info("%s: selected %d driver(s) -> %s", cid, len(names), path)
            outcomes.append(DriverSelectionOutcome(cid, "selected", selected_driver_names=names, selected_yaml_path=path))
        except Exception as exc:
            logger.warning("%s: driver selection failed -- commodity stays on its current drivers_status", cid, exc_info=True)
            outcomes.append(DriverSelectionOutcome(cid, "error", detail=f"{type(exc).__name__}: {exc}"))

    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONLY driver selection for one or more commodities -- no training.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--commodities", help="Comma-separated commodity ids, e.g. acetic_acid,pvc")
    group.add_argument("--all-candidates", action="store_true", help="Every commodity currently at drivers_status: candidates")
    parser.add_argument("--force", action="store_true", help="Re-select even for a commodity already at drivers_status: selected")
    parser.add_argument("--fast", action="store_true", help="Throwaway/quick selection (fewer stability refits) -- not for a real result")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    commodity_ids = args.commodities.split(",") if args.commodities else None
    outcomes = run_driver_selection(commodity_ids, force=args.force, fast=args.fast)

    if not outcomes:
        print("Nothing to do -- no commodity currently at drivers_status: candidates.")
        return

    for o in outcomes:
        if o.status == "selected":
            print(f"{o.commodity_id}: SELECTED {len(o.selected_driver_names)} driver(s) -> {o.selected_yaml_path}")
            for name in o.selected_driver_names:
                print(f"    - {name}")
        elif o.status == "skipped":
            print(f"{o.commodity_id}: SKIPPED ({o.detail})")
        else:
            print(f"{o.commodity_id}: ERROR ({o.detail})")


if __name__ == "__main__":
    main()
