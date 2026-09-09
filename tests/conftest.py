"""
Shared pytest fixtures for both test_new_techniques.py and
test_regression_vs_pilot.py. Puts every src/ subpackage on sys.path the
same way every ad hoc test script this project has used all along (flat
module imports, no package __init__.py anywhere under src/ — see any
src/models/*.py's own imports for the same convention) — kept consistent
here rather than switching the whole project to package-relative imports
just for tests.

Fixtures are session-scoped where the underlying data/computation is
read-only and expensive to redo per test (loading + assembling
acetic_acid's features) — every fast=True technique run in this suite
still fits real models, it just does so with a tiny n_windows/n_trials, so
avoiding repeated re-loading matters for keeping the whole suite fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
for _sub in ("", "data", "features", "models", "scoring", "pipeline"):
    sys.path.insert(0, str(_SRC / _sub) if _sub else str(_SRC))

import pytest  # noqa: E402

from config_loader import load_commodities  # noqa: E402
from external_driver_loader import DATE_COLUMN_NAME, load_external_drivers  # noqa: E402
from internal_features import build_internal_features  # noqa: E402
from external_feature_merge import assemble_features  # noqa: E402

PILOT_COMMODITY_ID = "acetic_acid"   # monthly-frequency pilot, used across most fast tests
FAST_N_WINDOWS = 4
FAST_N_TRIALS = 2


@pytest.fixture(scope="session")
def pilot_commodity():
    matches = [c for c in load_commodities() if c.id == PILOT_COMMODITY_ID]
    assert matches, f"{PILOT_COMMODITY_ID} not found in commodities.yaml"
    return matches[0]


@pytest.fixture(scope="session")
def pilot_driver_data(pilot_commodity):
    return load_external_drivers(pilot_commodity)


@pytest.fixture(scope="session")
def pilot_price_series(pilot_driver_data):
    return pilot_driver_data.df.set_index(DATE_COLUMN_NAME)[pilot_driver_data.price_column]


@pytest.fixture(scope="session")
def pilot_internal_features(pilot_commodity, pilot_driver_data):
    return build_internal_features(pilot_commodity, pilot_driver_data.df, pilot_driver_data.price_column)


@pytest.fixture(scope="session")
def pilot_ext_assembly(pilot_commodity, pilot_internal_features, pilot_driver_data):
    return assemble_features(pilot_commodity, pilot_internal_features, pilot_driver_data)


@pytest.fixture(scope="session")
def pilot_ext_var_cols(pilot_ext_assembly):
    return [d.label for d in pilot_ext_assembly.decisions if d.action in ("included", "imputed")]
