"""
Loads and validates config/commodities.yaml plus each commodity's driver
config (config/drivers/selected|candidates/{id}.yaml) and horizon periods
(config/horizon_defaults.yaml, keyed by frequency). Returns one
CommodityConfig per commodity — the typed object every later pipeline step
(features, models, scoring) reads from, so nothing downstream touches yaml
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
COMMODITIES_YAML = CONFIG_DIR / "commodities.yaml"
HORIZON_DEFAULTS_YAML = CONFIG_DIR / "horizon_defaults.yaml"
VALIDATION_YAML = CONFIG_DIR / "validation.yaml"

VALID_FREQUENCIES = {"monthly", "quarterly"}


class ConfigError(Exception):
    """Raised when commodities.yaml or a per-commodity driver config fails validation."""


@dataclass(frozen=True)
class HorizonPeriods:
    short: tuple[int, int]
    medium: tuple[int, int]
    long: tuple[int, int]


@dataclass(frozen=True)
class CommodityConfig:
    id: str
    display_name: str
    grade: str | None
    region: str | None
    frequency: str
    start_period: str
    data_file: Path
    drivers_status: str  # "selected" | "candidates"
    drivers_config_path: Path
    drivers_config: dict = field(repr=False)
    horizon_periods: HorizonPeriods


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_horizon_defaults(horizon_defaults_path: Path = HORIZON_DEFAULTS_YAML) -> dict[str, HorizonPeriods]:
    raw = _load_yaml(horizon_defaults_path)
    result: dict[str, HorizonPeriods] = {}
    for freq, ranges in raw.items():
        try:
            result[freq] = HorizonPeriods(
                short=tuple(ranges["short"]),
                medium=tuple(ranges["medium"]),
                long=tuple(ranges["long"]),
            )
        except KeyError as exc:
            raise ConfigError(
                f"{horizon_defaults_path}: frequency '{freq}' is missing a short/medium/long range"
            ) from exc
    return result


def load_backtest_start_date(validation_yaml: Path = VALIDATION_YAML) -> pd.Timestamp:
    """The fixed anchor date the internal backtest window grows forward
    from (config/validation.yaml) -- replaces the old fixed-count,
    always-most-recent-N-periods sliding window."""
    raw = _load_yaml(validation_yaml)
    if "backtest_start_date" not in raw:
        raise ConfigError(f"{validation_yaml} is missing required field 'backtest_start_date'")
    return pd.Timestamp(raw["backtest_start_date"])


def _validate_commodity_entry(entry: dict, project_root: Path) -> None:
    commodity_id = entry.get("id", "<missing id>")
    required = ["id", "display_name", "frequency", "start_period", "data", "drivers_status", "drivers_config"]
    for f in required:
        if f not in entry:
            raise ConfigError(f"commodities.yaml entry '{commodity_id}' is missing required field '{f}'")

    if entry["frequency"] not in VALID_FREQUENCIES:
        raise ConfigError(
            f"commodities.yaml entry '{commodity_id}' has frequency='{entry['frequency']}', "
            f"must be one of {sorted(VALID_FREQUENCIES)}"
        )

    if entry["drivers_status"] not in {"selected", "candidates"}:
        raise ConfigError(
            f"commodities.yaml entry '{commodity_id}' has drivers_status='{entry['drivers_status']}', "
            f"must be 'selected' or 'candidates'"
        )

    data_file = project_root / entry["data"]["file"]
    if not data_file.is_file():
        raise ConfigError(f"commodities.yaml entry '{commodity_id}': data.file not found at {data_file}")

    drivers_config_path = project_root / entry["drivers_config"]
    if not drivers_config_path.is_file():
        raise ConfigError(
            f"commodities.yaml entry '{commodity_id}': drivers_config not found at {drivers_config_path}"
        )


def load_commodities(
    commodities_yaml: Path = COMMODITIES_YAML,
    horizon_defaults_yaml: Path = HORIZON_DEFAULTS_YAML,
    project_root: Path = PROJECT_ROOT,
) -> list[CommodityConfig]:
    """Load, validate, and return one CommodityConfig per commodity in
    commodities.yaml. Raises ConfigError on the first invalid entry."""
    doc = _load_yaml(commodities_yaml)
    entries = doc.get("commodities", [])
    if not entries:
        raise ConfigError(f"{commodities_yaml} has no commodities listed")

    horizon_defaults = load_horizon_defaults(horizon_defaults_yaml)

    configs: list[CommodityConfig] = []
    for entry in entries:
        _validate_commodity_entry(entry, project_root)

        frequency = entry["frequency"]
        if frequency not in horizon_defaults:
            raise ConfigError(
                f"commodities.yaml entry '{entry['id']}': no horizon_defaults entry for "
                f"frequency='{frequency}' in {horizon_defaults_yaml}"
            )

        drivers_config_path = project_root / entry["drivers_config"]
        drivers_config = _load_yaml(drivers_config_path)

        configs.append(
            CommodityConfig(
                id=entry["id"],
                display_name=entry["display_name"],
                grade=entry.get("grade"),
                region=entry.get("region"),
                frequency=frequency,
                start_period=entry["start_period"],
                data_file=project_root / entry["data"]["file"],
                drivers_status=entry["drivers_status"],
                drivers_config_path=drivers_config_path,
                drivers_config=drivers_config,
                horizon_periods=horizon_defaults[frequency],
            )
        )
    return configs


def load_commodity(commodity_id: str, **kwargs) -> CommodityConfig:
    """Convenience wrapper: load_commodities(...) filtered to one id."""
    for cfg in load_commodities(**kwargs):
        if cfg.id == commodity_id:
            return cfg
    raise ConfigError(f"No commodity '{commodity_id}' found in commodities.yaml")
