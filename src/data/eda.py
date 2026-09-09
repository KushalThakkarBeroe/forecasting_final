"""
EDA and data-quality flagging: statistical summary, trend, distribution,
seasonality (boxplot + heatmap), volatility, stationarity, STL
decomposition, ACF/PACF — operating on the single price series from
external_driver_loader.py's output, period-aware (STL period = 12 for
monthly, 4 for quarterly), writing to a dedicated, timestamped eda/ folder.

Also runs data-quality checks: non-numeric cells coerced to NaN, missing
calendar periods, duplicate dates, price outliers (z-score > 3), negative
prices, and long flat/constant stretches.

Non-blocking by design: run_eda_for_commodity wraps every individual
analysis in its own try/except (one broken chart type doesn't stop the rest
of that commodity's EDA) and run_eda_batch wraps each commodity in its own
try/except (one commodity's total failure doesn't stop run_batch.py). Every
problem becomes a row in data_quality_flags.xlsx, never a raised exception.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — batch/script use, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config_loader import CommodityConfig
from external_driver_loader import DATE_COLUMN_NAME, load_external_drivers

logger = logging.getLogger(__name__)


@dataclass
class DataQualityFlag:
    commodity_id: str
    check: str
    severity: str  # "info" | "warning" | "error"
    detail: str


def _period_for_frequency(frequency: str) -> int:
    return {"monthly": 12, "quarterly": 4}.get(frequency, 12)


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def basic_statistical_summary(s: pd.Series) -> dict:
    clean = s.dropna()
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "skewness": float(stats.skew(clean)),
        "kurtosis": float(stats.kurtosis(clean)),
        **{f"p{p}": float(clean.quantile(p / 100)) for p in [10, 25, 50, 75, 90]},
    }


# ---------------------------------------------------------------------------
# Plots — saved to output_dir, never shown (matplotlib backend is
# non-interactive — see module docstring).
# ---------------------------------------------------------------------------

def plot_price_trend(s: pd.Series, commodity_id: str, output_dir: Path, window: int = 12) -> Path:
    rolling = s.rolling(window)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(s.index, s, label="Actual Price", linewidth=1.5)
    ax.plot(s.index, rolling.mean(), label=f"{window}-period Rolling Mean", linestyle="--")
    ax.fill_between(
        s.index, rolling.mean() - rolling.std(), rolling.mean() + rolling.std(),
        alpha=0.2, label="±1 Std Dev Band",
    )
    ax.set_title(f"Price Trend — {commodity_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"trend_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distribution(s: pd.Series, commodity_id: str, output_dir: Path) -> Path:
    clean = s.dropna()
    skew = stats.skew(clean)
    kurt = stats.kurtosis(clean)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(clean, kde=True, ax=ax, color="steelblue", stat="density", label="Actual Distribution")
    x = np.linspace(clean.min(), clean.max(), 200)
    ax.plot(x, stats.norm.pdf(x, clean.mean(), clean.std()), "r--", linewidth=1.5, label="Normal Reference")
    ax.axvline(clean.mean(), color="orange", linestyle="--", linewidth=1, label=f"Mean: {clean.mean():.1f}")
    ax.axvline(clean.median(), color="green", linestyle="--", linewidth=1, label=f"Median: {clean.median():.1f}")
    ax.set_title(f"Price Distribution — {commodity_id}\nSkewness: {skew:.3f} | Excess Kurtosis: {kurt:.3f}")
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"distribution_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def seasonality_boxplot(s: pd.Series, commodity_id: str, output_dir: Path) -> Path:
    temp = s.reset_index()
    temp.columns = ["Date", "Price"]
    temp["Month"] = temp["Date"].dt.month
    temp["MonthName"] = temp["Date"].dt.strftime("%b")
    month_order = temp.groupby("Month")["MonthName"].first().sort_index().values
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=temp, x="MonthName", y="Price", order=month_order, ax=ax)
    ax.set_title(f"Seasonality — {commodity_id}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price")
    fig.tight_layout()
    path = output_dir / f"monthly_seasonality_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def seasonal_heatmap(s: pd.Series, commodity_id: str, output_dir: Path) -> Path:
    temp = s.reset_index()
    temp.columns = ["Date", "Price"]
    temp["Year"] = temp["Date"].dt.year
    temp["Month"] = temp["Date"].dt.month
    temp["MonthName"] = temp["Date"].dt.strftime("%b")
    pivot = temp.pivot_table(index="Year", columns="MonthName", values="Price")
    month_order = [m for m in temp.groupby("Month")["MonthName"].first().sort_index().values if m in pivot.columns]
    pivot = pivot[month_order]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", linewidths=0.5, ax=ax)
    ax.set_title(f"Seasonal Heatmap (Price by Month × Year) — {commodity_id}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    path = output_dir / f"seasonal_heatmap_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def volatility_analysis(s: pd.Series, commodity_id: str, output_dir: Path, window: int = 6) -> tuple[Path, dict]:
    returns = s.pct_change().dropna() * 100
    rolling_vol = returns.rolling(window).std()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(returns.index, returns, color="steelblue", linewidth=1)
    axes[0].axhline(0, linestyle="--", color="black", linewidth=0.8)
    axes[0].set_title(f"Period-on-Period Returns (%) — {commodity_id}")
    axes[0].set_ylabel("Return (%)")
    axes[1].plot(rolling_vol.index, rolling_vol, color="tomato", linewidth=1.5)
    axes[1].set_title(f"{window}-period Rolling Volatility — {commodity_id}")
    axes[1].set_ylabel("Std Dev of Returns (%)")
    fig.tight_layout()
    path = output_dir / f"volatility_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    summary = {
        "mean_return_pct": float(returns.mean()),
        "std_return_pct": float(returns.std()),
        "max_spike_pct": float(returns.max()),
        "max_spike_date": returns.idxmax().strftime("%Y-%m-%d"),
        "max_drop_pct": float(returns.min()),
        "max_drop_date": returns.idxmin().strftime("%Y-%m-%d"),
    }
    return path, summary


def stationarity_tests(s: pd.Series) -> dict:
    results = {}
    for label, series in [("level", s.dropna()), ("first_difference", s.diff().dropna())]:
        adf_stat, adf_p = adfuller(series)[:2]
        kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]
        results[label] = {
            "adf_stat": float(adf_stat), "adf_p": float(adf_p), "adf_stationary": bool(adf_p < 0.05),
            "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p), "kpss_stationary": bool(kpss_p > 0.05),
        }
    return results


def stl_decomposition(s: pd.Series, commodity_id: str, output_dir: Path, period: int) -> tuple[Path, dict]:
    result = STL(s.dropna(), period=period, robust=True).fit()
    fig = result.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle(f"STL Decomposition — {commodity_id}", y=1.01)
    fig.tight_layout()
    path = output_dir / f"stl_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    var_resid = result.resid.var()
    trend_strength = max(0.0, 1 - var_resid / (result.trend + result.resid).var())
    season_strength = max(0.0, 1 - var_resid / (result.seasonal + result.resid).var())
    return path, {"trend_strength": float(trend_strength), "seasonal_strength": float(season_strength)}


def acf_pacf_analysis(s: pd.Series, commodity_id: str, output_dir: Path, lags: int = 24) -> Path:
    s_clean = s.dropna()
    s_diff = s_clean.diff().dropna()
    safe_lags = max(1, min(lags, len(s_clean) // 2 - 1, len(s_diff) // 2 - 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(s_clean, lags=safe_lags, ax=axes[0, 0], title=f"ACF — Level ({commodity_id})")
    plot_pacf(s_clean, lags=safe_lags, ax=axes[0, 1], title=f"PACF — Level ({commodity_id})")
    plot_acf(s_diff, lags=safe_lags, ax=axes[1, 0], title=f"ACF — 1st Diff ({commodity_id})")
    plot_pacf(s_diff, lags=safe_lags, ax=axes[1, 1], title=f"PACF — 1st Diff ({commodity_id})")
    fig.tight_layout()
    path = output_dir / f"acf_pacf_{commodity_id}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Data-quality checks
# ---------------------------------------------------------------------------

def check_data_quality(commodity: CommodityConfig, price: pd.Series) -> list[DataQualityFlag]:
    flags: list[DataQualityFlag] = []
    cid = commodity.id

    # Non-numeric / uncoercible values already became NaN in price during
    # loading — flag if any are present outright (unexpected for the price
    # column specifically, unlike a driver, which the data-adequacy gate at
    # Step 5 handles separately).
    n_missing = int(price.isna().sum())
    if n_missing:
        flags.append(DataQualityFlag(cid, "missing_or_non_numeric_price", "warning",
                                      f"{n_missing} of {len(price)} price values are missing or non-numeric"))

    # Missing calendar periods (gaps in the expected sequence)
    freq_alias = {"monthly": "MS", "quarterly": "QS"}.get(commodity.frequency)
    if freq_alias and price.index.notna().all():
        expected = pd.date_range(price.index.min(), price.index.max(), freq=freq_alias)
        missing_periods = expected.difference(price.index)
        if len(missing_periods):
            flags.append(DataQualityFlag(cid, "missing_calendar_periods", "error",
                                          f"{len(missing_periods)} missing periods: "
                                          f"{', '.join(d.strftime('%Y-%m') for d in missing_periods[:10])}"
                                          + (" ..." if len(missing_periods) > 10 else "")))

    # Duplicate dates
    n_dupes = int(price.index.duplicated().sum())
    if n_dupes:
        flags.append(DataQualityFlag(cid, "duplicate_dates", "error", f"{n_dupes} duplicate date(s) in the price index"))

    # Outliers (z-score > 3)
    clean = price.dropna()
    if clean.std() > 0:
        z = (clean - clean.mean()) / clean.std()
        outliers = clean[z.abs() > 3]
        if len(outliers):
            flags.append(DataQualityFlag(cid, "price_outliers", "warning",
                                          f"{len(outliers)} value(s) with |z-score| > 3: "
                                          + ", ".join(f"{d.strftime('%Y-%m')}={v:.2f}" for d, v in outliers.items())))

    # Negative prices
    n_negative = int((clean < 0).sum())
    if n_negative:
        flags.append(DataQualityFlag(cid, "negative_price", "warning", f"{n_negative} negative price value(s)"))

    # Flat/constant stretches (>= 4 consecutive identical non-null values)
    run_length = 1
    max_run = 1
    max_run_end = None
    prev = None
    for date, val in clean.items():
        if prev is not None and val == prev:
            run_length += 1
        else:
            run_length = 1
        if run_length > max_run:
            max_run = run_length
            max_run_end = date
        prev = val
    if max_run >= 4:
        flags.append(DataQualityFlag(cid, "flat_stretch", "info",
                                      f"{max_run} consecutive identical price values ending {max_run_end.strftime('%Y-%m')}"))

    return flags


# ---------------------------------------------------------------------------
# Orchestration — non-blocking per commodity and per analysis
# ---------------------------------------------------------------------------

def run_eda_for_commodity(commodity: CommodityConfig, run_dir: Path) -> list[DataQualityFlag]:
    """Runs every EDA analysis + data-quality check for one commodity. Every
    individual step is wrapped so a single broken chart/stat becomes a flag,
    not a crash — this function itself never raises."""
    cid = commodity.id
    commodity_dir = run_dir / cid
    commodity_dir.mkdir(parents=True, exist_ok=True)
    flags: list[DataQualityFlag] = []

    def safe(label: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("%s: %s failed — %s", cid, label, exc)
            flags.append(DataQualityFlag(cid, f"eda_step_failed:{label}", "error",
                                          f"{type(exc).__name__}: {exc}"))
            return None

    driver_data = safe("load_external_drivers", load_external_drivers, commodity)
    if driver_data is None:
        return flags  # nothing else can run without the price series

    price = driver_data.df.set_index(DATE_COLUMN_NAME)[driver_data.price_column]
    period = _period_for_frequency(commodity.frequency)

    safe("basic_statistical_summary", basic_statistical_summary, price)
    safe("plot_price_trend", plot_price_trend, price, cid, commodity_dir)
    safe("plot_distribution", plot_distribution, price, cid, commodity_dir)
    safe("seasonality_boxplot", seasonality_boxplot, price, cid, commodity_dir)
    safe("seasonal_heatmap", seasonal_heatmap, price, cid, commodity_dir)
    safe("volatility_analysis", volatility_analysis, price, cid, commodity_dir)
    safe("stationarity_tests", stationarity_tests, price)
    safe("stl_decomposition", stl_decomposition, price, cid, commodity_dir, period)
    safe("acf_pacf_analysis", acf_pacf_analysis, price, cid, commodity_dir)

    quality_flags = safe("check_data_quality", check_data_quality, commodity, price)
    if quality_flags:
        flags.extend(quality_flags)

    return flags


def run_eda_batch(commodities: list[CommodityConfig], eda_root: Path, run_timestamp: str | None = None) -> Path:
    """Runs EDA for every commodity, writes one data_quality_flags.xlsx for
    the whole run. A commodity that fails entirely (should be unreachable,
    since run_eda_for_commodity already catches everything internally, but
    guarded again here as a last line of defense) is flagged and skipped —
    never stops the batch."""
    run_timestamp = run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = eda_root / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_flags: list[DataQualityFlag] = []
    for commodity in commodities:
        try:
            all_flags.extend(run_eda_for_commodity(commodity, run_dir))
        except Exception:
            logger.error("%s: EDA failed entirely, skipping", commodity.id)
            all_flags.append(
                DataQualityFlag(commodity.id, "eda_commodity_failed", "error", traceback.format_exc(limit=3))
            )

    flags_path = run_dir / "data_quality_flags.xlsx"
    flags_df = pd.DataFrame([vars(f) for f in all_flags]) if all_flags else pd.DataFrame(
        columns=["commodity_id", "check", "severity", "detail"]
    )
    flags_df.to_excel(flags_path, index=False)
    logger.info("EDA batch complete: %d flag(s) across %d commodities -> %s", len(all_flags), len(commodities), flags_path)
    return flags_path
