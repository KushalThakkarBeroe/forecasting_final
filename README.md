# Commodity Price Forecasting Pipeline

A config-driven pipeline that forecasts short/medium/long-term commodity prices,
tries up to 14 forecasting techniques per commodity per horizon, scores every one
of them with the same formula, and produces a client-facing recommendation.
Adding a new commodity never requires touching code — just data and config files.

This document explains: the folder layout, what each script does, which knobs you
can turn without touching code, how to actually run things, and what to do when a
new commodity or a data refresh comes in.

---

## 1. Folder structure

```
commodity_forecasting_final_v1/
├── README.md                         <- this file
├── requirements.txt                  <- Python packages needed to run this
├── .gitignore
│
├── config/                           <- EVERY tunable knob lives here, not in code
│   ├── commodities.yaml              <- registry of all commodities (auto-generated)
│   ├── horizon_defaults.yaml         <- short/medium/long period ranges, by frequency
│   ├── validation.yaml               <- backtest start date
│   ├── internal_features.yaml        <- lag/rolling-window sizes for feature engineering
│   ├── feature_selection.yaml        <- driver-selection thresholds/method
│   ├── driver_projection.yaml        <- how far ahead + how to project driver values
│   ├── scoring.yaml                  <- scoring weights, thresholds, penalties
│   ├── technique_matrix.yaml         <- which technique is allowed at which horizon
│   └── drivers/
│       ├── candidates/{id}.yaml      <- every driver column found in a commodity's file (auto-generated)
│       └── selected/{id}.yaml        <- the finalized driver set for a commodity
│
├── data/
│   ├── external/{id}/{id}_ext_var.xlsx        <- ONE excel per commodity: price + all drivers
│   └── benchmark_history/{id}/{id}_benchmark.xlsx  <- Beroe's own past forecasts, for comparison
│
├── src/
│   ├── config_loader.py              <- loads & validates every config file above
│   ├── data/
│   │   ├── external_driver_loader.py <- reads a commodity's excel file, computes driver lags
│   │   └── eda.py                    <- exploratory charts + data-quality checks
│   ├── features/
│   │   ├── internal_features.py      <- builds lag/rolling/seasonal features from price alone
│   │   ├── external_feature_merge.py <- merges driver data into the feature table
│   │   └── feature_selection.py      <- narrows candidate drivers down to a final set
│   ├── models/                       <- one file per forecasting technique (Section 3)
│   ├── scoring/                      <- scoring, ranking, decision log, benchmark comparison
│   ├── pipeline/                     <- orchestration: run one commodity, or all of them
│   └── scripts/
│       ├── scaffold_config.py        <- scans data/external/ and updates commodities.yaml
│       └── predict_from_saved_model.py <- forecasts forward from a saved model pickle
│
├── eda/{timestamp}/                  <- diagnostic charts + data-quality flags, one folder per run
├── logs/{timestamp}/                 <- run logs, one folder per run
├── outputs/{commodity_id}/{horizon}/ <- the actual forecast files (Section 5)
│   ├── review_forecast_file.xlsx     <- internal: every technique tried, side by side
│   ├── review_summary_file.xlsx
│   ├── final_forecast_file.xlsx      <- client-facing: the confirmed technique only
│   ├── final_summary_file.xlsx
│   ├── beroe_benchmark.xlsx
│   └── history/                      <- timestamped final_* snapshot per run (Section 5)
├── outputs/consolidated_summary.xlsx        <- cross-commodity rollup (all commodities at once)
├── outputs/technique_selection_history.xlsx <- which technique got confirmed each cycle
├── outputs/decision_log.xlsx                <- audit trail of every analyst confirm/override
├── outputs/predictions/               <- CSVs from predict_from_saved_model.py (Section 5)
├── saved_models/{commodity_id}/{horizon}/   <- top-3 pickled models per commodity x horizon (Section 5)
│
└── tests/                            <- automated tests (Section 6)
```

---

## 2. The pipeline, in order

1. **Ingest** — read a commodity's price + driver data from its one Excel file.
2. **EDA & data-quality check** — charts and sanity checks, never blocks the run.
3. **Driver selection** (only if not already finalized) — narrow candidate drivers
   down to a final set.
4. **Feature engineering** — lags, rolling stats, momentum, seasonality, calendar flags.
5. **Merge drivers into features**, with a data-adequacy gate (exclude/impute
   sparse drivers).
6. **Project each driver's own future value** (needed so external-variable models
   have something to condition on beyond the training window).
7. **Run every eligible technique** for that commodity × horizon.
8. **Robustness Gate** — external-variable models are only usable at Long Term if
   they pass two extra statistical tests.
9. **Score** every technique with one composite formula.
10. **Recommend** the best + runner-up.
11. **Decision Log** — an analyst can confirm or override the recommendation
    (optional, non-blocking).
12. **Write output files** — an internal review copy (every technique) and a
    client-facing final copy (the confirmed technique only).

Steps 1–2 and 3 run once per commodity per batch. Steps 4–12 run once per
commodity **per horizon** (short/medium/long).

---

## 3. What each script does (plain language)

### Core

| File | What it does |
|---|---|
| `src/config_loader.py` | Reads all the yaml config files, checks they're valid, hands back one typed object per commodity that everything else reads from. Nothing downstream ever touches yaml directly. |

### `src/data/`

| File | What it does |
|---|---|
| `external_driver_loader.py` | Reads a commodity's `{id}_ext_var.xlsx`. Column 1 is always date, column 2 is always price, column 3+ are drivers — read by **position**, not by column header text (headers aren't reliable across files). For each driver, it tests a range of lags (e.g. lag 1, 2, 3 months) and picks whichever lag correlates best with price. |
| `eda.py` | For every commodity: trend/distribution/seasonality/volatility charts, stationarity tests, and data-quality checks (missing months, duplicate dates, outliers, negative prices, flat stretches). Writes to `eda/{timestamp}/`. A problem here is logged as a flag, never a crash — one bad commodity never stops the batch. |

### `src/features/`

| File | What it does |
|---|---|
| `internal_features.py` | Builds every feature that only needs the price history itself: lags, rolling mean/std/min/max, momentum, cumulative returns, trend ratios, calendar flags, seasonality index, expanding stats. Every window size is stored in **months** in config and automatically converted to the right number of periods for monthly vs. quarterly commodities. |
| `external_feature_merge.py` | Merges the driver columns from `external_driver_loader.py` into the feature table above. Applies the **data-adequacy gate**: a driver with too little real data is dropped; one with some gaps is imputed; a complete one is used as-is. |
| `feature_selection.py` | For a commodity whose drivers aren't finalized yet: ranks every candidate driver by (a) how much real data it has, (b) how important it is to a Random Forest / LightGBM model, checked for stability across repeated fits, (c) SHAP importance, then applies a 4-test statistical filter (importance, correlation, Granger causality, Johansen cointegration — a driver only needs to pass **one**), followed by a manual override layer and a final redundancy check. Writes the winner(s) to `config/drivers/selected/{id}.yaml`. |

### `src/models/` — one forecasting technique per file

| Technique | File | Notes |
|---|---|---|
| ARIMA | `arima.py` | Univariate, classic. |
| SARIMA | `sarima.py` | ARIMA + seasonality. |
| ETS (Exponential Smoothing) | `ets.py` | Univariate. |
| Random Forest | `rf.py` | Univariate, uses the internal feature set. |
| LightGBM | `lgbm.py` | Same as RF but gradient-boosted, with early stopping. |
| Markov-Switching Regression | `markov_switching.py` | Regime-switching AR model (bull/bear-style regimes). |
| ARIMA + GARCH | `arima_garch.py` | ARIMA for the price forecast, GARCH models the volatility around it (doesn't change the forecast itself, only how confident to be in it). |
| SARIMA + GARCH | `sarima_garch.py` | Same idea, seasonal. |
| RF + External Variables | `rf_ext.py` | Random Forest using the commodity's selected drivers too. |
| LightGBM + External Variables | `lgbm_ext.py` | Same, with LightGBM. |
| ARIMAX | `arimax.py` | ARIMA with drivers as exogenous inputs. |
| SARIMAX | `sarimax.py` | SARIMA with drivers as exogenous inputs. |
| VAR | `var_vecm.py` (`run_var`) | Models price and drivers jointly. Always runs if drivers exist. |
| VECM | `var_vecm.py` (`run_vecm`) | Like VAR, but only runs when price and drivers are statistically confirmed to move together long-term (a cointegration test). If they aren't, no VECM row is produced for that commodity. |
| `_common.py` | *(not a technique)* | The shared backtesting engine every technique above calls into — this is what actually builds the sliding backtest windows, computes accuracy/directional-accuracy metrics, and assembles each technique's output into one consistent shape. Also runs a **multicollinearity check** (`config/multicollinearity.yaml`, same pairwise-correlation method and 0.7 default threshold as the driver-redundancy check) on the 63 internal engineered features before they reach RF/LightGBM/RF+ext/LightGBM+ext — never applied to selected drivers themselves, only to the engineered feature set. |
| `driver_projection.py` | *(not a technique)* | Forecasts each driver's **own** future value (needed so ARIMAX/SARIMAX/etc. have something to condition on beyond the historical data). |

TST and TFT (deep-learning techniques) exist as code but are **disabled** in this
deployment — see Section 7.

### `src/scoring/`

| File | What it does |
|---|---|
| `composite_score.py` | Turns one technique's backtest result into a single 0–100 score, using the weights/thresholds in `config/scoring.yaml`. Also decides if a technique is "disqualified" (hard fail) or merely "not eligible for recommendation" (softer). |
| `robustness_gate.py` | For external-variable models at Long Term only: (1) does removing the driver hurt accuracy enough to justify using it (ablation test)? (2) is the driver's contribution statistically real, not noise (permutation test)? Both must pass. |
| `recommend.py` | Picks the best + runner-up technique from everyone's scores, and builds the two output table shapes (Forecast File, Summary File). |
| `decision_log.py` | Records every time an analyst confirms or overrides the automatic recommendation, with a timestamp and justification. Append-only — nothing is ever edited or deleted, only added to. |
| `selection_history.py` | Records which technique actually got used each forecast cycle, so the next cycle knows what to default to. |
| `model_persistence.py` | After a horizon's scoring finishes, re-fits the top-3 ranked techniques' final models on all available history (reusing the `best_params` already found — no re-searching) and pickles each one to `saved_models/{commodity_id}/{horizon}/`. Ranked by composite score alone — **not** gated by the 85%/55% eligibility threshold that decides the client-facing `final_*` file (see "What the eligibility threshold actually does" below), so a horizon can still save its top-3 models even when nothing was confident enough to auto-recommend. Only excludes a technique that's hard-`disqualified` or had insufficient history to be scored at all. Supports all 14 techniques. Standalone by design: it never touches `_common.py` or any technique's own `run_*()`, so a bug here can only affect what gets saved, never a backtest number or score. See "Where the saved models actually are" in Section 5. |
| `benchmark_accuracy.py` | Generic "how accurate was this forecast" calculator, given a vintage-history table (one column per forecast date, one row per target month). |
| `beroe_actuals_loader.py` | Reads Beroe's own historical price/forecast files from `data/benchmark_history/`. |
| `beroe_benchmark.py` | Scores Beroe's own past forecasts against the real price, using the same accuracy engine as our own techniques — this produces the "Benchmark" row you see next to every real technique. |

### `src/pipeline/` — orchestration

| File | What it does |
|---|---|
| `run_horizon.py` | Runs every step (feature engineering → modeling → scoring → output files) for **one commodity, one horizon**. |
| `run_batch.py` | The actual entry point. Loops every commodity × every horizon. One commodity's failure never stops the rest. Has an optional `parallel=True` mode to run multiple commodity/horizon combinations at once. |
| `generate_outputs.py` | Writes the actual `.xlsx` files (review + final) for one commodity/horizon. Every `final_*` write also drops a never-overwritten timestamped copy into that folder's `history/` subdirectory — see "Comparing runs over time" in Section 5. |
| `generate_consolidated_summary.py` | Builds the cross-commodity rollup (`outputs/consolidated_summary.xlsx`) after a batch finishes. Its Driver Selection sheet's last 5 columns report Item 1's multicollinearity filter — **"Total Internal Features"** (the ~63-column count before filtering) and, side by side, **"Internal Features Kept (Univariate/Ext-Aware) Count/Names"** — see "What's in the Driver Selection sheet" in Section 5. |

### `src/scripts/`

| File | What it does |
|---|---|
| `scaffold_config.py` | Scans `data/external/` for commodity Excel files and updates `config/commodities.yaml` + `config/drivers/candidates/{id}.yaml`. This is the **only** way new commodities get added to the registry — see Section 8. |
| `predict_from_saved_model.py` | Loads one of `model_persistence.py`'s saved rank-1/2/3 pickles and forecasts forward from it — no re-fitting, re-searching, or touching the backtest engine. See "Predicting from a saved model" in Section 5. |

---

## 4. Config knobs you can change without touching code

Everything below lives in `config/*.yaml`. Change the value, save, re-run — no
code edits needed.

| Want to change... | Edit this | Field |
|---|---|---|
| How wide the short/medium/long horizons are | `horizon_defaults.yaml` | `monthly.short/medium/long`, `quarterly.short/medium/long` |
| The date backtesting starts from | `validation.yaml` | `backtest_start_date` |
| Scoring weights (Accuracy/DA/Dynamism/Recency) | `scoring.yaml` | `criterion_weights` |
| Below what accuracy a technique is disqualified | `scoring.yaml` | `disqualification_criteria.mape_accuracy_below_pct` / `directional_accuracy_below_pct` |
| Below what accuracy a technique just isn't "eligible" (softer) | `scoring.yaml` | `eligibility_thresholds` |
| Flat-forecast penalty | `scoring.yaml` | `penalty_flags.flat_line_lt_penalty_points` / `flat_stddev_threshold_pct` |
| Systematic-bias penalty | `scoring.yaml` | `normalized_bias_penalty.bands`, `.penalty_points` |
| HIGH/MEDIUM/LOW confidence cutoffs | `scoring.yaml` | `confidence_tiers` |
| Robustness Gate strictness (Long-Term ext-var models) | `scoring.yaml` | `robustness_gate.ablation_test.min_mape_drop_pp`, `.permutation_test.max_p_value` |
| Which technique is allowed at which horizon | `technique_matrix.yaml` | `eligibility: { short, medium, long }` per technique |
| How a driver is chosen as "used" or "excluded" | `feature_selection.yaml` | `selection_mode`, `max_selected_drivers`, `min_stability_score`, `price_correlation_threshold`, `redundancy_threshold`, `granger_pvalue_threshold` |
| Below what data coverage % a driver is excluded/imputed | `scoring.yaml` | `data_adequacy.min_data_coverage_pct` |
| Feature lag/rolling-window sizes | `internal_features.yaml` | `lag_periods_months`, `roll_window_months`, `long_window_months` |
| How far ahead drivers get projected | `driver_projection.yaml` | `max_horizon_months` |
| Force a specific driver in/out for one commodity | `config/drivers/candidates/{id}.yaml` | `must_include_drivers`, `force_exclude_drivers` |
| Multicollinearity filtering of internal engineered features (RF/LightGBM/RF+ext/LightGBM+ext) | `multicollinearity.yaml` | `enabled`, `correlation_threshold` |

**Do not hand-edit** `config/commodities.yaml`'s auto-generated fields
(`display_name`, `grade`, `region`, `frequency`, `start_period`, `data.file`) — they
get overwritten the next time `scaffold_config.py` runs. `drivers_status` and
`drivers_config` on that same file ARE safe to hand-edit.

---

## 5. How to run it

### One-time setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run everything (all commodities, all horizons)
```bash
python src/pipeline/run_batch.py
```
This produces, per commodity × horizon, the files under `outputs/{id}/{horizon}/`,
plus `outputs/consolidated_summary.xlsx` and `outputs/technique_selection_history.xlsx`.
It also runs EDA and (for any commodity not yet finalized) driver selection first.

The same thing, as a one-liner from the project root (useful if you'd rather not
`cd` into `src/pipeline/` or write a script file) — run this from
`commodity_forecasting_final_v1/` itself, since the `src` paths below are relative
to it:
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/pipeline')
sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/features')
sys.path.insert(0, 'src/models')
sys.path.insert(0, 'src/scoring')

from config_loader import load_commodities
from run_batch import run_batch

commodities = load_commodities()   # every commodity in commodities.yaml
result = run_batch(commodities=commodities)

print('results:', len(result.results), 'errors:', result.errors)
print('consolidated summary:', result.consolidated_summary_path)
"
```
Swap `load_commodities()` for `[c for c in load_commodities() if c.id in ("copper",)]`
(or any set of ids) to scope a run to specific commodities instead of all of them —
useful while testing a config change before running the full batch.

### Run just one commodity, programmatically
```python
from config_loader import load_commodity
from run_horizon import run_horizon

commodity = load_commodity("acetic_acid")
result = run_horizon(commodity, "short")
```

### Run a quick, low-cost smoke test
```python
from run_batch import run_batch
result = run_batch(fast=True)   # scales down window/trial counts, for a fast sanity check
```
`fast=True` output is for checking "did this run without crashing", not for real numbers.

**Redirect every path, not just `output_root`, for a throwaway test run.**
`run_horizon()`/`run_batch()` both take four independent path overrides —
`output_root`, `decision_log_path`, `selection_history_path`, and
`saved_models_root` — each defaulting to its real production location if
not given. A test/verification call that only redirects `output_root` still
writes real pickles to `saved_models/` and a real row into
`technique_selection_history.xlsx`/`decision_log.xlsx`, since those three
have their own independent defaults. Redirect all four together:
```python
result = run_horizon(
    commodity, "short", fast=True,
    output_root=scratch / "outputs",
    decision_log_path=scratch / "outputs" / "decision_log.xlsx",
    selection_history_path=scratch / "outputs" / "technique_selection_history.xlsx",
    saved_models_root=scratch / "saved_models",
)
```
(`run_batch()` takes the same four, plus threads them through its
`parallel=True` path too.) This exists because a `fast=True` verification
run that redirected only `output_root` genuinely leaked 3 real pickles into
`saved_models/copper/short/` and overwrote a real `technique_selection_
history.xlsx` row with fast=True's unreliable (few-trial) ranking — caught
and corrected after the fact, which is what prompted adding
`saved_models_root` as its own parameter.

### Where the client-facing output actually is
`outputs/{commodity_id}/{horizon}/final_forecast_file.xlsx` and
`final_summary_file.xlsx` — these reflect whichever technique the Decision Log
currently holds (auto-defaulted to the top score until an analyst acts).
`review_*` files are the internal, every-technique-shown version.

### What's in the Driver Selection sheet
`outputs/consolidated_summary.xlsx`'s Driver Selection sheet, one row per
commodity, has two distinct halves:
- **External drivers** (`Total/Candidate/Selected Driver...` columns) — the
  candidate driver list and whichever ones `feature_selection.py` actually
  picked, read straight from `config/drivers/candidates/{id}.yaml` and
  `selected/{id}.yaml`.
- **Internal engineered features surviving Item 1's multicollinearity
  filter** (the last 5 columns) — `Total Internal Features` is the count
  before filtering (63 for a monthly commodity, fewer for quarterly, since
  the 12 `is_month` flags are monthly-only); `Internal Features Kept
  (Univariate) Count/Names` is what `rf.py`/`lgbm.py` actually train on;
  `Internal Features Kept (Ext-Aware) Count/Names` is the same filter
  re-run on internal features + selected drivers merged (what
  `rf_ext.py`/`lgbm_ext.py` actually train on) — reported side by side
  rather than reconciled into one answer, because the two genuinely can
  differ (different row set after the driver merge's own dropna, so the
  same pairwise-correlation math can land on a different surviving set). A
  commodity with no selected drivers gets identical values in both, since
  there's nothing for the ext-aware path to differ on. Both are recomputed
  fresh every time this sheet is built (same non-blocking, best-effort
  pattern as everything else in this function) — see
  `_multicollinearity_kept_features` in `generate_consolidated_summary.py`.

### Comparing runs over time
`final_forecast_file.xlsx`/`final_summary_file.xlsx` at their normal path
get overwritten on every rerun — there's no "what did we publish last month"
built into those two files themselves. For that, every `stage="final"` write
(`generate_outputs.py`) also drops a paired, never-overwritten timestamped
copy into
`outputs/{commodity_id}/{horizon}/history/final_forecast_file_{YYYY-MM-DD_HH-MM-SS}.xlsx`
and `final_summary_file_{YYYY-MM-DD_HH-MM-SS}.xlsx` (same timestamp for both
files from the same run, so a forecast/summary pair is easy to match up) —
open two of these side by side to see exactly what changed between runs.

Scoped deliberately narrow, to keep everything else in this pipeline exactly
as it already behaves:
  - Only `final_*` gets a history copy — not `review_*` (internal, always
    reproducible from a rerun) and not `consolidated_summary.xlsx` (a live
    cross-commodity rollup whose own `_merge_with_existing` logic already
    depends on reading one fixed path — see `generate_consolidated_summary.py`
    above; timestamping it would break that).
  - The fixed-name `final_*.xlsx` files are untouched otherwise — the
    Decision Log, "where's the client file" (above), and every other reader
    of that fixed path keep working exactly as before. `history/` is purely
    additive.
  - This is raw snapshots, not a diff tool — nothing computes what changed
    between two runs for you, you're opening two Excel files yourself.

### Where the saved models actually are
`saved_models/{commodity_id}/{horizon}/{commodity_id}_{horizon}_rank{1,2,3}_{technique}_{YYYY-MM-DD_HH-MM-SS}.pkl`
— one pickle per top-3 ranked technique, written automatically at the end of
every `run_horizon()` call (see `model_persistence.py` above). Each pickle is
a dict: `commodity_id`, `horizon_bucket`, `technique`, `rank`, `composite_score`,
`best_params`, `trained_through` (last date the model actually saw),
`saved_at`, and `fitted` — whose own shape depends on the technique: a bare
fitted result object for arima/sarima/ets/arima_garch/sarima_garch/
markov_switching/var/vecm; `{"model":..., "feature_cols":[...]}` for rf/lgbm;
`{step: {"model":..., "feature_cols":[...]}}` for rf_ext/lgbm_ext (one entry
per forecast month); `{"model":..., "exog_cols":[...]}` for arimax/sarimax.
See `model_persistence.py`'s own module docstring for exactly why each shape
differs. Loading and forecasting from one of these is
`predict_from_saved_model.py` — see "Predicting from a saved model" below.

**These 3 saved ranks are NOT the same 3 techniques the eligibility threshold
would call "recommendable."** Model saving is ranked by composite score
alone; it does not check `eligible_for_recommendation` (85% MAPE accuracy /
55% Directional Accuracy — see "What the eligibility threshold actually
does" right below). A horizon where every technique misses that bar (e.g.
copper's Long Term today — best MAPE accuracy was 82.3%, just under 85%)
still gets its top-3 by score saved here, even though `outputs/{id}/{horizon}/final_*`
doesn't get written for that horizon at all (no eligible technique to
recommend). The only techniques never saved are ones `disqualified` (hard
fail) or with too little evaluated history to have a real score.

### What the eligibility threshold actually does
`eligibility_thresholds` in `config/scoring.yaml` (`min_mape_accuracy_pct:
85`, `min_directional_accuracy_pct: 55`) is the bar `recommend()` uses to
decide whether a technique is even in the running for the **client-facing**
output — `outputs/{id}/{horizon}/final_forecast_file.xlsx` /
`final_summary_file.xlsx` and the auto-default written to the Decision Log.
It's a quality floor, deliberately separate from the (relative) Composite
Score / Rank ranking and from Beroe's own Benchmark row — a technique can
out-rank Benchmark and still miss this absolute bar (as VAR does for copper
Long Term above), and it doesn't need to beat Benchmark to pass it either.
Nothing below this bar gets a `final_*` file or becomes a Decision Log
default, no matter how it ranks against the other techniques. It's
independent of, and stricter than, the harder `disqualification_criteria`
gate (MAPE < 75% AND DA < 55%) that forces a technique's composite score to
0 — a technique can be scored, not disqualified, and still fail this softer
eligibility bar (the 75-85% MAPE-accuracy gap band mentioned in Section 8).
Model saving (above) deliberately does NOT use this threshold, so it isn't
blocked by the same quality floor the client-facing file is.

### Predicting from a saved model
```bash
python src/scripts/predict_from_saved_model.py --commodity copper --horizon short --rank 1 --months 3
```
or programmatically:
```python
from predict_from_saved_model import predict_from_saved_model
csv_path = predict_from_saved_model("copper", "short", rank=1, n_months_ahead=3)
```
Loads the chosen rank's pickle and forecasts `n_months_ahead` periods past
that pickle's own `trained_through` date — never re-fitting or re-searching
anything, and never reading real price/driver data past `trained_through`
even if the commodity's own Excel file has since grown (this answers "what
would this saved snapshot forecast", not "blend the saved model with newer
real data" — a re-fit, i.e. re-running `run_horizon()`, is what that needs).
Each of the 14 techniques forecasts forward using the exact same mechanism
the backtest engine already uses at execution time for that technique
(univariate techniques and VAR/VECM forecast directly off the saved fitted
object; RF/LightGBM recurse one step at a time via `_common.py`'s
`_build_future_row`; ARIMAX/SARIMAX project future driver values via
`driver_projection.py`; RF+ext/LightGBM+ext call a different saved model per
requested month, since each forecast distance has its own small model — see
`predict_from_saved_model.py`'s own module docstring for the full per-
technique breakdown). Requesting more months than RF+ext/LightGBM+ext's
horizon actually searched (`best_params_by_horizon`'s own step range) skips
those extra steps with a warning rather than extrapolating past what was
ever tuned.

Writes one CSV row per forecast step to
`outputs/predictions/{commodity_id}_{horizon}_rank{rank}_{technique}_{n}m_{YYYY-MM-DD_HH-MM-SS}.csv`
— columns: `Commodity, Horizon, Rank, Technique, Composite Score, Trained
Through, Step, Date, Predicted`.

**`--horizon` and `--months` are two different, independent knobs.**
`--horizon` picks WHICH saved model to load (short/medium/long are
different models, tuned/backtested over different `config/horizon_
defaults.yaml` ranges — for a monthly commodity: short = months 1-3,
medium = 4-6, long = 7-18). `--months` picks how many periods forward to
actually forecast, once a model's chosen — the script has no way to infer
that on its own. They're meant to be paired sensibly (short model for 1-3
months out, long model for 7-18), but nothing enforces that pairing for
most techniques — ask a short-horizon model for `--months 12` and
ARIMA-family/VAR/RF's own recursion will still produce numbers, just well
past the range that model was ever backtested at, so treat those extra
months with a lot more skepticism than the ones inside its own horizon
range. RF+ext/LightGBM+ext are the one case where this is enforced rather
than left to judgment — see the step-skipping behavior above.

**Predicting multiple commodities at once isn't built in** — the CLI takes
exactly one `--commodity` per call. Loop it yourself:
```bash
for c in copper wheat aluminum; do
  python src/scripts/predict_from_saved_model.py --commodity "$c" --horizon short --rank 1 --months 3
done
```
A commodity with no saved rank-{rank} model yet (nothing in `saved_models/
{commodity}/{horizon}/`) raises `FileNotFoundError` and prints a traceback
for that one commodity — a plain bash `for` loop doesn't stop on that (no
`set -e`), so it just moves on to the next commodity rather than halting
the whole loop. Every commodity needs its own real `run_batch()`/
`run_horizon()` run first (which is what actually writes `saved_models/`)
before this loop can predict from it.

### Run the automated tests
```bash
pytest
```
Two test files: `test_regression_vs_pilot.py` (structural checks on the
techniques with a legacy point of comparison) and `test_new_techniques.py`
(unit tests for Markov-Switching/VAR/VECM/GARCH and the scoring/decision-log
modules, which have no legacy baseline to compare against).

---

## 6. Onboarding a brand-new commodity

1. **Drop the data file in**: `data/external/{commodity_id}/{commodity_id}_ext_var.xlsx`
   — one Excel file, column 1 = date, column 2 = this commodity's price, column 3
   onward = its candidate drivers. (If you have Beroe benchmark history for it too,
   drop that at `data/benchmark_history/{commodity_id}/{commodity_id}_benchmark.xlsx`
   — optional, only used for the "Benchmark" comparison row.)

2. **Run the scaffolder**:
   ```bash
   python src/scripts/scaffold_config.py
   ```
   This scans `data/external/`, adds a new entry to `config/commodities.yaml` for
   the commodity (frequency, start date, display name, region/grade — inferred
   best-effort from the file), and writes `config/drivers/candidates/{id}.yaml`
   listing every driver column it found. New commodities always start at
   `drivers_status: candidates`.

3. **Run the batch** (or just this commodity):
   ```bash
   python src/pipeline/run_batch.py
   ```
   Because the commodity is still `candidates`, this automatically runs
   `feature_selection.py` for it first — narrowing the driver list down to a
   final set, written to `config/drivers/selected/{id}.yaml`, and flipping
   `drivers_status` to `selected` in `commodities.yaml`. From that point on the
   commodity is finalized and won't be re-selected automatically.

4. **Check the output** in `outputs/{commodity_id}/{short,medium,long}/` — the
   `review_*` files show every technique that was tried; pick your recommended
   technique via the Decision Log (Section 7) if you want to override the
   auto-picked top score.

If a driver is genuinely known-bad (e.g., a redundant near-duplicate) before
you've even run selection, you can hand-edit
`config/drivers/candidates/{id}.yaml` to drop that column, or use
`must_include_drivers`/`force_exclude_drivers` to force a decision regardless
of what the statistics say.

---

## 7. Retraining / re-running an existing commodity

- **Just want fresh numbers with the same driver set?** Re-run
  `run_batch.py` (or `run_horizon.py` for just that commodity). Driver
  selection is **not** re-run for a commodity already at `drivers_status: selected`
  — it's static once finalized, by design.

- **A commodity's drivers changed** (a new candidate driver appeared, an old one
  should be dropped, or you just want to re-check the selection)?
  - Easiest: in `config/commodities.yaml`, change **both** of that commodity's
    fields together — they must always agree, never one `selected` and the
    other `candidates`:
    1. `drivers_status: candidates`
    2. `drivers_config: config/drivers/candidates/{id}.yaml` (that same
       commodity's own id, e.g. `config/drivers/candidates/copper.yaml` for
       copper — not another commodity's file)

    Then re-run the batch — it will re-select automatically. Changing only
    one of the two fields leaves them pointing at mismatched folders
    (e.g. `drivers_status: candidates` but `drivers_config` still pointing at
    `selected/{id}.yaml`) and crashes driver selection with a `KeyError`.
  - Or: call `run_batch(force_reselect=True)` to force **every** commodity in
    that batch call through selection again (e.g. after changing
    `feature_selection.yaml`'s thresholds and wanting it to actually take
    effect everywhere) — this doesn't require hand-editing `commodities.yaml`
    first.

- **The underlying price/driver Excel file was refreshed** (new months of
  data)? No special step — just re-run. The pipeline always reads the latest
  file and grows its backtest window forward automatically
  (`config/validation.yaml`'s `backtest_start_date` is a fixed anchor, not a
  rolling cap — every new month of data adds one more backtest window).

- **Want to confirm or override which technique gets used**, instead of
  letting it auto-default to the top score?
  ```python
  from decision_log import confirm_recommendation, override_recommendation

  confirm_recommendation("acetic_acid", "short", recommendation, analyst="Jane Doe")
  # or, to pick something else:
  override_recommendation("acetic_acid", "short", "sarima", analyst="Jane Doe",
                           justification="ARIMA's forecast looks unrealistic given the recent supply shock.")
  ```
  This writes to `outputs/decision_log.xlsx` (append-only, full history kept)
  and the next `final_*` output file will reflect it.

---

## 8. Current known limitations (worth knowing before relying on this)

- **TST, TFT, and LSTM are disabled.** TST/TFT exist as working code but are
  moved out of `src/models/` (they need `torch`, which isn't part of this
  environment's dependency set) — move the files back to re-enable wherever
  torch is available. LSTM was never built.
- **Quarterly horizon ranges in `horizon_defaults.yaml` are placeholders** —
  don't treat a quarterly commodity's Long-Term forecast as final until
  those are confirmed.
- **No analyst decision has been recorded yet** for most commodities —
  `outputs/decision_log.xlsx` only has rows once someone calls
  `confirm_recommendation`/`override_recommendation`; until then, every
  output reflects the auto-picked top score.
- **`config/scoring.yaml` flags two of its own open items inline**: a gap
  band between the "not eligible" (85% MAPE accuracy) and "hard
  disqualified" (75%) thresholds, and two values (Forecast Dynamism's
  reference %, Recency Bonus's multiplier) that are reasonable defaults, not
  confirmed final values.
- **A saved ARIMAX/SARIMAX pickle from before `model_persistence.py`'s
  `exog_cols` change won't load correctly in `predict_from_saved_model.py`.**
  `_fit_arimax`/`_fit_sarimax` used to save a bare fitted model; they now
  save `{"model":..., "exog_cols":[...]}` instead, since SARIMAX is fit on
  a plain array and has no memory of which driver each exog column was or
  what order they were in. Any ARIMAX/SARIMAX `.pkl` saved before this
  change is in the old shape — re-run that commodity/horizon to get a
  fresh one before predicting from it. Every other technique's saved shape
  is unchanged.
