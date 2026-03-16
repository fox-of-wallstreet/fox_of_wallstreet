# Fox of Wallstreet - Project Documentation

## Quick Start

1. Install dependencies:
  python -m pip install -r requirements.txt
2. Set Alpaca credentials for news ingestion:
  export ALPACA_API_KEY=your_key
  export ALPACA_SECRET_KEY=your_secret
3. Configure run settings in config/settings.py (symbol, timeframe, dates, flags).
4. Build raw checkpoints:
  python scripts/data_engine.py
  python scripts/news_engine.py
  python scripts/macro_engine.py
5. Train model artifacts:
  python scripts/train.py
6. Run deterministic backtest:
  python scripts/backtest.py

Outputs are written to data/raw, data/intermediate, and artifacts/<EXPERIMENT_NAME>.
Run index is written to artifacts/experiment_journal.csv.
Backtest also generates a report bundle under artifacts/<RESOLVED_RUN_ID>/reports/.

## Architecture Overview

The system is a fully modular PPO reinforcement learning trading pipeline. Every component is an executor, while `settings.py` is the only file that makes decisions.

Data flows in one direction:

```text
settings.py (config)
      |
      v
data_engine.py --------------------> data/raw/tsla_1h_prices.csv
news_engine.py --------------------> data/raw/tsla_news.csv
macro_engine.py -------------------> data/raw/tsla_1h_macro.csv
      |
      v
processor.py
  |- load_raw_prices()
  |- load_raw_news()
  |- load_raw_macro()
  |- build_news_sentiment()       -> data/intermediate/tsla_1h_news_sentiment.csv
  |- merge_prices_news_macro()    -> data/intermediate/tsla_1h_merged.csv
  |- add_technical_indicators()   -> registry-driven, reads FEATURES_LIST
  |- build_training_dataset()     -> data/intermediate/tsla_1h_train_features.csv
  '- prepare_features()           -> RobustScaler (fit on train, reused elsewhere)
      |
      v
environment.py (TradingEnv)       <- scaled features + raw df
      |
      v
optimize.py ----------------------> artifacts/optuna_study.db
      |
      v
train.py
  |- _resolve_ppo_params()        <- settings defaults OR Optuna best
  '- run_training()               -> artifacts/<EXPERIMENT_NAME>/
                                       |- model.zip
                                       |- scaler.pkl
                                       |- metadata.json
                                       '- backtest_ledger.csv (from backtest.py)
```

## File-by-File Changes

### config/settings.py

What changed:

- Feature flags (`USE_NEWS_FEATURES`, `USE_MACRO_FEATURES`, `USE_TIME_FEATURES`) now programmatically build `FEATURES_LIST`.
- `EXPECTED_MARKET_FEATURES = len(FEATURES_LIST)` keeps environment shape checks in sync.
- Indicator params are centralized in settings:
  - `RSI_WINDOW`
  - `MACD_FAST`
  - `MACD_SLOW`
  - `MACD_SIGNAL`
  - `VOLATILITY_WINDOW`
  - `NEWS_EMA_SPAN`
- `EXPERIMENT_NAME` now uses a hybrid naming scheme: readable config tags + timestamp.
- `from datetime import datetime` added.

Example experiment name:

```text
ppo_TSLA_1h_discrete_3_news_macro_time_20260312_1445
```

### core/processor.py

What changed:

- `DEFAULT_FEATURES` removed to avoid duplication with `settings.FEATURES_LIST`.
- `_get_setting()` removed; params now come directly from settings.
- `add_technical_indicators()` is registry-driven and computes only requested features.
- Pre-flight validation raises an immediate error if any requested feature is not registered.
- Each feature has an isolated private compute function (`_compute_rsi`, `_compute_macd_hist`, etc.).
- `prepare_features()` defaults to `settings.FEATURES_LIST`.

Registry pattern:

```python
FEATURE_REGISTRY = {
    "RSI": _compute_rsi,
    "MACD_Hist": _compute_macd_hist,
    # ... one entry per feature
}
```

### core/environment.py

What changed:

- `EXPECTED_MARKET_FEATURES` shape check is now fully wired to settings.
- Hardcoded slippage now uses `settings.SLIPPAGE_PCT`.
- Hardcoded invalid action penalty now uses `settings.INVALID_ACTION_PENALTY`.
- `_check_sl_tp()` added so `STOP_LOSS_PCT` and `TAKE_PROFIT_PCT` trigger forced closes before agent action.
- `sl_tp_triggered` added to `info` for separate logging in evaluation.
- Zero-division guard added to portfolio return denominator (`+ 1e-8`).
- Observation shape is dynamic:
  - `self.num_features = self.features.shape[1] + NUM_PORTFOLIO_FEATURES`
- Portfolio state observation updated to 5 features: `cash_ratio`, `position_size`, `inventory_fraction`, `unrealized_pnl`, `last_action`.
- `last_action` is normalized by max action index so it stays in `[0, 1]`.

### scripts/train.py

What changed:

- Legacy `data/{symbol}_hybrid.csv` load removed in favor of `build_training_dataset()`.
- Signature-aware checkpoint cache added for train features:
  - Reuses `TRAIN_FEATURES_CSV` only when settings + raw input mtimes match.
  - Rebuilds automatically when config/data changed.
- Hardcoded feature list removed; `prepare_features()` uses `settings.FEATURES_LIST`.
- PPO params now come from settings (`LEARNING_RATE`, `ENT_COEF`, `N_STACK`).
- `seed=settings.RANDOM_SEED` added for reproducibility.
- `_resolve_ppo_params()` added to consume Optuna best params when enabled.
- Metadata expanded with feature flags, feature count, and PPO settings.
- Auto-logs training runs into `artifacts/experiment_journal.csv`.

### scripts/backtest.py

What changed:

- Resolves latest compatible artifact run when current timestamped run has no model/scaler yet.
- Validates runtime settings against resolved training metadata before execution.
- Signature-aware checkpoint cache added for test features.
- Auto-logs backtest metrics into `artifacts/experiment_journal.csv`.
- Uses the scaler from the resolved training run (same run as loaded model).
- Stores richer ledger rows with position transitions (`Position_Before`, `Position_After`).
- Writes compact summary JSON (`backtest_summary.json`) for each resolved run.
- Generates a report bundle under `artifacts/<RUN_ID>/reports/`:
  - `figures/actions_overlay.png`
  - `figures/equity_vs_benchmark.png`
  - `figures/drawdown_curve.png`
  - `figures/trade_return_hist.png` (when cycle returns exist)
  - `tables/equity_timeseries.csv`
  - `summary/report_index.json`

### scripts/macro_engine.py

What changed:

- New dedicated macro ingestion script for QQQ/ARKK/VIX/TNX closes.
- Writes raw macro checkpoint consumed by processor merge flow.

### scripts/artifact_manager.py

What changed:

- New optional maintenance utility for artifact hygiene.
- Supports listing runs, pruning empty runs, and keep-latest retention.

### scripts/optimize.py

What changed:

- Global import-time data loading removed; now loaded inside `_load_train_data()`.
- Objective factory pattern (`build_objective`) loads data once and reuses it across trials.
- Uses an optimization-local in-memory `RobustScaler` fit on current train features.
- No dependency on a pre-existing scaler file from a prior training run.
- Hardcoded values replaced by settings:
  - `OPTUNA_TRIALS`
  - `OPTUNA_EVAL_TIMESTEPS`
  - `OPTUNA_DB_PATH`
  - `OPTUNA_STUDY_NAME`
  - `N_STACK`
  - `RANDOM_SEED`
- Uses resumable SQLite storage with `load_if_exists=True`.

## All Files Produced

| File | Produced by | Purpose |
|---|---|---|
| `data/raw/tsla_1h_prices.csv` | `data_engine.py` | Raw OHLCV checkpoint |
| `data/raw/tsla_news.csv` | `news_engine.py` | Raw Alpaca news checkpoint |
| `data/raw/tsla_1h_macro.csv` | `macro_engine.py` | Raw macro checkpoint (QQQ/ARKK/VIX/TNX closes) |
| `data/intermediate/tsla_1h_news_sentiment.csv` | `processor.py` | FinBERT-scored and EMA-smoothed sentiment |
| `data/intermediate/tsla_1h_merged.csv` | `processor.py` | Prices + sentiment backward asof merge |
| `data/intermediate/tsla_1h_train_features.csv` | `processor.py` | Final engineered features in training window |
| `data/intermediate/tsla_1h_train_features_signature.json` | `train.py` | Train checkpoint compatibility signature |
| `data/intermediate/tsla_1h_test_features_signature.json` | `backtest.py` | Test checkpoint compatibility signature |
| `artifacts/<EXPERIMENT_NAME>/model.zip` | `train.py` | Trained PPO model |
| `artifacts/<EXPERIMENT_NAME>/scaler.pkl` | `train.py` | Fitted RobustScaler for training feature set |
| `artifacts/<EXPERIMENT_NAME>/metadata.json` | `train.py` | Full reproducibility receipt |
| `artifacts/<EXPERIMENT_NAME>/backtest_ledger.csv` | `backtest.py` | Per-step trade log over test window |
| `artifacts/<EXPERIMENT_NAME>/backtest_summary.json` | `backtest.py` | Compact machine-readable backtest summary |
| `artifacts/<EXPERIMENT_NAME>/reports/figures/actions_overlay.png` | `backtest.py` | Price chart with buy/sell/forced-exit markers |
| `artifacts/<EXPERIMENT_NAME>/reports/figures/equity_vs_benchmark.png` | `backtest.py` | Portfolio index vs TSLA buy-and-hold index |
| `artifacts/<EXPERIMENT_NAME>/reports/figures/drawdown_curve.png` | `backtest.py` | Drawdown curve over the backtest window |
| `artifacts/<EXPERIMENT_NAME>/reports/figures/trade_return_hist.png` | `backtest.py` | Histogram of cycle returns (if cycles exist) |
| `artifacts/<EXPERIMENT_NAME>/reports/tables/equity_timeseries.csv` | `backtest.py` | Time series of close and portfolio value per step |
| `artifacts/<EXPERIMENT_NAME>/reports/summary/report_index.json` | `backtest.py` | Report bundle index for downstream tooling |
| `artifacts/optuna_study.db` | `optimize.py` | Resumable SQLite Optuna trial history |
| `artifacts/experiment_journal.csv` | `train.py` + `backtest.py` | Central run registry with params and outcomes |

## How the Environment Adapts to Feature Count

This is automatic and end-to-end:

1. Change features in `settings.py` (`_BASE_FEATURES` and/or flags).
2. `EXPECTED_MARKET_FEATURES = len(FEATURES_LIST)` updates immediately.
3. `processor.py` computes exactly those features.
4. `prepare_features()` scales exactly those columns.
5. `TradingEnv` validates `features.shape[1] == settings.EXPECTED_MARKET_FEATURES`.
6. Observation width updates dynamically with portfolio state:
   - `self.num_features = features.shape[1] + 5`
7. PPO receives the correct observation shape via `spaces.Box(shape=(self.num_features,))`.

No manual shape edits are required anywhere.

## How to Add a New Feature

Example: add `OBV`.

Step 1. Add compute function in `core/processor.py`:

```python
def _compute_obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df
```

Step 2. Register it in `FEATURE_REGISTRY`:

```python
FEATURE_REGISTRY = {
    # ...
    "OBV": _compute_obv,
}
```

Step 3. Add it to `_BASE_FEATURES` in `config/settings.py`:

```python
_BASE_FEATURES = [
    "Log_Return", "Volume_Z_Score", "RSI",
    "MACD_Hist", "ATR_Pct",
    "OBV",
]
```

Done. `FEATURES_LIST`, `EXPECTED_MARKET_FEATURES`, scaler columns, and environment shape all update automatically.

If a feature name exists in settings but not in the registry, processing fails early with a clear error:

```text
Features requested but not in FEATURE_REGISTRY: ['OBV']
Add a _compute_xxx() function and register it.
```

## Running the Full Pipeline

```bash
# First-time data checkpoints
python scripts/data_engine.py
python scripts/news_engine.py
python scripts/macro_engine.py

# Optional hyperparameter search
python scripts/optimize.py
# Then set USE_OPTUNA_BEST_PARAMS = True in config/settings.py

# Train
python scripts/train.py

# Evaluate
python scripts/backtest.py

# Re-train after changing features
# train.py uses signature-aware checkpoints and rebuilds if settings/data changed
python scripts/train.py
```

## Core Principle

Single control plane, deterministic data flow, explicit checkpoints, and artifact-level reproducibility for every experiment run.


## Training diagnostics (`training_evaluate.py`)

`training_evaluate.py` generates training diagnostic plots from the Stable-Baselines3 TensorBoard logs created during training.

By default, it analyzes the **latest compatible experiment run** in the `artifacts/` directory:

```bash
python scripts/training_evaluate.py
```

The diagnostics are saved inside the corresponding artifact folder:

```
artifacts/<experiment_name>/training_diagnostics/
    plots/
    csv/
```

### Evaluating a specific run

To analyze a **specific previous experiment**, provide the artifact folder explicitly:

```bash
python scripts/training_evaluate.py --artifact-dir artifacts/<your_exact_run_folder>
```

The script reads the TensorBoard logs from that run and saves the diagnostics in the same artifact directory.


## Additional Guide

For practical tuning workflows, see EXPERIMENT_PLAYBOOK.md.
