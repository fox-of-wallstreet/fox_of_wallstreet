<<<<<<< HEAD
# 🛡️ Sentinel V7 — Reinforcement Learning Swing Trading System

**Status:** Active Development
**Core Framework:** Stable-Baselines3 (PPO)
**Domain:** Algorithmic Trading / Reinforcement Learning
**Architecture:** Config-driven ML pipeline

Sentinel V7 is a **Reinforcement Learning based trading system** designed to learn swing-trading strategies from historical market data and news sentiment.

The system trains a **Proximal Policy Optimization (PPO)** agent to interact with a simulated trading environment and learn optimal buy/sell/hold decisions.

The architecture emphasizes:

* **Reproducibility**
* **Experiment tracking**
* **Modular ML pipelines**
* **Config-driven experimentation**

All experiments are controlled through a central configuration file (`settings.py`), allowing systematic testing of different environments, reward functions, and hyperparameters.
=======
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
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

Data flows in one direction:

<<<<<<< HEAD
# 1. Project Goal

The objective of Sentinel V7 is to build an **AI trading agent that can learn profitable trading behavior from historical data**, using:

* market indicators
* volatility regimes
* macro context
* sentiment information

The agent learns a **policy** that maximizes long-term portfolio value by interacting with a simulated trading environment.

Key goals:

• learn profitable swing-trading strategies
• minimize over-trading and drawdowns
• create a reproducible ML experimentation framework
• evaluate RL trading vs classical baselines
=======
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
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

### core/processor.py

<<<<<<< HEAD
# 2. System Architecture

The system is structured as a **modular ML pipeline**.

All major experiment parameters are defined in:

```
config/settings.py
```

This file acts as the **Control Room** of the project.

Changing parameters in this file automatically updates the behavior of the entire pipeline.
=======
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
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

What changed:

<<<<<<< HEAD
# 3. Project Pipeline

The project workflow consists of the following stages/python files.

| Step | Script           | Purpose                                     |
| ---- | ---------------- | ------------------------------------------- |
| 1    | `data_engine.py` | Downloads and builds the hybrid dataset     |
| 2    | `processor.py`   | Feature engineering and feature scaling     |
| 3    | `train.py`       | Trains the PPO reinforcement learning agent |
| 4    | `backtest.py`    | Evaluates the trained model on unseen data  |
| 5    | `optimize.py`    | Hyperparameter optimization using Optuna    |
| 6    | `live_trader.py` | Deploys the trained model for live trading  |

---

# 4. Data Sources

The system combines **market data and news sentiment**.

## Market Data — Yahoo Finance

Retrieved via:

```
yfinance
```

Used features include:

• Open
• High
• Low
• Close
• Volume

These form the base dataset for all technical indicators.

---

## News Data — Alpaca API

The Alpaca API is used to retrieve **financial news headlines** related to the traded asset.

From this data the system derives:

• news intensity (news frequency)
• sentiment scores

These features allow the agent to consider **market sentiment** alongside technical indicators.

---

# 5. Feature Engineering

The raw data is transformed into a set of **technical, macro, and contextual features**.

Feature generation occurs in:

```
core/processor.py
```

### 5.1 Technical Indicators

Derived from price and volume data.

Examples:

| Feature                 | Description                    |
| ----------------------- | ------------------------------ |
| RSI                     | Relative Strength Index        |
| MACD Histogram          | Momentum indicator             |
| Bollinger Band Position | Relative position within bands |
| ATR Percent             | Volatility indicator           |
| Volume Z-Score          | Normalized trading volume      |

---

### 5.2 Volatility Regime Features

The agent receives information about the current volatility environment.

Examples:

• short-term realized volatility
• long-term realized volatility
• volatility regime classification

---

### 5.3 Market Context

Macro-market indicators are incorporated to provide broader context.

Examples:

| Feature           | Source          |
| ----------------- | --------------- |
| QQQ returns       | Nasdaq ETF      |
| ARKK returns      | Innovation ETF  |
| Relative strength | Stock vs market |

---

### 5.4 Macro Risk Indicators

These capture overall market risk.

Examples:

| Feature | Description                 |
| ------- | --------------------------- |
| VIX_Z   | Z-score of volatility index |
| TNX_Z   | Z-score of treasury yields  |

---

### 5.5 Time Features

For hourly trading models the agent receives cyclical time information.

Examples:

• sin(time)
• cos(time)
• minutes to market close

These allow the model to learn **intraday behavioral patterns**.

---

# 6. Feature Scaling

All features are normalized using:

```
RobustScaler
```

This scaler is chosen because it is robust to financial data outliers.

The scaler is:

• fitted during training
• saved to the experiment artifact directory
• reused during inference and backtesting

---

# 7. Reinforcement Learning Environment

The trading environment is implemented in:

```
core/environment.py
```

The agent interacts with the environment in discrete timesteps.

At each timestep the agent receives:

```
Observation = market_features + portfolio_state
```

Portfolio state features include:

• whether a position is open
• unrealized profit/loss
• remaining cash ratio
• time spent in the trade

---

# 8. Action Space

The agent can operate under two trading styles.

### Discrete 3 (Conviction Trading)

```
0 → Sell all
1 → Buy all
2 → Hold
```

Encourages decisive swing-trading behavior.

---

### Discrete 5 (Scaling)

```
0 → Sell 100%
1 → Sell 50%
2 → Hold
3 → Buy 50%
4 → Buy 100%
```

Allows gradual position sizing.

---

# 9. Reward Function

The reward function determines how the agent learns.

Two strategies are implemented.

### Absolute Asymmetric Reward

```
profit reward = +1x
loss penalty = -2x
```

Encourages strong loss aversion and capital preservation.

---

### Pure PnL Reward

```
reward = portfolio return
```

Directly optimizes profitability.

---

# 10. Hyperparameter Optimization

The PPO hyperparameters are optimized using **Optuna**.

```
scripts/optimize.py
```

Parameters optimized include:

• learning rate
• batch size
• discount factor (gamma)
• entropy coefficient

Optimization uses a **train/validation split** to avoid overfitting.

---

# 11. Artifact Tracking (Reproducibility)

Each experiment automatically generates an artifact directory.

Example:

```
artifacts/
ppo_TSLA_1h_d5_asym_pen10_lr00007_bs128_g091_v1/
```

This folder contains:

| File                | Purpose               |
| ------------------- | --------------------- |
| model.zip           | trained PPO policy    |
| scaler.pkl          | feature scaler        |
| metadata.json       | experiment parameters |
| backtest_ledger.csv | executed trades       |

This design ensures **full reproducibility of every experiment**.

---

# 12. Running the Pipeline

### 1️⃣ Configure experiment

Edit:

```
config/settings.py
```

Set:

• asset symbol
• timeframe
• action space
• reward function
• training dates

---

### 2️⃣ Build dataset

```
python scripts/data_engine.py
```

Downloads market and news data.

---

### 3️⃣ Train the RL agent

```
=======
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
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
python scripts/train.py
```

<<<<<<< HEAD
This step:

• creates the environment
• trains the PPO model
• saves the trained policy

---

### 4️⃣ Backtest the strategy

```
python scripts/backtest.py
```

This evaluates the model on **unseen test data**.

Outputs include:

• portfolio value
• total return
• transaction log

---

### 5️⃣ Optimize PPO hyperparameters (optional)

```
python scripts/optimize.py
```

Runs an Optuna search to find better PPO settings.

---

# 13. Future Work

Potential improvements include:

• multi-asset portfolios
• risk-adjusted reward functions
• transaction cost modeling
• ensemble RL agents
• live trading deployment

---

If you want, I can also help you add **two things that teachers often expect in ML project READMEs but are currently missing**:

1️⃣ **A visual architecture diagram of the pipeline**
2️⃣ **An explanation of PPO and reinforcement learning in this project**

Both usually improve project grading significantly.
=======
# Evaluate
python scripts/backtest.py

# Re-train after changing features
# train.py uses signature-aware checkpoints and rebuilds if settings/data changed
python scripts/train.py
```

## Core Principle

Single control plane, deterministic data flow, explicit checkpoints, and artifact-level reproducibility for every experiment run.

## Additional Guide

For practical tuning workflows, see EXPERIMENT_PLAYBOOK.md.
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
