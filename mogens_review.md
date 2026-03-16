MD file explaining another project architecture and the progress of the each script, to add this as fresh AI context and as guidline of merging these progresses
=========================
# Important features and advances script-by-script — Mogens

This document summarizes the most important conceptual and architectural advances that were developed and tested in Mogens’ branch, so they can be merged into the teammate architecture even if the exact code structure differs.

The goal is **not** to copy code blindly, but to transfer the **most important ideas, interfaces, artifacts, and evaluation logic** into the current project architecture.

---

## 0. Global conceptual advances across the whole project

These are the most important cross-script improvements. They should be merged even if the exact implementation changes.

### 0.1 Experiment-centric artifact system

Each experiment creates its own artifact folder. This is crucial for:

* reproducibility
* comparison across agents
* later deployment of selected agents

Typical artifact folder contents now include:

* `model.zip`
* `scaler.pkl`
* `metadata.json`
* `backtest_ledger.csv`
* `backtest_summary.json`
* `backtest_timeseries.csv`
* optional backtest plot(s)
* optional TensorBoard logs
* optional training diagnostics

### 0.2 Clean experiment naming convention

Experiment names should encode only the **main experiment axes**, not every low-level hyperparameter.

Recommended pattern:

```python
ppo_{SYMBOL}_{TIMEFRAME}_{ACTION}_{REWARD}_t{TIMESTEPS}_s{SEED}_v{VERSION}
```

Example:

```python
ppo_TSLA_1h_d5_asym_t200k_s42_v1
```

Low-level settings such as learning rate, gamma, penalties, feature windows, etc. belong in `metadata.json`, not in the artifact folder name.

### 0.3 Structured metadata tracking

`metadata.json` was expanded substantially and should contain:

* experiment identity
* train/test split
* dataset statistics after preprocessing
* training configuration
* feature engineering settings
* environment settings
* PPO hyperparameters
* artifact paths

### 0.4 Standardized backtest outputs

Backtesting now produces standardized artifacts:

* `backtest_ledger.csv`
* `backtest_summary.json`
* `backtest_timeseries.csv`
* optional diagnostic plot(s)

### 0.5 Master experiment comparison table

A separate script (`evaluate_experiments.py`) scans all artifact folders and builds a master experiment summary table across all agents. This was one of the biggest methodological improvements.

### 0.6 Training diagnostics separated from training

TensorBoard logging is enabled during training, and a separate script (`training_evaluate.py`) extracts and saves training diagnostic plots and CSVs.

### 0.7 Timeframe-aware configuration

Several settings were made conditional on `TIMEFRAME` (`1h` vs `1d`) so that feature windows, stack sizes, and data horizons are economically more meaningful.

---

## 1. `core/tools.py`

### Main purpose

Contains utility functions used across several scripts to centralize shared logic and avoid duplication.

### Important advances

#### `get_features_list()`

Returns the exact ordered list of features used by the model.

This is critical because:

* training
* backtesting
* live inference
* metadata tracking

must all use the **same feature order**.

#### `get_stack_size()`

Returns how many past timesteps are stacked into the observation.

Example logic:

* `1h` → `5`
* `1d` → `10`

This gives the feedforward PPO model short-term temporal context, because `MlpPolicy` has no internal memory.

#### `fnline()`

Formatting helper for console logging.

### Why it matters

These helper functions guarantee consistency across:

* training
* backtesting
* optimization
* live inference

### Merge priority

**High**

---

## 2. `config/settings.py`

### Main purpose

Central control room for the project.

### Important advances

### 2.1 Flexible data split design

The train/test split was made timeframe-aware.

Example:

* `1h` uses a shorter historical training span
* `1d` uses a longer historical training span

This prevents daily agents from being trained on too few observations relative to hourly agents.

### 2.2 Centralized feature-engineering windows

Parameters such as:

* `RSI_WINDOW`
* `MACD_FAST`
* `MACD_SLOW`
* `MACD_SIGNAL`
* `SHORT_VOL_WINDOW`
* `LONG_VOL_WINDOW`
* `MA_FAST_WINDOW`
* `MA_SLOW_WINDOW`

are defined in `settings.py` and used in `core/processor.py`.

These define the rolling look-back windows for indicator computation.

### 2.3 Centralized environment settings

Environment and reward-shaping settings are also defined here, including:

* `CASH_RISK_FRACTION`
* `STOP_LOSS_PCT`
* `TAKE_PROFIT_PCT`
* `TRADE_PENALTY_FULL`
* `TRADE_PENALTY_HALF`
* `INVALID_ACTION_PENALTY`
* `BANKRUPTCY_PENALTY`

These are not PPO hyperparameters; they define the behavior of the RL environment.

### 2.4 Artifact and logging paths created here

Important paths created from the experiment name include:

* `ARTIFACT_DIR`
* `MODEL_PATH`
* `SCALER_PATH`
* `METADATA_PATH`
* `BACKTEST_SUMMARY_PATH`
* `BACKTEST_TIMESERIES_PATH`
* `TB_LOG_DIR`
* `TRAINING_DIAGNOSTICS_DIR`

### 2.5 Improved experiment naming logic

A helper such as `build_experiment_name()` creates short, readable, reproducible experiment names.

### 2.6 Explicit seed and versioning

Settings include:

* `RANDOM_SEED`
* `EXPERIMENT_VERSION`

### Why it matters

This script is the main source of truth for:

* data split design
* environment rules
* feature-engineering windows
* training configuration
* artifact structure

### Merge priority

**Very high**

---

## 3. `core/processor.py`

### Main purpose

Transforms raw hybrid data into engineered features used by the RL agent.

### Important advances

### 3.1 Technical indicators use centralized settings

The feature windows are no longer hard-coded. They are controlled by `settings.py`.

### 3.2 Feature engineering made explicit and auditable

The feature set includes:

* price/momentum features
* volume features
* volatility regime features
* moving-average distance features
* market context features
* macro features
* sentiment/news features
* time-of-day features for hourly mode

### 3.3 Scaling is separated from feature generation

`prepare_features(...)` handles scaling and uses the saved scaler when `is_training=False`.

This is critical for:

* backtesting
* live inference

### Why it matters

This script defines the actual model input. Any mismatch here breaks reproducibility between:

* training
* backtesting
* live trading

### Outputs / artifacts

Indirectly produces:

* scaled feature matrix
* saved `scaler.pkl`

### Merge priority

**Very high**

---

## 4. `core/environment.py`

### Main purpose

Defines the RL trading environment.

### Important advances

### 4.1 Observation = engineered market features + portfolio state

Portfolio-state features include:

* in position / not in position
* unrealized PnL
* cash ratio
* bars in trade

### 4.2 Support for both action spaces

* `discrete_3`
* `discrete_5`

### 4.3 Reward shaping via configurable penalties

Includes:

* trade penalties
* invalid action penalty
* bankruptcy penalty

### 4.4 Action logic is settings-driven

Trade execution behavior is controlled via settings such as:

* `CASH_RISK_FRACTION`
* `TRADE_PENALTY_FULL`
* `TRADE_PENALTY_HALF`

### Why it matters

This is the core RL problem definition. Changes here directly affect:

* profitability
* overtrading
* stability
* risk behavior

### Merge priority

**Very high**

---

## 5. `scripts/train.py`

### Main purpose

Train one PPO agent on the configured training period and save all training artifacts.

### Important advances

### 5.1 Cleaner PPO initialization

Training now uses:

* settings-driven PPO hyperparameters
* explicit seed
* experiment-specific TensorBoard logging

### 5.2 TensorBoard logging enabled

PPO now writes training diagnostics to an experiment-specific log directory inside the artifact folder.

### 5.3 Expanded metadata generation

`metadata.json` now contains:

#### Experiment identity

* experiment name
* symbol
* timeframe
* action space
* reward strategy
* seed
* version

#### Data split

* train start/end
* test start/end

#### Dataset statistics

* raw rows loaded
* rows after feature engineering
* train rows
* validation rows
* total rows used

#### Training configuration

* total timesteps
* timesteps per train row
* cash risk fraction
* stop loss
* take profit
* max bars normalization

#### Feature-engineering configuration

* RSI / MACD / volatility / moving average windows
* `features_used`
* `n_features`
* `stack_size`

#### Environment settings

* initial balance
* slippage
* trade penalties
* invalid action penalty
* bankruptcy penalty
* thresholds / normalization

#### PPO hyperparameters

* learning rate
* batch size
* gamma
* entropy coefficient

#### Artifact paths

* model path
* scaler path
* metadata path
* artifact dir

### Why it matters

Training now produces a full experiment receipt, not just a trained model.

### Outputs / artifacts

* `model.zip`
* `scaler.pkl`
* `metadata.json`
* TensorBoard logs in `tb_logs/`

### Merge priority

**Very high**

---

## 6. `scripts/backtest.py`

### Main purpose

Run out-of-sample evaluation on the test period and save standardized backtest outputs.

### Important advances

### 6.1 Backtest metrics are much richer

The backtest now reports and saves metrics such as:

* final portfolio value
* total return
* total real transactions
* trades per 100 bars
* completed cycles
* average holding duration
* average PnL per cycle
* average return per cycle

### 6.2 Metrics are configuration-aware

For `discrete_3`:

* metrics are interpreted as completed trades

For `discrete_5`:

* metrics are interpreted as flat-to-flat position episodes

This distinction is important because `discrete_5` allows scaling in and out.

### 6.3 Backtest ledger saved

`backtest_ledger.csv` records transaction-level events including:

* `Date`
* `Action`
* `Price`
* `Portfolio_Value`
* `Position_Before`
* `Position_After`

### 6.4 Backtest summary saved

`backtest_summary.json` stores the core summary metrics in a compact structured format.

### 6.5 Full stepwise backtest timeseries saved

`backtest_timeseries.csv` stores one row per evaluation bar, including portfolio value across time.

This enables:

* proper equity curve plots
* drawdown analysis
* benchmark comparison later

### 6.6 Improved backtest plotting

Plots now include:

* top panel: close price + action markers
* bottom panel: portfolio value over the evaluation period

Action colors were improved:

* `discrete_3`: buy = blue, sell = red
* `discrete_5`: strong and light buy/sell use different shades

### Why it matters

This was one of the biggest methodological improvements. Backtesting is no longer just console output; it now produces proper experiment artifacts.

### Outputs / artifacts

* `backtest_ledger.csv`
* `backtest_summary.json`
* `backtest_timeseries.csv`
* optional plot PNG

### Merge priority

**Very high**

---

## 7. `scripts/optimize.py`

### Main purpose

Run Optuna-based hyperparameter optimization for PPO.

### Important advances

### 7.1 Validation-based optimization

Optimization is done using a train/validation split, not directly on the final test set.

### 7.2 Explicit optimization objective

The objective uses:

* validation return
* drawdown penalty
* optionally trade-count penalty

Examples of explored objective styles:

* aggressive
* balanced
* conservative

### 7.3 Search space was revised conceptually

Especially important:

* `gamma` should include lower values around `0.90–0.94`
* `gamma` should be searched **linearly**, not logarithmically
* learning rate bounds should include the empirically promising region

### 7.4 Study naming/versioning matters

Optuna study names should include a version or search-space version. Otherwise changing categorical choices breaks compatibility with old studies.

### 7.5 Optimization training horizon matters

Optimization timesteps should not be too short relative to the real useful training horizon.

### Why it matters

This script became much more than brute-force tuning. It now acts as a controlled model-selection layer, but only if the search space and objective are aligned with the actual task.

### Merge priority

**High**

---

## 8. `scripts/evaluate_experiments.py`

### Main purpose

Build a master comparison table across all experiment artifact folders.

### Important advances

### 8.1 Scans all artifact folders

Each experiment becomes one row in the master summary table.

### 8.2 Prefers `backtest_summary.json`

If present, the script loads the standardized summary directly.

### 8.3 Falls back to the ledger

If the summary file is missing, some metrics can still be recomputed from `backtest_ledger.csv`.

### 8.4 Produces a master experiment registry

This becomes the central CSV for comparing agents across:

* timeframes
* action spaces
* reward strategies
* PPO settings
* training lengths

### Why it matters

This was one of the main transitions from ad hoc experimentation to systematic evaluation.

### Outputs / artifacts

* `experiment_summary.csv`

### Merge priority

**Very high**

---

## 9. `scripts/training_evaluate.py`

### Main purpose

Read TensorBoard logs and generate training diagnostic plots and CSVs.

### Important advances

### 9.1 Supports current experiment and retrospective analysis

By default it analyzes the current experiment from `settings.py`, but it can also analyze an older artifact folder.

### 9.2 Exports the main RL training diagnostics

The most important tracked diagnostics include:

* episode reward mean
* episode length mean
* entropy loss
* value loss
* explained variance
* approx KL
* clip fraction
* total loss
* policy gradient loss (if present)

### 9.3 Saves outputs in a structured diagnostics folder

Recommended structure:

* `training_diagnostics/plots/`
* `training_diagnostics/csv/`

### 9.4 Creates a combined overview plot

This gives a compact view of the learning process and is useful for identifying:

* undertraining
* overtraining
* instability
* policy collapse

### Important implementation note

To reliably obtain episode reward and episode length, the training environment should be wrapped with `VecMonitor`.

### Why it matters

This script is the RL equivalent of supervised-learning training curves and is important for diagnosing:

* whether the agent is still learning
* when learning plateaus
* whether PPO updates are stable

### Merge priority

**High**

---

## 10. `scripts/live_trader.py`

### Main purpose

Single-run live inference script for Alpaca paper trading plus Telegram notification.

### Important advances / observations

### 10.1 Uses pretrained artifacts

It loads:

* `model.zip`
* `scaler.pkl`

### 10.2 Recomputes live engineered features before inference

This is required because the model was trained on engineered features, not raw price data.

### 10.3 Supports both action spaces

* `discrete_3`
* `discrete_5`

### 10.4 Can trade autonomously in paper mode

The current implementation already submits Alpaca paper orders, not only Telegram recommendations.

### 10.5 Also sends Telegram alerts

Reports:

* buy/sell execution
* hold/no-action cases

### Important limitations identified

These are conceptually important for future refinement:

1. it currently executes one decision cycle and exits; it is not a continuously scheduled daemon
2. it lacks persistent local state for `bars_in_trade`
3. it should ideally support both:

   * recommendation-only mode
   * autonomous execution mode
4. it should confirm order status/fill result more explicitly
5. it should use a rolling recent data window rather than rebuilding large datasets

### Why it matters

This script is already a functioning prototype for deployment, but it is not yet the most robust production-ready live-trading architecture.

### Merge priority

**Medium**

---

## 11. Important methodological findings from experiments so far

These are not tied to one single script, but they are very important for the teammate AI to know.

### 11.1 Training length is crucial

Longer PPO training does **not** automatically improve the agent.

For TSLA `1h` `discrete_5`, the best observed performance was around **200k timesteps**, while longer runs degraded.

### 11.2 Action space matters strongly

For TSLA `1h`:

* `discrete_5` was profitable
* `discrete_3` traded less but was not profitable

### 11.3 PPO optimization can underperform defaults

If the Optuna search space or objective is poorly aligned, the “optimized” PPO configuration can be worse than the default/manual one.

### 11.4 Experiment infrastructure is now mature enough for systematic sweeps

This includes:

* training-length sweeps
* action-space comparison
* reward comparison
* later transfer to other stocks

---

## 12. Recommended merge priority

### Merge first

These are highest priority because they define the core workflow and artifact/evaluation structure:

* `settings.py`
* `tools.py`
* `processor.py`
* `environment.py`
* `train.py`
* `backtest.py`
* `evaluate_experiments.py`

### Merge second

These add important methodology but are not absolute blockers:

* `optimize.py`
* `training_evaluate.py`

### Merge later

Useful but less urgent:

* `live_trader.py`

---

## 13. Recommended handover package

For the teammate / teammate AI, the best additional context to provide is:

1. this markdown document
2. one clean example artifact folder from Mogens’ branch
3. one current `metadata.json`
4. one current `backtest_summary.json`
5. one current `experiment_summary.csv`

This gives both:

* the intended architecture
* the exact output formats expected by the new workflow

---

## 14. Final merge philosophy

The most important goal is **not** to copy Mogens’ scripts line-by-line into the teammate architecture.

The important goal is to preserve these major advances:

* experiment-centric artifact system
* structured metadata
* standardized backtest outputs
* master experiment comparison table
* training diagnostics
* timeframe-aware settings
* stronger evaluation methodology

If these concepts are preserved, the exact implementation can be adapted to the teammate architecture without losing the key progress.
