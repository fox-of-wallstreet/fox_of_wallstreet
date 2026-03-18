# Fox of Wallstreet – Project Documentation

## Quick Start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Set Alpaca credentials (for news ingestion):

   ```bash
   export ALPACA_API_KEY=your_key
   export ALPACA_SECRET_KEY=your_secret
   ```

3. Configure run in `config/settings.py`
   (symbol, timeframe, features, Optuna flags)

4. Build raw checkpoints:

   ```bash
   python scripts/data_engine.py
   python scripts/news_engine.py
   python scripts/macro_engine.py
   ```

5. (Optional) Hyperparameter optimization:

   ```bash
   python scripts/optimize.py
   ```

6. Train:

   ```bash
   python scripts/train.py
   ```

7. Evaluate:

   ```bash
   python scripts/backtest.py
   ```

8. Training diagnostics:

   ```bash
   python scripts/training_evaluate.py
   ```

Outputs are written to:

* `data/raw`, `data/intermediate`
* `artifacts/<EXPERIMENT_NAME>/`
* global index: `artifacts/experiment_journal.csv`

---

## Architecture Overview

Fully modular PPO pipeline with **single control plane (`settings.py`)**.

```text
settings.py (config)
      |
      v
data_engine.py / news_engine.py / macro_engine.py
      |
      v
processor.py  (shared dataset builder)
      |
      v
environment.py (TradingEnv)
      |
      +--> optimize.py           → artifacts/optuna_study.db
      |
      +--> train.py              → artifacts/<RUN>/
      |
      +--> training_evaluate.py  → diagnostics
      |
      +--> backtest.py           → reports + metrics
```

### Core Principle

* Single configuration source (`settings.py`)
* Shared dataset logic across all scripts
* No duplication between optimize/train/backtest
* Deterministic runs (seeded)
* Artifact-based reproducibility

---

## Data & Feature Pipeline

* Single source of truth: `settings.FEATURES_LIST`
* Processor:

  * merges price + news + macro
  * computes indicators via registry
* Environment:

  * dynamically adapts observation size
  * includes portfolio state

### Automatic feature flow

1. Define features in `settings.py`
2. Processor computes them
3. Scaler fits on training set
4. Environment adapts observation size automatically

No manual shape handling required.

---

## scripts/train.py

* Uses `build_training_dataset()` (shared logic)
* Signature-based caching for features
* `_resolve_ppo_params()`:

  * loads Optuna best params if enabled
  * otherwise uses defaults
* Full reproducibility via `metadata.json`
* Logs runs to `experiment_journal.csv`

---

## scripts/optimize.py

* Uses same dataset pipeline as training
* In-memory scaler (no dependency on saved scaler)
* Deterministic rollout evaluation:

  * portfolio-based objective (return + drawdown)
* Stores:

  * best params
  * diagnostics (return, drawdown, trades)

### Important

* Study name must match exactly between optimize/train
* DB: `artifacts/optuna_study.db`

---

## scripts/backtest.py

* Reuses **training scaler + dataset**
* Resolves compatible artifact automatically
* Produces report bundle:

```
reports/
  figures/
  tables/
  summary/
```

Includes:

* equity vs benchmark
* drawdown curve
* action overlays
* trade distribution

---

## Training Diagnostics (`training_evaluate.py`)

Default (latest run):

```bash
python scripts/training_evaluate.py
```

Specific run:

```bash
python scripts/training_evaluate.py --artifact-dir artifacts/<RUN>
```

Outputs:

```
artifacts/<RUN>/training_diagnostics/
```

---

## Optuna Usage

Settings:

```python
USE_OPTUNA_BEST_PARAMS = True
OPTUNA_STUDY_NAME = f"ppo_{SYMBOL.lower()}_{TIMEFRAME}"
OPTUNA_DB_PATH = "artifacts/optuna_study.db"
```

Workflow:

```bash
python scripts/optimize.py
python scripts/train.py
```

Notes:

* Study name mismatch → fallback to defaults
* Verify via console:

  ```
  DEBUG PPO PARAMS USED: {...}
  ```

Recommendation:
Use **fully qualified study names** (include action space, features, reward config)

---

## All Files Produced

| File                                                  | Produced by       | Purpose               |
| ----------------------------------------------------- | ----------------- | --------------------- |
| `data/raw/..._prices.csv`                             | `data_engine.py`  | Raw OHLCV checkpoint  |
| `data/raw/..._news.csv`                               | `news_engine.py`  | Raw news              |
| `data/raw/..._macro.csv`                              | `macro_engine.py` | Macro features        |
| `data/intermediate/..._news_sentiment.csv`            | `processor.py`    | Sentiment features    |
| `data/intermediate/..._merged.csv`                    | `processor.py`    | Merged dataset        |
| `data/intermediate/..._train_features.csv`            | `processor.py`    | Final features        |
| `data/intermediate/..._train_features_signature.json` | `train.py`        | Train cache signature |
| `data/intermediate/..._test_features_signature.json`  | `backtest.py`     | Test cache signature  |
| `artifacts/<RUN>/model.zip`                           | `train.py`        | PPO model             |
| `artifacts/<RUN>/scaler.pkl`                          | `train.py`        | Feature scaler        |
| `artifacts/<RUN>/metadata.json`                       | `train.py`        | Reproducibility       |
| `artifacts/<RUN>/backtest_ledger.csv`                 | `backtest.py`     | Trade log             |
| `artifacts/<RUN>/backtest_summary.json`               | `backtest.py`     | Metrics               |
| `artifacts/<RUN>/reports/...`                         | `backtest.py`     | Figures + tables      |
| `artifacts/optuna_study.db`                           | `optimize.py`     | Optuna DB             |
| `artifacts/experiment_journal.csv`                    | train + backtest  | Global run registry   |

---

## How to Add a New Feature

1. Implement compute function in `processor.py`
2. Register in `FEATURE_REGISTRY`
3. Add to `_BASE_FEATURES` in `settings.py`

The system updates automatically.

---

## Minimal Workflow

```bash
python scripts/data_engine.py
python scripts/news_engine.py
python scripts/macro_engine.py

python scripts/optimize.py   # optional
python scripts/train.py
python scripts/backtest.py
python scripts/training_evaluate.py
```

---

## Additional Notes

* Backtest automatically validates compatibility with training metadata
* Feature mismatches fail early with explicit errors
* All runs are reproducible via artifacts

---

## Summary

This system implements a **fully reproducible RL trading pipeline** with:

* unified data processing
* configurable feature engineering
* Optuna-based hyperparameter search
* deterministic training and evaluation
* artifact-based experiment tracking
