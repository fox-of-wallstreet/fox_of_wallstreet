Below is a **revised version of your README**, keeping structure intact while integrating the latest changes (Optuna usage clarity, shared dataset logic, and training diagnostics). Style is kept consistent and concise.

---

# Fox of Wallstreet - Project Documentation

## Quick Start

1. Install dependencies:
   python -m pip install -r requirements.txt
2. Set Alpaca credentials:
   export ALPACA_API_KEY=your_key
   export ALPACA_SECRET_KEY=your_secret
3. Configure run in `config/settings.py` (symbol, timeframe, features, Optuna flags)
4. Build raw checkpoints:
   python scripts/data_engine.py
   python scripts/news_engine.py
   python scripts/macro_engine.py
5. (Optional) Hyperparameter search:
   python scripts/optimize.py
6. Train:
   python scripts/train.py
7. Evaluate:
   python scripts/backtest.py

Outputs are written to `data/` and `artifacts/<EXPERIMENT_NAME>/`.

---

## Architecture Overview

Fully modular PPO pipeline with **single control plane (`settings.py`)**.

```text
ettings.py (config)
      |
      v
data_engine.py / news_engine.py / macro_engine.py
      |
      v
processor.py  (shared dataset builder)
      |
      v
environment.py (TradingEnv)        → scaled features + raw df
      |
      +--> optimize.py             → artifacts/optuna_study.db
      |
      +--> train.py                → artifacts/<RUN>/
      |
      +--> training_evaluate.py    → evaluate training + diagnostic plots
      |
      +--> backtest.py             → additional reports + metrics
```

**Key principle:**
All scripts (`optimize`, `train`, `backtest`) reuse the same dataset construction logic → no duplication, no drift.

---

## scripts/train.py

What changed:

* Uses `build_training_dataset()` → identical preprocessing as optimization/backtest
* Signature-based caching for features
* `_resolve_ppo_params()`:

  * loads Optuna best params if enabled
  * otherwise uses settings defaults
* Reproducibility:

  * full metadata stored (`metadata.json`)
  * seed-controlled training

---

## scripts/optimize.py

What changed:

* Uses same dataset pipeline as training
* In-memory scaler (no dependency on saved scaler)
* Deterministic rollout evaluation (portfolio-based objective, not reward-only)
* Stores:

  * best params
  * diagnostics (return, drawdown, trades)

Important:

* Study name must match exactly between `optimize.py` and `train.py`
* DB: `artifacts/optuna_study.db`

---

## scripts/backtest.py

What changed:

* Reuses **training scaler + dataset logic** (no redundant FinBERT recomputation)
* Automatically resolves compatible run
* Produces full report bundle:

```
reports/
  figures/
  tables/
  summary/
```

Includes:

* actions overlay (improved color-coded trades)
* equity vs benchmark
* drawdown
* trade distribution

---

## Training Diagnostics (`training_evaluate.py`)

Generates diagnostics from SB3 TensorBoard logs.

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
OPTUNA_DB_PATH = artifacts/optuna_study.db
```

Workflow:

1. Run optimization:

```bash
python scripts/optimize.py
```

2. Train using best params:

```bash
python scripts/train.py
```

Notes:

* If study name mismatch → fallback to defaults
* Always verify via:

```text
DEBUG PPO PARAMS USED: {...}
```

Recommendation: use **fully qualified study names** (include action space, features, reward config).

---

## Data & Feature Pipeline

* Single source of truth: `settings.FEATURES_LIST`
* Processor:

  * builds dataset
  * computes indicators via registry
  * merges price + news + macro
* Environment:

  * dynamically adapts observation size
  * includes portfolio state

---

## Core Principle

* Single configuration source (`settings.py`)
* Shared dataset logic across all scripts
* Deterministic runs (seeded)
* Artifact-based reproducibility
* No hidden state between training, optimization, and evaluation

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

If you want, I can also tighten the Optuna section further into a “failure modes” checklist (study mismatch, DB mismatch, fallback behavior), which is often useful for team debugging.
