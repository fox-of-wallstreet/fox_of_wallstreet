# 🛡️ Sentinel V7: Multi-Stock AI Swing Trader

Status: Active | Framework: Stable Baselines 3 (PPO) | Architecture: MLOps Control Room

## 1. System Architecture (The Control Room)

Sentinel V7 is a Reinforcement Learning (RL) system managed entirely via a central "Control Room" (`config/settings.py`). Your team does not need to edit core logic to run experiments. By modifying `settings.py`, the entire pipeline adapts dynamically.

### The Pipeline

| Script | Role |
|---|---|
| `data_engine.py` | Fetches data based on `SYMBOL` and `TIMEFRAME` from settings |
| `processor.py` | Builds features and scales them, saving the scaler to the dynamic artifact vault |
| `train.py` | Trains the PPO agent and generates a `metadata.json` receipt |
| `backtest.py` | Validates the model on unseen data, exporting a transaction ledger |

---

## 2. Configurable Core Logic

### The Action Space (`ACTION_SPACE_TYPE`)

You can easily toggle between two heavily tested action spaces:

- **`"discrete_3"`** — Conviction Trading (`0=Sell All`, `1=Buy All`, `2=Hold`). Best for aggressive swing trading.
- **`"discrete_5"`** — Scaling (`0=Sell 100%`, `1=Sell 50%`, `2=Hold`, `3=Buy 50%`, `4=Buy 100%`). Best for testing advanced portfolio management.

### The Reward Strategy (`REWARD_STRATEGY`)

- **`"absolute_asymmetric"`** — Simulates the Sortino Ratio. Rewards profits 1:1, but punishes losses 2:1. Teaches the AI strict loss aversion and capital preservation.
- **`"pure_pnl"`** — Rewards and punishes strictly based on 1:1 portfolio percentage change.

> **Note:** All actions are subject to strict **Invalid Action Masking**. If the AI attempts to sell 0 shares, it is penalized `-0.05` points to prevent "Ghost Trading" loops.

---

## 3. The Artifact Vault (Reproducibility)

Every experiment automatically creates a dedicated folder in `artifacts/` named after your parameters (e.g., `ppo_TSLA_1h_discrete_3_v1`).

Each folder contains:

| File | Description |
|---|---|
| `model.zip` | The trained neural network weights |
| `scaler.pkl` | The `RobustScaler` fitted specifically to that training run |
| `metadata.json` | A receipt detailing the exact dates, features, and settings used |
| `backtest_ledger.csv` | A log of every transaction made during validation |

This ensures your team can perfectly reproduce any winning model without guessing which settings were used.

---

## 4. Operational Guide

```bash
# Step 1: Edit your experiment parameters
config/settings.py  # Set your Symbol, Timeframe, and Dates

# Step 2: Ingest raw data
python scripts/data_engine.py

# Step 3: Train the model and generate a metadata receipt
python scripts/train.py

# Step 4: Evaluate performance on unseen data
python scripts/backtest.py
