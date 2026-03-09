# 🛡️ Sentinel V7: AI Algorithmic Trading System

**Status:** Live / Active  
**Framework:** Stable Baselines3 (PPO)  
**Markets:** `TSLA`, `NVDA`, `AAPL`

Sentinel V7 is a Reinforcement Learning-based algorithmic trading system built for long-only swing trading on Alpaca markets. It combines financial news sentiment from FinBERT with macroeconomic and market features to make hourly trading decisions using a PPO agent.

## System Architecture

Sentinel V7 is designed with a modular, one-click pipeline. By changing the `SYMBOL` in `config/settings.py`, the full data, feature engineering, optimization, training, and backtesting workflow automatically adapts.

### One-Click Pipeline

- `data_engine.py` fetches \(1\)-hour candles from `yfinance` and real-time news from Alpaca.
- `processor.py` scores headlines with FinBERT and builds \(16\) micro/macro/contextual features.
- `optimize.py` uses Optuna Bayesian Optimization to search for the best hyperparameters.
- `train.py` trains the PPO agent with a \(5\)-hour rolling memory using `VecFrameStack`.
- `backtest.py` evaluates the trained model on an unseen holdout dataset.

## The RL Brain

The agent observes a rolling "combat dashboard" of \(20\) variables every hour. These are stacked over a \(5\)-hour memory window using `VecFrameStack(n_stack=5)`, resulting in \(100\) values per observation.

### Observation Space

| Category | Features | Scaled? |
|---|---|---|
| Micro (Asset) | Log Returns, Volatility Z-Score, RSI, MACD Histogram, Bollinger `%B`, ATR `%` | Yes (`RobustScaler`) |
| Macro (Market) | QQQ Returns, ARKK Returns, Relative Strength, VIX, TNX | Yes (`RobustScaler`) |
| Context (News) | FinBERT Sentiment EMA, News Intensity, Sin/Cos Time, Minutes to Close | Yes (`RobustScaler`) |
| Portfolio | Position Flag, Unrealized PnL `%`, Cash Ratio, Time-in-Trade | Manually scaled |

## Action Space

To avoid the "hovering trap" often seen in continuous action spaces while still preserving portfolio sizing flexibility, Sentinel V7 uses an extended discrete action space with \(5\) bins.

### Extended Discrete Actions

- `0` — **Strong Sell**: Liquidate \(100\%\) of held shares.
- `1` — **Light Sell**: Liquidate \(50\%\) of held shares.
- `2` — **Hold**: Maintain current portfolio state.
- `3` — **Light Buy**: Invest \(50\%\) of available cash.
- `4` — **Strong Buy**: Invest \(100\%\) of available cash.

> **Note:** Short selling is disabled to prevent unlimited downside risk.

## Reward Function

Sentinel V7 uses an absolute return reward baseline with asymmetric loss aversion.

### Trading Friction

Each transaction includes a simulated round-trip slippage/fee cost of \(0.1\%\), modeled as \(0.05\%\) per side.

### Reward Logic

- If the portfolio grows by \(1\%\) in an hour, reward \(+= 1.0\)
- If the portfolio drops by \(1\%\) in an hour, reward \(-= 2.0\)

This asymmetry penalizes losses twice as heavily as gains, encouraging the agent to develop dynamic stop-loss-like behavior without hardcoded rules.

## Project Structure

```text
Final_Project/
├── artifacts/              # STORE: Models (.zip), scalers (.pkl), and trade ledgers
├── config/                 # CONTROL: settings.py and environment variables
├── core/                   # ENGINE:
│   ├── environment.py      # Custom Gym trading environment
│   └── processor.py        # Feature engineering and scaling logic
├── data/                   # DATA: Raw and processed historical CSVs
├── scripts/                # TRIGGERS:
│   ├── optimize.py         # Optuna Bayesian hyperparameter search
│   ├── train.py            # PPO training script
│   ├── backtest.py         # Holdout evaluation script
│   └── live_trader.py      # Live deployment via Alpaca and Telegram
```

## Maintenance and Troubleshooting

### Feature Mismatch Error

```text
ValueError: Unexpected observation shape
```

**Cause:**  
The feature list in `processor.py` was changed, or the lookback window was modified without retraining the `RobustScaler`.

**Solution:**

- Delete the `.zip` and `.pkl` files in `artifacts/`
- Clear all `__pycache__` folders
- Re-run `train.py`

### Sentiment Lag

The data engine scores the latest \(100+\) headlines. If FinBERT runs slowly:

- Check that your internet connection is stable
- Configure the model to run on a local GPU using CUDA or MPS if available

## Notes

- Strategy type: **Long-only swing trading**
- Core algorithm: **Proximal Policy Optimization (PPO)**
- Sentiment engine: **FinBERT**
- Broker/integration: **Alpaca**
- Memory mechanism: **`VecFrameStack(n_stack=5)`**

## Disclaimer

This project is for research and educational purposes only. It does not constitute financial advice. Live trading involves risk, including the potential loss of capital.
