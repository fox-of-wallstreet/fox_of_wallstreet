# Live Trader App

Real-time AI trading dashboard for Fox of Wallstreet.

## Features

- **🧠 AI Decision Card**: See recommendations with confidence scores and key signals
- **📊 Position Sizing Preview**: Know exactly how much you're trading before executing
- **💰 P&L Tracker (Trading 212 style)**: Realized + Unrealized profit/loss with win rate stats
- **🧮 Position Size Calculator**: Manual calculator for planning trades
- **🔄 Multi-Stock Support**: Load models for different stocks, auto-fetches correct prices
- **🛡️ Three Trading Modes**: 
  - Simulate (virtual/paper trading)
  - Secure (real broker + confirmation)
  - Autopilot (fully automated)
- **⏱️ Auto-Refresh**: Schedule AI analysis automatically
- **✅ Price Verification**: Protects against slippage before executing
- **📈 Portfolio Tracking**: Real-time sync with Alpaca

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have trained models in artifacts/
# Run from project root:
python scripts/train.py

# Start the app
cd apps/live_trader
streamlit run app.py
```

## Usage

1. **Select a Model**: Go to Models page, choose a trained model, click Load
2. **Configure**: Go to Settings to add Alpaca API keys (optional for simulate mode)
3. **Trade**: Go to Trade Dashboard
   - View current price and chart
   - Click "Run AI Analysis"
   - Review AI recommendation with confidence score
   - Execute or override the action

## Architecture

```
app.py (entry)
    │
    ├── pages/01_trade.py      ← Main trading interface
    │   ├── fetches price data from yfinance
    │   ├── runs AI inference via utils/feature_fetcher.py
    │   ├── displays decision via components/decision_card.py
    │   └── executes trades (simulated or real)
    │
    ├── pages/02_models.py     ← Model browser/loader
    ├── pages/03_history.py    ← Trade history
    └── pages/04_settings.py   ← API keys
```

## Key Components

### utils/feature_fetcher.py
- `fetch_recent_prices()`: Get market data from yfinance
- `build_live_features()`: Compute technical indicators
- `run_ai_inference()`: Complete inference pipeline
- `calculate_confidence()`: Extract confidence from policy distribution

### components/decision_card.py
- `render_decision_card()`: Visual AI recommendation display
- `render_action_buttons()`: Execute/Override buttons
- `render_portfolio_card()`: Portfolio status

## Trading Modes

| Mode | Description |
|------|-------------|
| **Simulate** | Shows what AI would do, no real orders |
| **Secure** | AI suggests, you confirm each trade |
| **Autopilot** | AI executes automatically (use with caution!) |

## Session State

The app uses Streamlit's session_state to persist:
- `loaded_model`: The PPO model instance
- `scaler`: The fitted RobustScaler
- `model_info`: Metadata about loaded model
- `portfolio`: Cash, position, entry price
- `last_ai_decision`: Most recent inference result
- `activity_log`: Timestamped events

## Documentation

- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** - Trading modes, SELL_50/BUY_50 logic, budget explanation
- **[DATA_INTEGRITY.md](DATA_INTEGRITY.md)** - Pipeline verification, data sources, real vs simulated
- **[FAQ.md](FAQ.md)** - Common questions answered quickly
- **[MULTI_STOCK_SUPPORT.md](MULTI_STOCK_SUPPORT.md)** - How multi-stock trading works
- **[TRADING_MECHANICS.md](TRADING_MECHANICS.md)** - Price verification, schedules, hourly vs daily

## Quick Answers

| Question | Answer |
|----------|--------|
| Same pipeline as training? | ✅ YES - uses `core/processor.py` directly |
| Real prices? | ✅ YES - from Yahoo Finance (delayed ~15-20 min) |
| YF vs Alpaca prices? | ⚠️ Mostly same, can differ $0.01-$0.50 - see FAQ.md |
| Autonomous mode? | ✅ Auto-refresh available in sidebar |
| Alpaca trading? | ⚠️ Framework ready, UI buttons need wiring |
| Other stocks? | ✅ YES - auto-detected from model metadata |
| Daily timeframe? | ✅ YES - auto-detected from model metadata |

## Notes

- Feature computation reuses `core/processor.py` from the main project
- Observation includes stacked frames (VecFrameStack compatible)
- Confidence score derived from policy entropy
- News features default to 0 (requires Alpaca news API for live data)
