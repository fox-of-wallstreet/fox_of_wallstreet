# Live Trader App - Architecture Document

## Purpose
Real-time AI trading dashboard for production use. This app consumes trained models and executes/monitor trades. It does NOT train models.

## Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│   (Separate - runs offline, produces artifacts)                 │
│   • scripts/train.py                                            │
│   • scripts/optimize.py                                         │
│   • scripts/data_engine.py                                      │
│   • core/processor.py                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ produces model.zip + scaler.pkl
┌─────────────────────────────────────────────────────────────────┐
│                    ARTIFACTS STORAGE                            │
│   • artifacts/ppo_TSLA_1h_20260318_1445/                        │
│     ├── model.zip                                               │
│     ├── scaler.pkl                                              │
│     └── metadata.json                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ consumed by
┌─────────────────────────────────────────────────────────────────┐
│  🎯 LIVE TRADER APP (This Application)                          │
│                                                                 │
│  Responsibilities:                                              │
│  • Load and validate trained models                             │
│  • Fetch real-time market data                                  │
│  • Run inference (model.predict)                                │
│  • Display AI decisions to user                                 │
│  • Execute trades (paper or live)                               │
│  • Monitor portfolio performance                                │
│                                                                 │
│  Explicitly NOT responsible for:                                │
│  • Training new models                                          │
│  • Hyperparameter optimization                                  │
│  • Feature engineering from scratch                             │
│  • Backtesting on historical data                               │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Principles

### 1. Model-Agnostic Design
The app should work with any trained model regardless of:
- Action space (discrete_3 vs discrete_5)
- Feature set (18 features or different combination)
- Training hyperparameters

**Detection**: Read `metadata.json` from selected artifact.

### 2. Safety-First Trading
Three modes with increasing risk:
```
SIMULATE  →  SECURE  →  AUTOPILOT
   │            │           │
   │            │           └── AI executes immediately
   │            │
   │            └── AI suggests, user confirms via UI
   │
   └── AI suggests, shows what would happen, no orders
```

### 3. Real-Time Data Flow
```
Price Feed (yfinance/Alpaca)
    │
    ▼
Feature Builder (core/processor.py reuse)
    │
    ▼
Scaler (from artifacts/)
    │
    ▼
Model Inference
    │
    ▼
Action Display → User Confirmation (if needed) → Order Execution
```

## Data Flow Architecture

### State Management
```python
# Streamlit session_state structure
session_state = {
    # Model State
    "loaded_model": PPO model instance | None,
    "model_metadata": dict | None,
    "scaler": RobustScaler instance | None,
    
    # Trading State
    "trading_mode": "simulate" | "secure" | "autopilot",
    "alpaca_client": TradingClient | None,
    "last_ai_decision": {
        "timestamp": str,
        "action": int,
        "action_name": str,
        "confidence": float,
        "features": dict,
    } | None,
    
    # Portfolio State (fetched from Alpaca)
    "portfolio": {
        "cash": float,
        "position": float,
        "entry_price": float | None,
        "unrealized_pnl": float,
    } | None,
    
    # UI State
    "selected_artifact": str,
    "price_history": pd.DataFrame,
    "activity_log": list,
}
```

### Component Hierarchy
```
app.py (entry point)
│
├── 📄 pages/
│   ├── 01_trade.py          ← Main interface (default)
│   ├── 02_models.py         ← Model browser
│   ├── 03_history.py        ← Past trades
│   └── 04_settings.py       ← API keys, preferences
│
├── 🧩 components/
│   ├── model_selector.py    ← Artifact dropdown + load
│   ├── decision_card.py     ← AI decision display
│   ├── feature_panel.py     ← Current feature values
│   ├── portfolio_card.py    ← Cash, position, P&L
│   ├── price_chart.py       ← Real-time price
│   ├── action_buttons.py    ← Execute/Override buttons
│   └── activity_log.py      ← Timestamped events
│
└── 🔧 utils/
    ├── model_loader.py      ← Load model.zip + metadata
    ├── feature_fetcher.py   ← yfinance/Alpaca data
    ├── alpaca_client.py     ← TradingClient wrapper
    └── action_mapper.py     ← int → action_name
```

## Key Technical Decisions

### 1. Model Loading
- Models loaded once per session (cached via `@st.cache_resource`)
- Validation: check metadata matches current settings.py
- Graceful fallback: if current artifact missing, suggest alternatives

### 2. Real-Time Prices
**Option A: yfinance** (simpler, no API keys needed)
- Poll every 60 seconds
- Good for demo

**Option B: Alpaca Market Data** (more professional)
- Requires API keys
- More accurate for live trading
- Recommended for production

**Decision**: Support both, default to yfinance for zero-friction demo.

### 3. Feature Computation
Reuse `core/processor.py` functions:
- `load_raw_prices()` → fetch recent prices
- `add_technical_indicators()` → compute features
- `prepare_features()` → scale using artifact's scaler

### 4. Action Space Handling
Dynamic UI based on `metadata['action_space']`:

```python
def render_action_buttons(action_space: str):
    if action_space == "discrete_3":
        cols = st.columns(3)
        return cols[0].button("SELL ALL"), cols[1].button("HOLD"), cols[2].button("BUY ALL")
    else:  # discrete_5
        cols = st.columns(5)
        return (
            cols[0].button("SELL 100%"),
            cols[1].button("SELL 50%"),
            cols[2].button("HOLD"),
            cols[3].button("BUY 50%"),
            cols[4].button("BUY 100%"),
        )
```

### 5. Order Execution
```python
async def execute_order(action: int, mode: str):
    if mode == "simulate":
        log_simulated_order(action)
        return {"status": "simulated"}
    
    elif mode == "secure":
        user_confirmed = await show_confirmation_dialog(action)
        if not user_confirmed:
            return {"status": "rejected_by_user"}
        return submit_to_alpaca(action)
    
    else:  # autopilot
        return submit_to_alpaca(action)
```

## User Flow

### First-Time User
1. Land on Settings page
2. Enter Alpaca API keys (optional, can use simulate)
3. Go to Models page, see list of trained models
4. Select model, view metadata
5. Go to Trade page
6. Click "Load Model"
7. See current market snapshot
8. Click "Run AI Analysis"
9. See AI recommendation with confidence
10. Click "Simulate" to see what would happen
11. Switch to "Secure" mode
12. Run AI again, click "Execute" on recommendation

### Regular User
1. App loads with last used model
2. Auto-refreshes price every 60s
3. User clicks "Run AI Cycle" each hour
4. Decision card shows recommendation
5. One-click execute or override

## Error Handling Strategy

| Error | User Experience | Recovery |
|-------|-----------------|----------|
| Model load fails | Red banner: "Model incompatible" | Suggest compatible models |
| Alpaca API error | Yellow banner: "Trading disabled, showing simulation" | Fallback to simulate mode |
| Feature fetch fails | Retry 3x, then "Data unavailable" | Manual refresh button |
| Action space mismatch | "This UI doesn't support discrete_5" | Redirect to models page |

## Files to Create

```
apps/live_trader/
├── ARCHITECTURE.md          ← This file
├── requirements.txt         ← streamlit, plotly, alpaca-py
├── app.py                   ← Entry point, sidebar nav
├── pages/
│   ├── 01_trade.py
│   ├── 02_models.py
│   ├── 03_history.py
│   └── 04_settings.py
├── components/
│   ├── __init__.py
│   ├── model_selector.py
│   ├── decision_card.py
│   ├── feature_panel.py
│   ├── portfolio_card.py
│   ├── price_chart.py
│   ├── action_buttons.py
│   └── activity_log.py
└── utils/
    ├── __init__.py
    ├── model_loader.py
    ├── feature_fetcher.py
    ├── alpaca_client.py
    └── action_mapper.py
```

## Dependencies on Core Project

This app imports from parent project:
```python
# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Reuse these
from config import settings
from core.processor import add_technical_indicators, load_raw_prices
from core.environment import TradingEnv  # For constants only
```

## Success Criteria

- [ ] User can load any model from artifacts/
- [ ] App auto-detects action space and adjusts UI
- [ ] Real-time price updates every 60s
- [ ] AI inference completes in <2 seconds
- [ ] Three trading modes work correctly
- [ ] Activity log shows all events
- [ ] Portfolio value updates after trades
