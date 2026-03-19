# Fox of Wallstreet - Streamlit Apps

This directory contains two complementary Streamlit applications for interacting with trained trading models.

## Apps Overview

```
apps/
├── 📁 live_trader/     ← Real-time AI trading dashboard
├── 📁 backtester/      ← Historical analysis & model comparison
└── 📁 shared/          ← Common components and utilities
```

## Separation of Concerns

| Component | Training | Backtesting | Live Trading | Analysis |
|-----------|----------|-------------|--------------|----------|
| `scripts/train.py` | ✅ | ❌ | ❌ | ❌ |
| `scripts/backtest.py` | ❌ | ✅ | ❌ | ❌ |
| `scripts/live_trader.py` | ❌ | ❌ | ✅ | ❌ |
| `apps/backtester/` | ❌ | ❌ | ❌ | ✅ |
| `apps/live_trader/` | ❌ | ❌ | ✅ | ❌ |

## Quick Start

### 1. Install Dependencies
```bash
cd apps/live_trader
pip install -r requirements.txt

cd ../backtester
pip install -r requirements.txt
```

### 2. Run Live Trader
```bash
cd apps/live_trader
streamlit run app.py
```

### 3. Run Backtester
```bash
cd apps/backtester
streamlit run app.py
```

## Data Flow

```
Training Pipeline (scripts/)
    │
    ├──► Trained Models ──► artifacts/ppo_*/
    │                           ├── model.zip
    │                           ├── scaler.pkl
    │                           ├── metadata.json
    │                           ├── backtest_ledger.csv
    │                           └── reports/
    │
    │                        ▲
    │                        │ reads
    │                        │
    ├──► Live Trader ◄───────┘
    │      • Real-time trading
    │      • Model inference
    │      • Order execution
    │
    └──► Backtester
           • Historical analysis
           • Model comparison
           • Performance metrics
```

## Architecture Documents

- [Live Trader Architecture](live_trader/ARCHITECTURE.md)
- [Backtester Architecture](backtester/ARCHITECTURE.md)

## Development

### Adding Shared Components

Place reusable components in `apps/shared/`:

```python
# apps/shared/components/price_chart.py
import streamlit as st
import plotly.graph_objects as go

def render_price_chart(df, title="Price Chart"):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=title)
    st.plotly_chart(fig)
```

Use in apps:
```python
import sys
sys.path.append('../shared')
from components.price_chart import render_price_chart
```

### Model Discovery

Both apps use the same logic to find trained models:

```python
from shared.utils.model_discovery import list_available_models

models = list_available_models()  # Returns list of artifact folders
```

## Environment Variables

Create `.env` files in each app directory:

**live_trader/.env:**
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true
TELEGRAM_TOKEN=optional
TELEGRAM_CHAT_ID=optional
```

**backtester/.env:**
```
# No secrets needed for backtester - reads local files only
```
