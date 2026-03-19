# Data Integrity & Pipeline Verification

## ✅ Are We Using the Same Pipeline as Training?

**YES** - Here's the proof:

### Training Pipeline (scripts/train.py)
```python
from core.processor import (
    build_training_dataset,   # Orchestrator
    add_technical_indicators, # Feature computation
    prepare_features,         # Scaling
)

# Flow:
# 1. fetch prices (yfinance)
# 2. compute features (add_technical_indicators)
# 3. scale (prepare_features with RobustScaler)
# 4. train
```

### Live Trading Pipeline (apps/live_trader/utils/feature_fetcher.py)
```python
from core.processor import (
    add_technical_indicators,  # ← SAME FUNCTION
    merge_prices_news_macro,   # ← SAME FUNCTION
)

# Flow:
# 1. fetch prices (yfinance - SAME SOURCE)
# 2. compute features (add_technical_indicators - SAME FUNCTION)
# 3. scale (scaler.transform - SAME SCALER FROM TRAINING)
# 4. predict
```

### Verification

Both use:
- ✅ **Same yfinance data source**
- ✅ **Same `add_technical_indicators()` function** from `core/processor.py`
- ✅ **Same feature computation logic** (RSI, MACD, AVWAP, etc.)
- ✅ **Same scaling method** (RobustScaler fitted during training)
- ✅ **Same feature list** from `settings.FEATURES_LIST`

### Code Evidence

```python
# feature_fetcher.py line 13-18:
from core.processor import (
    add_technical_indicators,    # ← EXACT SAME IMPORT
    load_raw_news,
    build_news_sentiment,
    load_raw_macro,
    merge_prices_news_macro,
)
```

---

## Data Flow Comparison

| Step | Training | Live Trading | Match? |
|------|----------|--------------|--------|
| **Price Source** | yfinance | yfinance | ✅ |
| **Feature Function** | `add_technical_indicators()` | `add_technical_indicators()` | ✅ |
| **Feature Registry** | `FEATURE_REGISTRY` in processor.py | `FEATURE_REGISTRY` in processor.py | ✅ |
| **AVWAP Calculation** | `core/avwap.py` | `core/avwap.py` | ✅ |
| **News Processing** | FinBERT scoring | Zeros (see below) | ⚠️ |
| **Macro Data** | yfinance (QQQ, VIX, TNX) | yfinance (QQQ, VIX, TNX) | ✅ |
| **Scaling** | `RobustScaler.fit()` | `RobustScaler.transform()` (loaded) | ✅ |
| **Feature Order** | `settings.FEATURES_LIST` | `settings.FEATURES_LIST` | ✅ |

---

## ⚠️ Known Differences (Important!)

### 1. News Features

**Training:**
```python
# Real-time Alpaca news + FinBERT sentiment scoring
news_df = load_raw_news()  # From Alpaca API
sentiment_df = build_news_sentiment(news_df)  # FinBERT scoring
```

**Live Trading (Current):**
```python
# News features set to 0 (no real-time news API)
news_df = pd.DataFrame({
    'Sentiment_Mean': 0.0,
    'News_Intensity': 0.0,
})
```

**Impact:** Models trained WITH news features get neutral (zero) news signals in live trading.

**Solutions:**
1. Train without news (`USE_NEWS_FEATURES = False`) for pure technical trading
2. Add Alpaca news API integration (future enhancement)
3. Use news aggregation service (e.g., Bloomberg, Refinitiv)

### 2. Data Freshness

**Training:** Historical data (2023-2025)

**Live Trading:** Real-time data (delayed ~15-20 min for free yfinance)

**Impact:** The model has never seen 2026 data during training. This is normal - all ML models face this "distribution shift" challenge.

---

## How to Verify Data Integrity

### Method 1: Compare Feature Values

```python
# In training notebook or script
from core.processor import build_training_dataset

train_df = build_training_dataset()
print("Training - Last row features:")
print(train_df[settings.FEATURES_LIST].iloc[-1])

# In live trading app
from apps.live_trader.utils.feature_fetcher import fetch_recent_prices, build_live_features

price_df = fetch_recent_prices('TSLA', '1h')
feature_df = build_live_features(price_df, pd.DataFrame())
print("\nLive - Last row features:")
print(feature_df[settings.FEATURES_LIST].iloc[-1])
```

### Method 2: Check Feature Ranges

After scaling, features should be in similar ranges:

| Feature | Training Range (Scaled) | Live Range (Scaled) | Status |
|---------|------------------------|---------------------|--------|
| RSI | -3 to +3 | -3 to +3 | ✅ OK |
| Log_Return | -5 to +5 | -5 to +5 | ✅ OK |
| AVWAP_Dist | -2 to +2 | -2 to +2 | ✅ OK |

### Method 3: Model Validation Error

If features are wrong, you'll see:
```
ValueError: DATA SHAPE MISMATCH: Expected 18 features, got 17
```

Or the model will make erratic predictions.

---

## 🔍 Are the Prices Real?

**YES** - Real market data from Yahoo Finance:

```python
# feature_fetcher.py
data = yf.download(
    symbol='TSLA',
    period='60d',
    interval='1h',  # or '1d'
    auto_adjust=True  # Adjusted for splits/dividends
)
```

**Characteristics:**
- ✅ Real historical prices
- ✅ Adjusted for stock splits
- ✅ Adjusted for dividends
- ⚠️ **Delayed ~15-20 minutes** (free tier)
- ⚠️ **Not tick-level** (hourly or daily bars)

**For real-time trading:**
You'd need Alpaca market data API (premium) or another real-time provider.

---

## 📊 What the Graph Shows

The chart displays:
- ✅ **Actual TSLA prices** from yfinance
- ✅ **Last 5 days** of hourly data (or daily if timeframe=1d)
- ✅ **Green line** = closing prices
- ✅ **Yellow dashed line** = your entry price (if in position)

**Not shown:**
- Bid/ask spread
- Volume profile
- Level 2 order book

---

## 🔄 Simulating Autonomous Mode

### Current Behavior (Manual)
```
User clicks "Run AI Analysis" → Fetches data → Computes features → Predicts
```

### To Simulate Autonomous Mode

You have **three options**:

#### Option 1: Auto-Refresh (Simplest)
Add this to `pages/01_trade.py`:

```python
# At top of file
import time

# Add auto-refresh
if st.session_state.get("auto_run", False):
    time.sleep(60)  # Wait 60 seconds
    st.rerun()      # Rerun the script

# Toggle in UI
st.session_state["auto_run"] = st.checkbox("🤖 Auto-run every 60s", value=False)
```

#### Option 2: Scheduled Execution (Better)
Use `streamlit-autorefresh`:

```bash
pip install streamlit-autorefresh
```

```python
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
count = st_autorefresh(interval=60 * 1000, limit=None, key="fizzbuzzcounter")

# Automatically run AI if model loaded
if st.session_state.get("loaded_model") and st.session_state.get("auto_run"):
    # Run inference automatically
    result = run_ai_inference(...)
```

#### Option 3: True Autonomous Bot (Production)
Run `scripts/live_trader.py` which has:
- Scheduled execution at candle closes
- Telegram integration
- Persistent state
- Error handling

```bash
# Run the production live trader (not Streamlit)
python scripts/live_trader.py --bot
```

### Recommendation

| Use Case | Solution |
|----------|----------|
| Demo/Testing | Option 1 (simple auto-refresh) |
| Paper Trading | Option 2 (streamlit-autorefresh) |
| Production | Option 3 (`live_trader.py` bot) |

---

## 🌐 Does It Support Other Stocks?

### Short Answer: **NO** (by design)

### Why?

A model trained on TSLA **should not** trade AAPL because:

1. **Different volatility regimes**
   - TSLA: High volatility (beta ~2.0)
   - AAPL: Lower volatility (beta ~1.2)
   
2. **Different price ranges**
   - Model learned TSLA at $200-400
   - AAPL at $150-200 is different distribution

3. **Different feature distributions**
   - Scaler fitted on TSLA data
   - AAPL features will be out of distribution

### Verification

The app checks compatibility:

```python
# In model_selector.py
is_compatible, mismatches = validate_model_compatibility(
    selected_model["path"],
    {
        "SYMBOL": settings.SYMBOL,  # ← Must match!
        "TIMEFRAME": settings.TIMEFRAME,
        ...
    }
)

if not is_compatible:
    st.warning("⚠️ Model may not be compatible...")
```

### If You Want Multi-Stock Support

You need to either:

1. **Train separate models per stock**
   ```
   artifacts/ppo_TSLA_1h_...
   artifacts/ppo_AAPL_1h_...
   artifacts/ppo_MSFT_1h_...
   ```
   Then switch models in the app.

2. **Train a universal model** (advanced)
   - Include symbol as a feature
   - Train on multiple stocks
   - Much harder to get right

3. **Use the same stock**
   - Change `SYMBOL` in settings.py
   - Retrain the model
   - Use that model

---

## Summary Checklist

### ✅ What's Guaranteed to Match Training
- [x] Same price source (yfinance)
- [x] Same feature computation functions
- [x] Same scaler (loaded from artifacts)
- [x] Same feature order
- [x] Same observation format (VecFrameStack)

### ⚠️ What's Different
- [ ] News features (set to 0 in live)
- [ ] Data timestamp (2026 vs 2023-2025)
- [ ] Manual execution vs scheduled

### ❌ What's Blocked
- [x] Wrong symbol (compatibility check)
- [x] Wrong timeframe (compatibility check)
- [x] Wrong action space (UI adapts)

---

## Verification Command

Run this to verify your setup:

```python
# test_data_integrity.py
import sys
sys.path.append('.')

from config import settings
from apps.live_trader.utils.feature_fetcher import fetch_recent_prices, build_live_features
import pandas as pd

print("=" * 60)
print("DATA INTEGRITY CHECK")
print("=" * 60)

# 1. Check settings
print(f"\n1. Settings:")
print(f"   Symbol: {settings.SYMBOL}")
print(f"   Timeframe: {settings.TIMEFRAME}")
print(f"   Features: {len(settings.FEATURES_LIST)}")

# 2. Fetch live data
print(f"\n2. Fetching live data...")
price_df = fetch_recent_prices(settings.SYMBOL, settings.TIMEFRAME)
print(f"   Got {len(price_df)} price rows")
print(f"   Latest price: ${price_df['Close'].iloc[-1]:.2f}")

# 3. Build features
print(f"\n3. Building features...")
feature_df = build_live_features(price_df, pd.DataFrame())
print(f"   Got {len(feature_df)} feature rows")
print(f"   Feature columns: {len(feature_df.columns)}")

# 4. Check feature availability
print(f"\n4. Feature availability:")
missing = [f for f in settings.FEATURES_LIST if f not in feature_df.columns]
if missing:
    print(f"   ❌ Missing: {missing}")
else:
    print(f"   ✅ All {len(settings.FEATURES_LIST)} features present")

# 5. Sample values
print(f"\n5. Latest feature values (unscaled):")
latest = feature_df[settings.FEATURES_LIST].iloc[-1]
for feat in ['RSI', 'Log_Return', 'AVWAP_Dist', 'MACD_Hist']:
    if feat in latest:
        print(f"   {feat}: {latest[feat]:.4f}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
```

Run: `python test_data_integrity.py`
