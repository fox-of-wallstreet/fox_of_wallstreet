# Multi-Stock Support Documentation

## Current State: ✅ FIXED - Multi-Stock Supported!

### The Fix (Applied)

The app now reads the symbol from model metadata:

```python
# In pages/01_trade.py
trading_symbol = model_info.get("symbol", settings.SYMBOL)
price_data = get_price_data(trading_symbol, settings.TIMEFRAME)
```

When you load a model, the symbol is extracted from `metadata.json` and used for price fetching.

### Old Behavior (Before Fix)

```python
# config/settings.py (hardcoded)
SYMBOL = "TSLA"  # ← Always TSLA

# apps/live_trader/pages/01_trade.py
price_data = get_price_data(settings.SYMBOL, settings.TIMEFRAME)
#                                    ↑ Always fetches TSLA!
```

**The Issue:**
- You load a model trained on AAPL
- App still fetches TSLA prices from yfinance
- AI makes decisions based on TSLA data
- This is WRONG!

### New Behavior (After Fix)

```python
# apps/live_trader/pages/01_trade.py
model_info = st.session_state.get("model_info", {})
trading_symbol = model_info.get("symbol", settings.SYMBOL)  # ← From model!

price_data = get_price_data(trading_symbol, settings.TIMEFRAME)
#                                    ↑ Uses model's symbol (AAPL, TSLA, etc.)
```

**How it works:**
1. Load model → Extract symbol from metadata.json
2. Fetch prices for THAT symbol
3. AI makes decisions on correct data

### What the Model Knows

Each model has metadata with the symbol:

```json
// artifacts/ppo_AAPL_1h_.../metadata.json
{
    "symbol": "AAPL",
    "timeframe": "1h",
    "action_space": "discrete_5",
    ...
}
```

**The app reads this but doesn't use it for price fetching!**

---

## The Fix: Read Symbol from Model

### Option 1: Quick Fix (Recommended)

Change `pages/01_trade.py` to use model's symbol:

```python
# Current (WRONG):
price_data = get_price_data(settings.SYMBOL, settings.TIMEFRAME)

# Fixed (CORRECT):
model_info = st.session_state.get("model_info", {})
symbol = model_info.get("symbol", settings.SYMBOL)  # Fallback to settings
price_data = get_price_data(symbol, settings.TIMEFRAME)
```

### Option 2: Full Multi-Stock Support

Track symbol in session state:

```python
# When loading model:
st.session_state["active_symbol"] = model_dict["metadata"]["symbol"]

# When fetching prices:
symbol = st.session_state.get("active_symbol", settings.SYMBOL)
```

---

## Does the App Support Other Stocks?

### Short Answer: YES, with code changes

**What works:**
- ✅ Loading models trained on other stocks
- ✅ Running inference (same code works)
- ✅ Feature computation (same functions)

**What doesn't work (currently):**
- ❌ Automatic price fetching for correct symbol
- ❌ Symbol display in UI
- ❌ Compatibility check for symbol mismatch

---

## Verification Script

Test if you're fetching correct prices:

```python
# test_symbol_matching.py
import sys
sys.path.append('.')

from apps.live_trader.utils.feature_fetcher import fetch_recent_prices
from config import settings

# 1. Check settings
print(f"Settings SYMBOL: {settings.SYMBOL}")

# 2. Check loaded model
import streamlit as st
model_info = st.session_state.get("model_info", {})
print(f"Model symbol: {model_info.get('symbol', 'Not loaded')}")

# 3. Check what we're actually fetching
# (This would be in the actual app)
# price_df = fetch_recent_prices(settings.SYMBOL, settings.TIMEFRAME)
# print(f"Fetched prices for: {settings.SYMBOL}")
```

---

## FAQ

### Q: I loaded an AAPL model but it's showing TSLA prices?

**A:** Yes, that's the current bug. The app always uses `settings.SYMBOL` instead of the model's symbol.

**Fix:** Update `pages/01_trade.py` to read symbol from model metadata.

### Q: Can I trade multiple stocks at once?

**A:** Not in the current UI. You'd need to:
1. Load AAPL model
2. Trade AAPL
3. Switch to TSLA model
4. Trade TSLA

For true multi-stock, you'd need:
- Multiple model instances
- Portfolio tracking per symbol
- More complex UI

### Q: Should I change settings.SYMBOL when loading different models?

**A:** As a workaround, YES:

```python
# config/settings.py
# Change this before loading model:
SYMBOL = "AAPL"  # or "MSFT", etc.
```

Then restart the app.

### Q: Will the model work with wrong symbol data?

**A:** NO - it will make bad decisions:

- Model trained on AAPL volatility patterns
- Sees TSLA price movements
- Predictions will be wrong
- Could lose money!

---

## Recommended Fix

Add this to `pages/01_trade.py` right after loading model check:

```python
# After line ~67 where we get model_info
model_info = st.session_state.get("model_info", {})

# Get symbol from model, fallback to settings
trading_symbol = model_info.get("symbol", settings.SYMBOL)

# Show warning if different
if trading_symbol != settings.SYMBOL:
    st.warning(f"⚠️ Model trained on {trading_symbol}, but settings has {settings.SYMBOL}. "
               f"Using {trading_symbol} for price fetching.")

# Use trading_symbol everywhere instead of settings.SYMBOL
price_data = get_price_data(trading_symbol, settings.TIMEFRAME)
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does app auto-fetch correct symbol? | ✅ YES - reads from model metadata |
| Can it support other stocks? | ✅ YES - automatic |
| Need to change settings? | ❌ NO - just load the model |
| Is it dangerous? | ❌ NO - fixed |

---

## How to Use Multi-Stock

1. **Train models on different stocks:**
   ```python
   # config/settings.py
   SYMBOL = "AAPL"
   # Then: python scripts/train.py
   
   SYMBOL = "TSLA"  
   # Then: python scripts/train.py
   ```

2. **Switch between them in the app:**
   - Go to Models page
   - Load AAPL model → App fetches AAPL prices
   - Load TSLA model → App fetches TSLA prices

3. **The app handles the rest automatically!**

---

## Action Items (All Done!)

- [x] Read symbol from model metadata
- [x] Use model symbol for price fetching
- [x] Show warning if model/settings mismatch
- [x] Update portfolio tracking per symbol
