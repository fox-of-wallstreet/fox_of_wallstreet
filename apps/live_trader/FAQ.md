# Live Trader FAQ

## Quick Answers to Common Questions

### Refresh Intervals by Mode - What's the Difference?

**Each mode has different defaults (industry best practices):**

| Mode | Default | Range | Why |
|------|---------|-------|-----|
| **🔍 Simulate** | 30s | 10-300s | Fast for testing/demo, no risk |
| **🛡️ Secure** | 60s | 30-600s | Balanced - you confirm each trade |
| **🤖 Autopilot** | 300s (5min) | 60-3600s | Conservative - real money at stake |

**Why 5 minutes for Autopilot?**
1. **Rate limits:** Alpaca limits API calls (200/min free tier)
2. **Slippage protection:** Prices don't change much in 5 min
3. **Overtrading prevention:** Reduces fees and emotional decisions
4. **Market hours:** Only trade during market hours anyway

**You can adjust in the sidebar when auto-refresh is enabled.**

---

### How do I deploy for demo access from another computer?

**Option 1: Streamlit Cloud (Easiest, Free)**
```bash
# 1. Push to GitHub (make sure .env is gitignored!)
git add .
git commit -m "Demo version"
git push origin main

# 2. Go to streamlit.io/cloud
# 3. Connect your GitHub repo
# 4. Add secrets in dashboard:
#    Settings → Secrets → Add ALPACA_API_KEY, ALPACA_SECRET_KEY
# 5. Share the URL!
```

**Option 2: Private Server (More control)**
```bash
# VPS (DigitalOcean, AWS, Linode $5-10/month)
# 1. SSH into server
# 2. Clone repo, install dependencies
# 3. Set env vars securely
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret

# 4. Run with --server.address=0.0.0.0
streamlit run apps/live_trader/app.py --server.address=0.0.0.0 --server.port=8501
```

**See full guide:** [DEPLOYMENT.md](DEPLOYMENT.md)

---

### Are my API keys safe in deployment?

**YES, if you follow these rules:**

✅ **DO:**
- Store keys in environment variables (never in code)
- Use Streamlit Cloud Secrets or Heroku Config Vars
- Use `.env` file for local development (gitignored!)
- Enable Paper Trading for demos

❌ **DON'T:**
- Hardcode API keys in Python files
- Commit `.env` to GitHub
- Share screenshots with visible keys
- Use Live trading for public demos

**How it works in Streamlit Cloud:**
```
Your code → GitHub → Streamlit Cloud
     ↑                           ↓
  No secrets               Secrets injected
                           at runtime (encrypted)
```

**See:** [DEPLOYMENT.md](DEPLOYMENT.md) for full security guide.

---

### Where is the Trading Mode Selector?

**It's in the LEFT SIDEBAR** under "🎮 Trading Mode":

```
┌─────────────────────┐
│ 🦊 Fox of Wallstreet │
│ Live AI Trader      │
│                     │
│ 🚀 Trade            │
│ 📊 Trade Dashboard  │
│ 🧠 Models           │
│ 📜 History          │
│ ⚙️ Settings         │
├─────────────────────┤
│ 🎮 TRADING MODE     │ ← HERE
│ ○ 🔍 Simulate       │
│ ○ 🛡️ Secure         │
│ ○ 🤖 Autopilot      │
│                     │
│ Current: SIMULATE   │
└─────────────────────┘
```

**Three modes:**
| Mode | What it does | Telegram? |
|------|--------------|-----------|
| **🔍 Simulate** | Virtual portfolio, no real orders | ❌ No |
| **🛡️ Secure** | Real orders, requires confirmation | ✅ Yes |
| **🤖 Autopilot** | Real orders, auto-executes | ✅ Yes |

---

### Where do I configure API Keys?

**Option 1: Settings Page (Recommended)**
1. Go to **⚙️ Settings** page
2. Enter Alpaca API Key & Secret
3. Enter Telegram Token & Chat ID (optional)
4. Click "💾 Save"
5. **Restart the app** to apply changes

**Option 2: .env File**
Edit `.env` in project root:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true

# Optional: Telegram notifications
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

### When do Telegram notifications trigger?

**Telegram ONLY sends notifications when:**
1. ✅ You have `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` set
2. ✅ You're in **SECURE** or **AUTOPILOT** mode (not SIMULATE)
3. ✅ An order is actually submitted to Alpaca

**Telegram does NOT send for:**
- ❌ Simulate mode trades
- ❌ HOLD decisions
- ❌ Blocked trades (price verification failed)

**Setup Telegram:**
1. Message [@BotFather](https://t.me/botfather) → `/newbot` → copy token
2. Message [@userinfobot](https://t.me/userinfobot) → copy your Chat ID
3. Go to Settings page → save both values

---

## Q1: Is the data pipeline the same as training?

**A: YES** - We use the exact same functions:

| Component | Training | Live Trading |
|-----------|----------|--------------|
| Price source | yfinance | yfinance ✅ |
| Feature computation | `add_technical_indicators()` | `add_technical_indicators()` ✅ |
| AVWAP calculation | `core/avwap.py` | `core/avwap.py` ✅ |
| Feature registry | `FEATURE_REGISTRY` | `FEATURE_REGISTRY` ✅ |
| Scaling | `RobustScaler.fit()` | `RobustScaler.transform()` ✅ |

**One difference:** News features are set to 0 in live trading (no real-time news API yet).

See [DATA_INTEGRITY.md](DATA_INTEGRITY.md) for detailed proof.

---

## Q2: How do I simulate autonomous mode?

**A: Currently it's manual, but you have options:**

### Quick Fix (Add to trade dashboard)
```python
# In pages/01_trade.py, add at the top:
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds
count = st_autorefresh(interval=60 * 1000, limit=None, key="autorefresh")
```

### Better: Use the Production Bot
```bash
# This runs truly autonomously (not Streamlit)
python scripts/live_trader.py --bot
```

**See:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md) "Simulating Autonomous Mode" section

---

## Q3: Are the prices real?

**A: YES - Real market data:**

- ✅ Source: Yahoo Finance (yfinance)
- ✅ Real TSLA prices
- ✅ Adjusted for splits/dividends
- ⚠️ Delayed ~15-20 minutes (free tier)
- ⚠️ Hourly or daily bars (not tick-level)

**The chart shows actual closing prices** from the market.

---

## Q4: Can I use models trained on other stocks?

**A: NO (by design)**

A TSLA model should NOT trade AAPL because:
- Different volatility (TSLA beta ~2.0 vs AAPL ~1.2)
- Different price ranges
- Scaler fitted on TSLA-specific distributions

**The app warns you if settings don't match:**
```
⚠️ Model may not be compatible with current settings:
  - symbol: trained=TSLA | current=AAPL
```

**To trade another stock:**
1. Change `SYMBOL` in `config/settings.py`
2. Retrain: `python scripts/train.py`
3. Load new model in app

---

## Q5: Does it support daily models?

**A: YES** - Automatically detected from model metadata:

```json
// metadata.json
{
  "timeframe": "1d",  // or "1h"
  "action_space": "discrete_5"
}
```

The app reads this and fetches daily prices instead of hourly.

---

## Q6: Why is confidence 50%?

**A: 50% is the minimum** - it means the AI is essentially random/uncertain.

Possible reasons:
- Model not confident in current market conditions
- Features are unusual (out of training distribution)
- First inference (policy entropy calculation)

**Normal range:** 50-95%. Above 80% is high confidence.

---

## Q7: Simulate mode uses my Alpaca balance?

**A: NO** - Simulate mode is completely isolated:

```python
# Virtual portfolio (starts fresh)
portfolio = {
    "cash": 10000.0,      # From INITIAL_BALANCE, not Alpaca
    "position": 0.0,      # Virtual shares
    "entry_price": 0.0,   # Virtual entry
}
```

**Simulate mode never connects to Alpaca.**

---

## Q8: SELL_50 means 50% of what?

**A: 50% of your POSITION (shares), not balance.**

```
You own: 100 shares
SELL_50 → Sell 50 shares
You keep: 50 shares
```

NOT: "Sell $5,000 worth"

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) "Understanding SELL_50 and BUY_50"

---

## Q9: Why news features are 0?

**A: No real-time news API integrated yet.**

Training uses historical Alpaca news + FinBERT scoring.
Live trading would need:
- Real-time news feed
- Low-latency FinBERT inference

**Workaround:** Train without news:
```python
# config/settings.py
USE_NEWS_FEATURES = False
```

---

## Q10: Are yfinance and Alpaca prices the same?

**A: MOSTLY YES, but there can be small differences.**

### Why Prices Might Differ

| Source | Data Provider | Delay | Potential Difference |
|--------|---------------|-------|---------------------|
| **yfinance** | Yahoo Finance (via exchanges) | ~15-20 min | OHLC aggregation |
| **Alpaca** | Direct exchange feeds | Real-time (paid) / 15 min (free) | Bid/ask spread, aggregation |

### Typical Differences
- **Closing prices**: Usually match within $0.01-$0.05
- **During volatile periods**: Can differ $0.10-$0.50
- **Opening prices**: May differ due to pre-market data

### The Risk

```
Scenario:
1. AI sees yfinance price: $250.00
2. AI recommends: BUY_50
3. You click execute
4. Alpaca executes at: $250.25 (0.1% higher)
5. Your expected position is slightly off
```

### Solutions

**Option 1: Accept small slippage (current)**
- Document the risk
- Use `SLIPPAGE_PCT` in settings (already exists: 0.0005 = 0.05%)

**Option 2: Use Alpaca for both data and trading (recommended)**
```python
# Replace yfinance with Alpaca market data
from alpaca.data.historical import StockHistoricalDataClient

# Get price from same source you trade with
```

**Option 3: Verify before execute**
```python
# In execute handler:
alpaca_price = get_alpaca_price()  # Real-time
yfinance_price = st.session_state["last_ai_decision"]["latest_price"]

diff_pct = abs(alpaca_price - yfinance_price) / yfinance_price
if diff_pct > 0.001:  # 0.1% difference
    st.warning(f"Price changed {diff_pct:.2%} since analysis!")
```

### Recommendation

For **demo/development**: yfinance is fine (free, no API keys)

For **production paper trading**: Switch to Alpaca market data

```bash
# Install alpaca data
pip install alpaca-py

# Use in feature_fetcher.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
```

See DATA_INTEGRITY.md for implementation details.

---

## Q11: Do we have Alpaca integration?

**A: YES - Fully implemented!**

### What's Working Now
- ✅ Order submission to Alpaca (paper or live)
- ✅ Portfolio sync from Alpaca
- ✅ Price verification before trades
- ✅ Secure mode with confirmation
- ✅ Autopilot mode (auto-execute)

### How It Works

**SECURE Mode:**
1. Syncs portfolio from Alpaca (real cash/position)
2. Verifies price (yfinance vs Alpaca)
3. Shows confirmation checkbox
4. Submits order when confirmed

**AUTOPILOT Mode:**
1-2. Same as above
3. Auto-submits order (no confirmation)

### Setup Required

**Option 1: Environment Variables (Recommended)**
Add to `.env` file in project root:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true  # Use paper trading (set to false for live)
```

**Option 2: Settings Page**
Enter keys directly in the app's Settings page

**Then:**
1. Select SECURE or AUTOPILOT mode
2. Set `LIVE_TRADING_BUDGET` in settings.py (caps your risk)

### Price Verification
All real trades verify price before executing:
- Compares yfinance price (AI analysis) with Alpaca price (execution)
- If difference > 0.1% → Trade blocked
- Protects against slippage and stale data

See TRADING_MECHANICS.md for full details.

---

## Q10: Can I paper trade without confirmation?

**A: YES - Use AUTOPILOT mode:**

1. Set mode to "Autopilot" in sidebar
2. Enable auto-refresh (see Q2)
3. AI will execute without confirmation

**⚠️ Warning:** Only use after extensive testing in SECURE mode!

---

## Quick Reference

| Question | Answer | File |
|----------|--------|------|
| Same pipeline as training? | ✅ YES | DATA_INTEGRITY.md |
| Real prices? | ✅ YES (delayed 15-20min) | DATA_INTEGRITY.md |
| Autonomous mode? | ⚠️ Manual now, options available | HOW_IT_WORKS.md |
| Other stocks? | ❌ NO (train separate models) | HOW_IT_WORKS.md |
| Daily timeframe? | ✅ YES (auto-detected) | HOW_IT_WORKS.md |
| News features? | ⚠️ Set to 0 (not implemented) | DATA_INTEGRITY.md |

---

## Getting Help

1. Check [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for trading logic
2. Check [DATA_INTEGRITY.md](DATA_INTEGRITY.md) for pipeline details
3. Run the verification script in DATA_INTEGRITY.md
4. Open an issue with screenshots and logs
