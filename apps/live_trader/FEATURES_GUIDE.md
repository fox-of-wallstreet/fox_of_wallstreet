# Live Trader Features Guide

Complete guide to all features and how to use them.

---

## 📊 Main Dashboard Features

### 1. AI Decision Card

**What it shows:**
- AI's recommended action (BUY_50, SELL_100, HOLD, etc.)
- Confidence score (0-100%)
- Current stock price
- Key signals driving the decision (RSI, AVWAP, etc.)

**How to interpret:**
- **Green card** = BUY recommendation
- **Red card** = SELL recommendation  
- **Yellow/Gray card** = HOLD recommendation
- **Confidence > 80%** = AI is very certain
- **Confidence < 60%** = AI is uncertain, be cautious

**How to use:**
1. Click "Run AI Analysis"
2. Review the recommendation
3. Check confidence level
4. Look at key signals to understand why
5. Decide whether to execute or override

---

### 2. Position Sizing Preview

**What it shows:**
- Investment amount ($)
- Shares to trade
- New position size after trade
- Remaining cash
- Risk percentage of portfolio

**How to interpret:**
- **Investment**: How much cash will be used
- **Shares**: Exact number of shares bought/sold
- **Risk %**: What portion of portfolio is at risk
- **New Position**: Your holdings after the trade

**How to use:**
- Appears automatically after AI analysis
- Review before executing
- Check "Risk %" to ensure it's within your limits
- Use manual calculator (below) to plan hypothetical trades

---

### 3. P&L Tracker (Like Trading 212)

**What it shows:**

#### Unrealized P&L
- Paper profit/loss on open position
- Changes with every price tick
- Becomes "Realized" when you sell

#### Realized P&L  
- Actual cash profit from closed trades
- Locked in when you sell
- Cumulative total of all completed trades

#### Trading Statistics
- **Win Rate**: % of trades that were profitable
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Avg Win/Loss Ratio**: How big winners are vs losers

**How to interpret:**
- **🟢 Green** = Profit
- **🔴 Red** = Loss
- **Total Return** = Realized + Unrealized
- **Win Rate > 50%** = More wins than losses (good)
- **Win/Loss Ratio > 1.5** = Winners bigger than losers (good)

**How to use:**
- Monitor Unrealized P&L to decide when to take profits
- Review Realized P&L to track overall performance
- Check Win Rate to evaluate strategy effectiveness
- Look at trading statistics to identify patterns

---

### 4. Position Size Calculator

**What it is:**
- Manual calculator for planning trades
- Independent of AI recommendations
- Helps you understand position sizing

**How to use:**
1. Click "Manual Calculator" to expand
2. Enter your account size
3. Enter current stock price
4. Select risk percentage
5. Select action (Buy/Sell 50% or 100%)
6. Click "Calculate"
7. See projected trade details

**Use cases:**
- Plan trades before loading a model
- Understand position sizing concepts
- Calculate hypothetical scenarios
- Educational tool for learning

---

### 5. Trading Modes

#### 🔍 SIMULATE Mode

**What it does:**
- No real money at risk
- Virtual portfolio updates locally
- No orders sent to Alpaca
- Perfect for testing

**How to use:**
1. Select "SIMULATE" in sidebar
2. Run AI Analysis
3. Click execute buttons
4. Watch virtual portfolio update
5. Learn without risk

**Best for:**
- Testing AI behavior
- Learning the interface
- Demo purposes

---

#### 🛡️ SECURE Mode

**What it does:**
- Syncs with real Alpaca account
- Verifies prices before trading
- Shows confirmation checkbox
- Submits order only after you confirm

**How it works:**
```
1. Sync portfolio from Alpaca
2. Verify price (yfinance vs Alpaca)
3. IF price mismatch > 0.1%:
     → BLOCK trade
   ELSE:
     → Show confirmation
4. You check box to confirm
5. Order submitted to Alpaca
```

**How to use:**
1. Add Alpaca API keys in Settings
2. Select "SECURE" mode
3. Run AI Analysis
4. Review Position Sizing Preview
5. Click execute
6. Check confirmation box
7. Order submitted

**Best for:**
- Paper trading with real broker
- Learning with real prices
- Safe automation

---

#### 🤖 AUTOPILOT Mode

**What it does:**
- Same as SECURE but auto-executes
- No confirmation checkbox
- Can combine with auto-refresh
- Fully automated trading

**How it works:**
```
1. Sync portfolio from Alpaca
2. Verify price
3. IF price OK:
     → Auto-submit order
   ELSE:
     → Block and wait
```

**⚠️ DANGER:** No confirmation step! Only use after extensive testing.

**How to use:**
1. Extensive testing in SECURE mode first
2. Add Alpaca API keys
3. Select "AUTOPILOT" mode
4. (Optional) Enable auto-refresh
5. AI will trade automatically

**Best for:**
- Production trading (after months of testing)
- Hands-off automation
- Experienced users only

---

### 6. Auto-Refresh / Autonomous Trading

**What it does:**
- Automatically runs AI analysis on schedule
- Fetches fresh data
- Updates recommendations
- Can auto-execute (in AUTOPILOT mode)

**How to use:**
1. Look for "🤖 Auto-Trading" in sidebar
2. Check "Enable Auto-Refresh"
3. Set interval (30-300 seconds)
4. App will refresh automatically
5. Shows "🤖 Auto-run triggered" badge

**Recommended intervals:**
| Mode | Interval | Why |
|------|----------|-----|
| SIMULATE | 60s | Fast feedback |
| SECURE | 300-900s | Time to review |
| AUTOPILOT | Match timeframe | 3600s for hourly, 86400s for daily |

---

### 7. Price Verification

**What it is:**
- Compares yfinance price (AI analysis) with Alpaca price (execution)
- Protects against slippage
- Blocks trades if prices differ too much

**How it works:**
```python
yfinance_price = $250.00  # What AI saw
alpaca_price = $250.30    # Current market price
difference = 0.12%

IF difference > 0.1%:
    BLOCK trade
ELSE:
    ALLOW trade
```

**When it runs:**
- ✅ SECURE mode: Yes
- ✅ AUTOPILOT mode: Yes
- ❌ SIMULATE mode: No (uses yfinance directly)

**What causes failures:**
- yfinance delay (15-20 min behind)
- Fast market moves
- After-hours trading

**What to do if blocked:**
1. Wait 60 seconds for prices to converge
2. Switch to SIMULATE mode to see what AI would do
3. Check market conditions
4. Try again

---

### 8. Activity Log

**What it shows:**
- Timestamped history of all events
- AI analysis results
- Trade executions
- Errors and warnings

**How to use:**
- Review recent activity
- Track what the AI has been doing
- Debug issues
- Export for analysis

---

## 📈 Components Explained

### Model Loading

**How it works:**
1. App scans `artifacts/` folder
2. Lists all trained models
3. Shows metadata (symbol, timeframe, performance)
4. You select and load
5. App reads symbol from model metadata
6. Automatically fetches correct prices

**Multi-stock support:**
- Load AAPL model → Fetches AAPL prices
- Load TSLA model → Fetches TSLA prices
- No need to change settings!

---

### Portfolio Tracking

**What it tracks:**
- Cash available
- Position (shares held)
- Entry price (average cost)
- Position value
- Portfolio value (cash + position)

**Where data comes from:**
- SIMULATE: Local session state
- SECURE/AUTOPILOT: Synced from Alpaca

---

### Confidence Score

**What it means:**
- Measures how certain the AI is about its decision
- Based on policy probability distribution
- 0% = Random guessing
- 100% = Absolute certainty

**Typical ranges:**
- **80-100%** = Very confident, strong signal
- **60-80%** = Moderate confidence
- **50-60%** = Uncertain, weak signal
- **<50%** = Very uncertain (rare)

**How it's calculated:**
```python
# PPO outputs probabilities for each action
probs = [0.1, 0.1, 0.6, 0.1, 0.1]  # 60% for action 2

# Compare to random guessing (20% for 5 actions)
confidence = (0.6 - 0.2) / (1 - 0.2) * 100 = 50%
```

---

## 🎯 Quick Start Workflows

### Workflow 1: Demo / Learning

```
1. Set Trading Mode: SIMULATE
2. Click "Run AI Analysis"
3. Review Position Sizing Preview
4. Click execute button
5. Watch virtual portfolio update
6. Check P&L Tracker
7. Repeat and learn!
```

### Workflow 2: Paper Trading

```
1. Add Alpaca API keys in Settings
2. Set Trading Mode: SECURE
3. Enable auto-refresh (optional)
4. Run AI Analysis (or wait for auto)
5. Review Position Sizing
6. Click execute
7. Check confirmation box
8. Order submitted to Alpaca
9. Check P&L for realized gains
```

### Workflow 3: Autonomous (Advanced)

```
1. Months of testing in SECURE mode
2. Add Alpaca API keys
3. Set Trading Mode: AUTOPILOT
4. Enable auto-refresh (3600s for hourly)
5. Set LIVE_TRADING_BUDGET in settings
6. Let it run
7. Monitor P&L dashboard daily
```

---

## 📋 Settings Reference

| Setting | Where | Purpose | Default |
|---------|-------|---------|---------|
| `SYMBOL` | settings.py | Training symbol | "TSLA" |
| `TIMEFRAME` | settings.py | "1h" or "1d" | "1h" |
| `INITIAL_BALANCE` | settings.py | Starting cash | $10,000 |
| `LIVE_TRADING_BUDGET` | settings.py | Max risk | $10,000 |
| `CASH_RISK_FRACTION` | settings.py | % per trade | 0.65 |
| `SLIPPAGE_PCT` | settings.py | Price tolerance | 0.0005 |
| `ALPACA_API_KEY` | .env or Settings | API key | None |
| `ALPACA_SECRET_KEY` | .env or Settings | Secret | None |

---

## 🚨 Common Issues

### "Price verification failed"
- Prices differ between yfinance and Alpaca
- Wait 60 seconds and retry
- Or switch to SIMULATE mode

### "Could not calculate confidence"
- Non-critical error
- Shows default 75% confidence
- Trade still works

### "Alpaca not connected"
- Missing API keys
- Add in Settings page or .env file

### "No model loaded"
- Go to Models page
- Select and load a trained model

---

## 📚 Documentation Index

- **README.md** - Quick start and overview
- **FEATURES_GUIDE.md** - This file - detailed feature explanations
- **HOW_IT_WORKS.md** - Trading mechanics and schedules
- **DATA_INTEGRITY.md** - Pipeline and data sources
- **TRADING_MECHANICS.md** - Price verification, hourly vs daily
- **MULTI_STOCK_SUPPORT.md** - Multi-stock trading
- **FAQ.md** - Quick answers
