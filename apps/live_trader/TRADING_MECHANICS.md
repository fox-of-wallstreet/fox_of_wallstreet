# Trading Mechanics & Schedule Guide

## Trading Modes - How They Actually Work

### 🔍 SIMULATE Mode

**What happens when you click "Execute":**

```python
1. Calculate trade locally
2. Update virtual portfolio in session_state
3. Log to activity log
4. NO ORDERS sent to Alpaca
```

**Use case:** Test AI decisions without risk

---

### 🛡️ SECURE Mode

**What happens when you click "Execute":**

```python
1. Sync portfolio from Alpaca (get real cash/position)
2. VERIFY price (compare yfinance vs Alpaca)
3. IF price mismatch > 0.1%:
       BLOCK trade, show error
   ELSE:
       Show confirmation checkbox
       IF user checks box:
           Submit order to Alpaca
           Update local portfolio
       ELSE:
           Wait for confirmation
```

**Price Verification:**
- Compares yfinance price (from AI analysis) with Alpaca live price
- Max allowed difference: `SLIPPAGE_PCT * 2` (default 0.1%)
- If prices differ too much → Trade blocked

**Why block?** Protects you from:
- Stale data (yfinance delayed 15-20 min)
- Fast market moves
- Slippage

---

### 🤖 AUTOPILOT Mode

**What happens:**

```python
1. Sync portfolio from Alpaca
2. VERIFY price (same as SECURE)
3. IF price OK:
       Submit order immediately (no confirmation)
   ELSE:
       BLOCK trade
```

**⚠️ DANGER:** No confirmation step! Use only after extensive testing.

---

## Price Verification FAQ

### Q: What happens if price verification fails?

**A: Trade is BLOCKED.** You'll see:
```
❌ Price verification failed: 
   Expected $250.00, Alpaca shows $252.50 (1.0% diff)

Trade blocked to protect against price slippage.
```

**Your options:**
1. Wait for prices to converge
2. Switch to SIMULATE mode (to see what AI would do)
3. Manually override (not recommended)

### Q: Does price verification run in all modes?

**A: YES**, for BUY and SELL actions:
- ✅ SIMULATE: No (uses yfinance price directly)
- ✅ SECURE: Yes
- ✅ AUTOPILOT: Yes

HOLD actions skip verification (no trade needed).

### Q: What causes price mismatch?

| Cause | Typical Diff | Solution |
|-------|--------------|----------|
| yfinance delay | 0.1-0.5% | Wait 60s, refresh |
| Market volatility | 0.5-2.0% | Wait for calm |
| After hours | Large | Trade during market hours |
| Data error | Any | Check other sources |

### Q: Can I disable price verification?

**A: NO** (by design). But you can:
- Increase `SLIPPAGE_PCT` in settings (e.g., 0.01 = 1%)
- Use SIMULATE mode

---

## Auto-Refresh & Trading Schedule

### How Auto-Refresh Works

```python
# When enabled in sidebar:
st_autorefresh(interval=60 * 1000)  # Every 60 seconds

# What happens each refresh:
1. Fetch new price data from yfinance
2. Compute features
3. Run AI inference
4. Display new recommendation

# Does it auto-trade?
- SIMULATE: No (shows recommendation only)
- SECURE: No (waits for confirmation)
- AUTOPILOT: YES (if price verification passes)
```

### Refresh Rate Recommendations

| Mode | Recommended | Why |
|------|-------------|-----|
| SIMULATE | 60-300s | For demo, watching AI think |
| SECURE | 300-900s (5-15min) | Time to review and confirm |
| AUTOPILOT | Match your timeframe | See below |

---

## Trading Schedule: Hourly vs Daily Models

### ⏰ Hourly Models (1h)

**When to trade:**
- Market hours: 9:30 AM - 4:00 PM ET
- New data available: Every hour
- Best refresh rate: 3600s (1 hour)

**Schedule:**
```
9:30 AM - Market opens, first hour bar
10:30 AM - Second hour bar available
11:30 AM - Third hour bar available
...
4:00 PM - Market close, last hour bar
```

**Auto-refresh for hourly:**
```python
# Set in sidebar:
Refresh Interval: 3600 seconds (1 hour)

# App will:
- Wait for each hour candle to close
- Run AI analysis
- Execute trade (if autopilot)
```

### 📅 Daily Models (1d)

**When to trade:**
- Once per day, after market close
- New data available: ~4:30 PM ET (after close)
- Best refresh rate: 86400s (24 hours) or manual

**Schedule:**
```
Day 1:
  9:30 AM - Market opens
  4:00 PM - Market closes
  4:30 PM - Daily bar available on yfinance
  
Day 2:
  9:30 AM - Can trade based on Day 1's analysis
```

**Auto-refresh for daily:**
```python
# Set in sidebar:
Refresh Interval: 86400 seconds (24 hours)

# Or better: Disable auto-refresh
# Run manually once per day after market close
```

---

## Close Price Availability & Execution

### The Timeline

**Hourly Trading:**
```
10:00 AM - 11:00 AM: Hourly candle forms
11:00 AM: Candle closes, data available on yfinance
11:01 AM: Your app fetches data (if auto-refresh)
11:02 AM: AI analysis runs
11:03 AM: Order submitted (if autopilot)
11:05 AM: Order filled by Alpaca
```

**Daily Trading:**
```
Day 1, 4:00 PM: Market closes
Day 1, 4:30 PM: Daily candle available on yfinance
Day 1, 5:00 PM: Your app fetches data (manual or auto)
Day 1, 5:05 PM: AI analysis runs
Day 1, 5:10 PM: Order submitted (for next day)

Day 2, 9:30 AM: Order filled at market open
```

### ⚠️ Important: You Can't Trade the "Close"

**Common misconception:**
> "The model saw the close price at $250, so I'll buy at $250"

**Reality:**
- By the time you see the close, market is closed
- Your order executes at **next available price** (next hour/day open)
- Price may gap up/down

**Example (Daily):**
```
Day 1 close: $250 (model says BUY)
Day 2 open:  $255 (you actually buy here)
Day 2 close: $260 (you profit, but entry was higher)
```

### Solutions

1. **Accept slippage** (normal for all trading)
2. **Use limit orders** (not yet implemented)
3. **Trade intraday** (hourly models have less gap risk)

---

## Recommended Workflows

### For Hourly Trading (Autopilot)

```python
# Settings:
TIMEFRAME = "1h"
Trading Mode: AUTOPILOT
Auto-refresh: 3600s (1 hour)
SLIPPAGE_PCT: 0.0005 (0.05%)

# Schedule:
During market hours (9:30 AM - 4:00 PM ET)
→ App runs every hour automatically
→ Trades execute at market price
→ Check activity log at end of day
```

### For Daily Trading (Secure)

```python
# Settings:
TIMEFRAME = "1d"
Trading Mode: SECURE
Auto-refresh: OFF (manual)
SLIPPAGE_PCT: 0.001 (0.1%)

# Schedule:
4:30 PM ET (after market close)
1. Manually click "Run AI Analysis"
2. Review recommendation
3. Check price verification
4. Confirm trade
5. Order submits for next day
```

### For Demo/Testing

```python
# Settings:
TIMEFRAME: Any
Trading Mode: SIMULATE
Auto-refresh: 60s (fast)

# What happens:
→ No real orders
→ Fast feedback loop
→ Test different refresh rates
→ Safe experimentation
```

---

## Price Verification in Detail

### The Comparison

```python
def verify_price(expected_price, max_diff_pct=0.005):
    # 1. Get price from yfinance (what AI used)
    yf_price = $250.00  # From analysis
    
    # 2. Get price from Alpaca (where we trade)
    alpaca_price = $250.30  # Real-time quote
    
    # 3. Calculate difference
    diff_pct = |250.30 - 250.00| / 250.00 = 0.12%
    
    # 4. Check threshold (0.5% default)
    if diff_pct > 0.005:
        return BLOCKED
    else:
        return APPROVED
```

### Why Both Prices?

| Source | Used For | Why |
|--------|----------|-----|
| yfinance | AI analysis | Free, easy, historical |
| Alpaca | Execution | Real trading, real fills |

**The problem:** AI made decision based on $250.00, but market moved to $250.30.

**The solution:** Verify before risking money.

---

## Summary Table

| Mode | Price Verify | Auto-Trade | Confirmation | P&L Tracking | Use Case |
|------|--------------|------------|--------------|--------------|----------|
| SIMULATE | ❌ | ❌ | ❌ | ✅ Virtual | Testing |
| SECURE | ✅ | ❌ | ✅ | ✅ Realized | Paper trading |
| AUTOPILOT | ✅ | ✅ | ❌ | ✅ Realized | Production |

## New Features

### Position Sizing Preview
Before you execute, see:
- Investment amount ($)
- Shares to trade
- New position size
- Risk % of portfolio

**How it helps:** Know exactly what you're doing before risking money.

### P&L Tracker (Trading 212 Style)
**Unrealized P&L:**
- Live profit/loss on open position
- Updates with every price tick
- Shows % gain/loss

**Realized P&L:**
- Actual cash profit from closed trades
- Win rate statistics
- Average winner vs loser ratio

**How it helps:** Track performance like a pro trading app.

### Position Size Calculator
Manual calculator for planning:
- Enter account size
- Select risk %
- See trade details

**How it helps:** Plan trades without loading a model.

| Timeframe | Refresh Rate | Trade Timing | Best For |
|-----------|--------------|--------------|----------|
| 1h | 3600s | Intraday | Active trading |
| 1d | Manual/24h | Next day open | Swing trading |

---

## Quick Commands

```bash
# Test price verification manually:
cd apps/live_trader
python -c "
from utils.alpaca_client import AlpacaTrader
from utils.feature_fetcher import fetch_recent_prices

alpaca = AlpacaTrader()
price_df = fetch_recent_prices('TSLA', '1h')
yf_price = price_df['Close'].iloc[-1]

valid, alpaca_price, msg = alpaca.verify_price(yf_price)
print(f'yfinance: ${yf_price:.2f}')
print(f'Alpaca: ${alpaca_price:.2f}')
print(f'Result: {msg}')
"
```
