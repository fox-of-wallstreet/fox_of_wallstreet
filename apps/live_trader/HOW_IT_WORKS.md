# How the Live Trader App Works

## Trading Modes Explained

### 🔍 SIMULATE Mode (Current Screenshot)

**What it does:**
- Uses a **virtual portfolio** starting with `INITIAL_BALANCE` ($10,000)
- Does **NOT** connect to Alpaca
- Does **NOT** check your real account balance
- Tracks trades internally in `session_state["portfolio"]`

**Use case:** Test AI decisions without any real money risk

```python
# Simulate mode portfolio (isolated from real account)
portfolio = {
    "cash": 10000.0,        # Starts fresh every session
    "position": 0.0,        # Shares held (virtual)
    "entry_price": 0.0,     # Average entry price
    "last_action": 0,       # For feature calculation
}
```

---

### 🛡️ SECURE Mode (Paper Trading with Confirmation)

**What it does:**
- **Fetches** your Alpaca paper trading balance
- **Caps** the available cash to `LIVE_TRADING_BUDGET` ($10,000)
- AI suggests → You click confirm → Order goes to Alpaca

**Budget Logic:**
```python
# From live_trader.py line 342:
cash = min(raw_cash, settings.LIVE_TRADING_BUDGET)  # Caps at $10k
```

**Why cap it?**
- Your Alpaca account might have $100,000
- You only want to risk $10,000 with this AI
- Prevents accidentally trading your entire account

**Use case:** Controlled live testing with manual oversight

---

### 🤖 AUTOPILOT Mode (Fully Automated)

**What it does:**
- Same balance logic as SECURE mode
- AI decides → Order executes immediately
- No confirmation step

**Use case:** Production trading (requires extensive testing first!)

---

## Auto-Refresh / Autonomous Mode

### Current Implementation

The app now supports auto-refresh via the sidebar:

1. **Enable Auto-Refresh** checkbox in sidebar
2. Set refresh interval (30-300 seconds)
3. App will automatically:
   - Fetch latest prices
   - Run AI analysis
   - Update recommendations

### How It Works

```python
# When auto-refresh is enabled:
if auto_refresh_enabled:
    st_autorefresh(interval=60 * 1000)  # Every 60 seconds
    
    # Trigger AI analysis automatically
    if time_since_last_run > interval:
        run_ai_inference(...)
```

### Manual Execution Still Works

Even with auto-refresh enabled, you can:
- Click "Run AI Analysis" manually
- Override AI recommendations
- Switch modes

---

## Understanding SELL_50 and BUY_50

### SELL_50 (Sell 50%)

```python
# SELL_50 means: Sell 50% of your CURRENT POSITION (shares)
# NOT 50% of balance!

shares_to_sell = current_position * 0.5

# Example:
# You own 100 shares
# SELL_50 → Sell 50 shares
# You now own 50 shares
```

### BUY_50 (Buy 50%)

```python
# BUY_50 means: Invest 50% of AVAILABLE CASH
# After applying CASH_RISK_FRACTION!

available_cash = min(alpaca_cash, LIVE_TRADING_BUDGET)
investment = available_cash * 0.5 * CASH_RISK_FRACTION

# Example:
# Available cash: $10,000
# CASH_RISK_FRACTION: 0.65
# BUY_50 → Invest $10,000 * 0.5 * 0.65 = $3,250
```

### Complete Example

```
Starting state (SIMULATE mode):
  Cash: $10,000
  Position: 0 shares

1. AI recommends: BUY_50
   Investment: $10,000 * 0.5 * 0.65 = $3,250
   Price: $250/share
   → Buy 13 shares
   
   New state:
     Cash: $6,750
     Position: 13 shares @ $250

2. Price rises to $300
   AI recommends: SELL_50
   → Sell 50% of position = 6.5 shares
   
   New state:
     Cash: $6,750 + ($300 * 6.5) = $8,700
     Position: 6.5 shares @ $250
```

---

## Settings Reference

| Setting | File | Purpose | Default |
|---------|------|---------|---------|
| `INITIAL_BALANCE` | settings.py | Starting cash for training & simulate | $10,000 |
| `LIVE_TRADING_BUDGET` | settings.py | Max cash to use from Alpaca account | $10,000 |
| `CASH_RISK_FRACTION` | settings.py | % of cash to invest per BUY | 0.65 (65%) |
| `ACTION_SPACE_TYPE` | settings.py | "discrete_3" or "discrete_5" | "discrete_5" |

---

## Does Budget Conflict with Alpaca Balance?

**Short answer: NO** - they work together.

**How it works:**

```python
def get_current_position_features(trading_client, latest_price):
    # 1. Get REAL Alpaca account
    account = trading_client.get_account()
    raw_cash = float(account.cash)  # e.g., $100,000
    
    # 2. CAP to your trading budget
    cash = min(raw_cash, settings.LIVE_TRADING_BUDGET)  # $10,000
    
    # 3. Get REAL position
    try:
        position = trading_client.get_open_position(settings.SYMBOL)
        current_shares = float(position.qty)  # Real shares
    except:
        current_shares = 0.0
    
    return current_shares, cash, entry_price, portfolio_features
```

**Why this is safe:**
- Even if Alpaca shows $100,000, we only use $10,000
- The position shares are real (synced with Alpaca)
- Budget protects you from over-trading

---

## Price Data Sources

### Current: Yahoo Finance (yfinance)

```python
# feature_fetcher.py
data = yf.download(
    symbol='TSLA',
    period='60d',
    interval='1h',  # or '1d'
    auto_adjust=True  # Adjusted for splits/dividends
)
```

**Pros:**
- Free
- No API key needed
- Easy to use

**Cons:**
- Delayed ~15-20 minutes
- May differ slightly from Alpaca prices

### Recommended for Production: Alpaca Market Data

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(api_key, secret_key)

request = StockBarsRequest(
    symbol_or_symbols="TSLA",
    timeframe=TimeFrame.Hour,
    start=datetime.now() - timedelta(days=5)
)

bars = client.get_stock_bars(request)
```

**Pros:**
- Same price source as trading
- Real-time (with paid subscription)
- Consistent data

**Cons:**
- Requires API key
- Rate limits on free tier

### Price Discrepancy Risk

**The problem:**
```
1. AI sees yfinance price: $250.00
2. AI recommends: BUY_50  
3. You click execute
4. Alpaca executes at: $250.25 (0.1% higher)
5. Position is slightly off from expectation
```

**Mitigation:**
- Use `SLIPPAGE_PCT` setting (default: 0.05%)
- For production, switch to Alpaca market data
- Verify price before execute

See FAQ.md for more details.

---

## Daily vs Hourly Models

**YES, the architecture supports both!**

### How it works:

The model metadata includes `timeframe`:

```python
# metadata.json
{
    "timeframe": "1h",  # or "1d"
    "action_space": "discrete_5",
    ...
}
```

### Feature Computation

Both timeframes use the same feature code:

```python
# core/processor.py - automatically adapts
def _compute_sin_time(df):
    if settings.TIMEFRAME == "1h":
        # Hourly: intraday cycle (1440 minutes)
        mins = df["Date"].dt.hour * 60 + df["Date"].dt.minute
        df["Sin_Time"] = np.sin(2 * np.pi * mins / 1440.0)
    else:
        # Daily: weekly cycle (7 days)
        df["Sin_Time"] = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7.0)
```

### What You Need to Do

When loading a **daily model**:

1. The app reads `metadata.json` → `timeframe: "1d"`
2. Price fetching uses `interval="1d"` automatically
3. Features compute correctly for daily bars

**The UI looks the same** - just less frequent updates.

---

## Quick Decision Cheat Sheet

| Action | With $10,000 Cash, $0 Position | With $5,000 Cash, 20 Shares |
|--------|-------------------------------|---------------------------|
| **BUY_100** | Invest $6,500 → Buy shares | Invest $3,250 → Buy shares |
| **BUY_50** | Invest $3,250 → Buy shares | Invest $1,625 → Buy shares |
| **HOLD** | Do nothing | Do nothing |
| **SELL_50** | ❌ Error (no position) | Sell 10 shares |
| **SELL_100** | ❌ Error (no position) | Sell 20 shares (flat) |

---

## Common Confusions

### ❌ "SELL_50 means sell 50% of my account"
**✅ NO**: It means sell 50% of your **shares**, not dollars.

### ❌ "The app uses my full Alpaca balance"
**✅ NO**: It caps at `LIVE_TRADING_BUDGET` (default $10k).

### ❌ "Simulate mode reads my Alpaca account"
**✅ NO**: Simulate uses isolated virtual portfolio.

### ❌ "Daily models need different code"
**✅ NO**: Same code, just change `TIMEFRAME` in settings.

### ❌ "Yahoo and Alpaca prices are identical"
**⚠️ MOSTLY**: Can differ by $0.01-$0.50, especially during volatility.

---

## Production Readiness Checklist

Before trading real money:

- [ ] Extensive testing in SIMULATE mode (2+ weeks)
- [ ] Paper trading in SECURE mode (1+ month)
- [ ] Profitable paper trading in AUTOPILOT mode (1+ month)
- [ ] Switch to Alpaca market data (not yfinance)
- [ ] Implement price verification before execute
- [ ] Set up alerts/notifications
- [ ] Document your strategy and risk limits
- [ ] Start with small position sizes

---

## Workflow Recommendation

```
Phase 1: SIMULATE mode
  ↓ Test for 1-2 weeks, verify AI makes sensible decisions
  
Phase 2: SECURE mode (Paper Trading)
  ↓ Add Alpaca keys, confirm each trade manually
  
Phase 3: AUTOPILOT mode (Paper Trading)
  ↓ Enable auto-refresh, let AI run automatically
  
Phase 4: Live Trading (Real Money)
  ↓ Only after months of profitable paper trading!
```

---

## Checking Your Settings

In the app, go to **Settings** page to see:
- `INITIAL_BALANCE`: $10,000
- `LIVE_TRADING_BUDGET`: $10,000
- `CASH_RISK_FRACTION`: 0.65
- `TIMEFRAME`: 1h (or 1d)

To change: Edit `config/settings.py` and restart the app.
