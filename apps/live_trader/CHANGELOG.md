# Changelog - New Features

## Version 1.1.0 - Major Update

### ✅ New Features Added

#### 1. Position Sizing Preview
**File:** `components/position_sizing.py`

**What it does:**
- Shows exact trade details before execution
- Calculates investment amount, shares, new position
- Displays risk % of portfolio

**How to use:**
- Appears automatically after AI analysis
- Review before clicking execute
- Check "Risk %" column

**Example output:**
```
Investment: $3,250.00
Shares: 13.0000
New Position: 13.0000 (+13.0000)
Risk %: 32.5%
```

---

#### 2. P&L Tracker (Trading 212 Style)
**File:** `components/pnl_tracker.py`

**What it does:**
- **Unrealized P&L**: Live profit/loss on open position (updates with price)
- **Realized P&L**: Actual cash profit from closed trades
- **Trading Stats**: Win rate, winning/losing trades, avg win/loss ratio

**How to interpret:**
- 🟢 Green = Profit
- 🔴 Red = Loss
- **Unrealized** = Paper gains (position still open)
- **Realized** = Locked-in profits (position closed)

**Example output:**
```
Total Return: +$523.45
Unrealized P&L: +$123.45 (+5.23%)
Realized P&L: +$400.00
Win Rate: 60.0% (3 wins, 2 losses)
```

---

#### 3. Position Size Calculator
**File:** `components/position_sizing.py`

**What it does:**
- Manual calculator for planning trades
- Independent of AI recommendations
- Educational tool

**How to use:**
1. Expand "Manual Calculator"
2. Enter account size and price
3. Select risk % and action
4. Click Calculate
5. See projected trade

---

#### 4. Full Alpaca Integration
**File:** `utils/alpaca_client.py`

**What was added:**
- ✅ Order submission (market orders)
- ✅ Portfolio sync (cash, positions)
- ✅ Price fetching for verification
- ✅ Price verification before trades
- ✅ Secure mode with confirmation
- ✅ Autopilot mode (auto-execute)

**How it works:**
```
SECURE Mode:
1. Sync portfolio from Alpaca
2. Verify price (yfinance vs Alpaca)
3. IF price OK:
     Show confirmation checkbox
     User checks box → Order submitted
   ELSE:
     Block trade

AUTOPILOT Mode:
1. Sync portfolio
2. Verify price
3. IF price OK:
     Auto-submit order
   ELSE:
     Block and wait
```

---

#### 5. Auto-Refresh / Autonomous Mode
**Updated:** `pages/01_trade.py`

**What it does:**
- Automatically runs AI analysis on schedule
- Can auto-execute trades (AUTOPILOT mode)
- Configurable interval (30-300 seconds)

**How to use:**
1. Check "Enable Auto-Refresh" in sidebar
2. Set interval
3. App refreshes automatically

---

#### 6. Price Verification
**File:** `utils/alpaca_client.py`, `pages/01_trade.py`

**What it does:**
- Compares yfinance price (AI analysis) with Alpaca price (execution)
- Blocks trades if prices differ by > 0.1%
- Protects against slippage

**When it fails:**
```
❌ Price verification failed: 
   Expected $250.00, Alpaca shows $252.50 (1.0% diff)
Trade blocked to protect against price slippage.
```

---

#### 7. Multi-Stock Support (Fixed)
**Updated:** `pages/01_trade.py`, `components/model_selector.py`

**What was fixed:**
- App now reads symbol from model metadata
- Automatically fetches correct prices
- Supports switching between stocks

**How it works:**
```
Load AAPL model → Fetches AAPL prices
Load TSLA model → Fetches TSLA prices
```

---

### 📁 New Files

| File | Purpose |
|------|---------|
| `utils/alpaca_client.py` | Alpaca trading integration |
| `components/position_sizing.py` | Position sizing calculator |
| `components/pnl_tracker.py` | P&L tracking and statistics |
| `FEATURES_GUIDE.md` | Complete feature documentation |
| `CHANGELOG.md` | This file |

### 📁 Updated Files

| File | Changes |
|------|---------|
| `pages/01_trade.py` | Added Alpaca integration, auto-refresh, P&L tracking, position sizing |
| `components/model_selector.py` | Added symbol tracking from model metadata |
| `requirements.txt` | Added `streamlit-autorefresh` |
| `README.md` | Updated feature list |
| `FAQ.md` | Updated Alpaca integration info, added .env instructions |
| `TRADING_MECHANICS.md` | Added new features section |

---

## How Each Feature Works

### Position Sizing Preview
```
User clicks "Execute" 
    ↓
System calculates:
  - Investment = Cash × 0.5 × 0.65 (for BUY_50)
  - Shares = Investment / Price
  - New Position = Current + Shares
  - Risk % = Investment / Portfolio
    ↓
Shows preview card
    ↓
User confirms execution
```

### P&L Tracking
```
When BUY happens:
  - Records entry price
  - Tracks position

When SELL happens:
  - Calculates realized P&L:
    Realized = (Exit Price - Entry Price) × Shares
  - Adds to trade history
  - Updates statistics

Every price update:
  - Calculates unrealized P&L:
    Unrealized = (Current Price - Entry Price) × Position
  - Updates display
```

### Alpaca Integration
```
User clicks "Execute" in SECURE mode
    ↓
Connect to Alpaca API
    ↓
Sync portfolio (get real cash/position)
    ↓
Verify price:
  yfinance_price vs alpaca_price
    ↓
IF difference < 0.1%:
  Show confirmation checkbox
  User checks box
      ↓
  Submit order to Alpaca
      ↓
  Get order ID
      ↓
  Update local portfolio
ELSE:
  Show error, block trade
```

---

## Quick Reference

### API Keys Setup

**Option 1: .env file (Recommended)**
```bash
# In project root .env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

**Option 2: Settings Page**
- Open Settings in app
- Enter keys directly

### Trading Mode Differences

| Feature | SIMULATE | SECURE | AUTOPILOT |
|---------|----------|--------|-----------|
| Real orders | ❌ | ✅ | ✅ |
| Price verify | ❌ | ✅ | ✅ |
| Confirmation | ❌ | ✅ | ❌ |
| P&L tracking | Virtual | Real | Real |
| Auto-refresh | ✅ | ✅ | ✅ |

### Confidence Score

| Range | Meaning |
|-------|---------|
| 80-100% | Very confident |
| 60-80% | Moderate confidence |
| 50-60% | Uncertain |
| <50% | Very uncertain (rare) |

---

## Next Steps

1. **Install new dependency:**
   ```bash
   pip install streamlit-autorefresh alpaca-py
   ```

2. **Add API keys to .env:**
   ```bash
   echo "ALPACA_API_KEY=your_key" >> .env
   echo "ALPACA_SECRET_KEY=your_secret" >> .env
   ```

3. **Run the app:**
   ```bash
   cd apps/live_trader
   streamlit run app.py
   ```

4. **Test new features:**
   - Load a model
   - Run AI Analysis
   - See Position Sizing Preview
   - Check P&L Tracker
   - Try manual calculator
