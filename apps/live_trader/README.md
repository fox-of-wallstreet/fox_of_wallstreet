# 🦊 Live Trader

Real-time AI-powered trading dashboard for production use.

## Quick Links
- [Setup & Installation](#setup)
- [How It Works](#how-it-works)
- [Deployment Guide](#deployment)
- [FAQ & Troubleshooting](#faq)

---

## Setup

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (create .env file)
cat > .env << EOF
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
EOF

# 3. Run the app
streamlit run app.py
```

### Configuration

Key settings in `config/settings.py`:
- `LIVE_TRADING_BUDGET`: Max capital to use ($10,000 default)
- `CASH_RISK_FRACTION`: Max 30% per trade
- `MAX_POSITION_PCT`: Max 30% in one stock

---

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADER FLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐     ┌───────────┐  │
│  │   yfinance   │──────▶│   Feature    │────▶│   Model   │  │
│  │  (prices)    │      │  Pipeline    │     │ Inference │  │
│  └──────────────┘      └──────────────┘     └─────┬─────┘  │
│                                                   │        │
│  ┌──────────────┐      ┌──────────────┐          ▼        │
│  │    User      │◀─────│   Decision   │◀────┌─────────┐  │
│  │  (confirm)   │      │    Card      │     │ Action  │  │
│  └──────┬───────┘      └──────────────┘     └────┬────┘  │
│         │                                         │        │
│         ▼                                         ▼        │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   EXECUTION                          │  │
│  │  ┌──────────────┐     ┌──────────────┐              │  │
│  │  │   Simulate   │     │    Alpaca    │              │  │
│  │  │  (virtual)   │     │   (real)     │              │  │
│  │  └──────────────┘     └──────────────┘              │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Pipeline

The live trading pipeline is **identical** to training:

| Component | Training | Live Trading |
|-----------|----------|--------------|
| Price source | yfinance | yfinance ✅ |
| Feature computation | `add_technical_indicators()` | Same ✅ |
| AVWAP calculation | `core/avwap.py` | Same ✅ |
| Feature registry | `FEATURE_REGISTRY` | Same ✅ |
| Scaling | `RobustScaler.fit()` | `RobustScaler.transform()` ✅ |

This ensures the AI sees the same data distribution it was trained on.

### Trading Modes

| Mode | Refresh Default | Behavior | Use Case |
|------|-----------------|----------|----------|
| **🔍 Simulate** | 30s | Virtual portfolio, no real orders | Testing/demo |
| **🛡️ Secure** | 60s | Real orders, requires confirmation | Safe live trading |
| **🤖 Autopilot** | 300s | Auto-executes after verification | Full automation |

**Why different refresh defaults?**
- Simulate: Fast feedback for demos
- Secure: Balanced for manual confirmation
- Autopilot: Conservative to prevent overtrading and respect rate limits

### Action Space

The AI outputs actions based on the loaded model:

**Discrete 3 (Buy All / Sell All / Hold):**
```
Action 0: SELL_ALL - Sell entire position
Action 1: BUY_ALL - Invest 30% of cash
Action 2: HOLD - Do nothing
```

**Discrete 5 (Graduated actions):**
```
Action 0: SELL_100% - Sell all shares
Action 1: SELL_50% - Sell half position
Action 2: HOLD - Do nothing
Action 3: BUY_50% - Invest 15% of cash
Action 4: BUY_100% - Invest 30% of cash
```

### Price Verification

Before executing real trades, the app verifies:
1. Compare yfinance price (AI analysis) with Alpaca price (execution)
2. If difference > 0.1% → Trade blocked
3. Protects against slippage and stale data

---

## Deployment

### Streamlit Cloud (Recommended for Demo)

1. Push code to GitHub (ensure `.env` is gitignored!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo and branch
4. Add secrets:
   ```
   ALPACA_API_KEY = "your_key"
   ALPACA_SECRET_KEY = "your_secret"
   ALPACA_PAPER = "true"
   DEMO_PASSWORD = "fox2024"  # optional password protection
   ```
5. Deploy!

**Password Protection:**
- Default password: `fox2024`
- Set custom: Add `DEMO_PASSWORD` to secrets
- Disable: Comment out `require_auth()` in `app.py`

### Other Options

- **VPS (AWS/DigitalOcean):** Full control, 24/7 uptime
- **Docker:** Consistent environments
- **Heroku/Railway:** Quick deployment

See full details in project-level deployment guides.

### Safety Checklist

Before deploying:
- [ ] `.env` is in `.gitignore`
- [ ] Using paper trading (`ALPACA_PAPER=true`)
- [ ] Trading budget capped
- [ ] Password protection enabled (for public demos)
- [ ] Telegram alerts configured

---

## FAQ

### General

**Q: Is the data pipeline the same as training?**
A: YES - We use identical functions for feature computation, AVWAP calculation, and scaling. The only difference is news features are set to 0 in live trading (no real-time news API yet).

**Q: Can I use models trained on other stocks?**
A: NO (by design). A TSLA model should not trade AAPL. The app warns you if settings don't match the model's training symbol.

**Q: Why is confidence sometimes 50%?**
A: 50% is the minimum (random). Normal range is 50-95%. Above 80% is high confidence.

### Deployment

**Q: Where is the mode selector?**
A: In the left sidebar under "🎮 Trading Mode"

**Q: How do I configure API keys?**
A: For cloud deployment, add them to Streamlit Secrets. For local, use `.env` file.

**Q: When do Telegram notifications trigger?**
A: Only in SECURE or AUTOPILOT modes (not SIMULATE), and only if TELEGRAM_TOKEN/CHAT_ID are set.

**Q: Are my API keys safe in deployment?**
A: YES if you:
- Store in environment variables (never hardcode)
- Use Streamlit Cloud Secrets or Heroku Config Vars
- Use paper trading for demos

### Trading

**Q: Does Simulate mode use my Alpaca balance?**
A: NO - Simulate uses a virtual portfolio starting at INITIAL_BALANCE. It never connects to Alpaca.

**Q: What does SELL_50 mean?**
A: Sell 50% of your POSITION (shares), not balance. If you own 100 shares → sell 50 shares.

**Q: Can I paper trade without confirmation?**
A: YES - Use AUTOPILOT mode. But only after extensive testing in SECURE mode!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No model loaded" | Go to Models page and load a model first |
| "Alpaca not connected" | Add API keys in Settings page or .env file |
| "Price verification failed" | Normal during volatility - prices differ between sources |
| App won't start | Check `requirements.txt` installed correctly |

---

## Project Structure

```
apps/live_trader/
├── app.py                    # Main entry point
├── requirements.txt          # Dependencies
│
├── pages/                    # Streamlit pages
│   ├── 01_trade.py          # Trading dashboard
│   ├── 02_models.py         # Model selector
│   ├── 03_history.py        # Trade history
│   └── 04_settings.py       # Configuration
│
├── components/               # Reusable UI components
│   ├── decision_card.py
│   ├── model_selector.py
│   ├── pnl_tracker.py
│   └── position_sizing.py
│
└── utils/                    # Utility functions
    ├── alpaca_client.py     # Alpaca trading API
    ├── auth.py              # Password protection
    ├── feature_fetcher.py   # Market data pipeline
    └── telegram.py          # Notifications
```

---

## Changelog

### v1.0.0 (Current)
- ✅ Alpaca integration (paper/live trading)
- ✅ Three trading modes: Simulate, Secure, Autopilot
- ✅ Mode-specific refresh intervals
- ✅ Password protection for demos
- ✅ Telegram notifications
- ✅ P&L tracking (Trading 212 style)
- ✅ Price verification before execution
- ✅ Multi-stock support (reads from model metadata)

---

## License & Disclaimer

⚠️ **DISCLAIMER:** This is educational software. Trading involves risk. Past performance does not guarantee future results. Always use paper trading before live trading.

---

## Support

- 📧 Email: your-email@example.com
- 🐛 Issues: GitHub Issues
- 📖 Docs: This README + code comments
