# Demo Deployment Safety Guide

This guide explains all safety features for sharing your Live Trader app with others.

---

## 🔐 Password Protection

**What it does:** Requires a password before anyone can access the app.

**How it works:**
```
User visits URL
    ↓
🔒 Login Screen appears
    ↓
Enter password: "fox2024" (or your custom password)
    ↓
✅ Access granted → Full app loads
```

**Setup:**

Option 1: Use default password (simple demos)
- Password is already set: `fox2024`
- Just deploy and share URL + password

Option 2: Custom password (recommended)
```bash
# In .env file (for local)
DEMO_PASSWORD=your_secure_password

# Or in Streamlit Cloud Secrets (for deployment)
DEMO_PASSWORD = "your_secure_password"
```

**To disable** (local development only):
```python
# In app.py, comment out this line:
# require_auth()
```

---

## 🛡️ Demo Safety Limits

### Current Limits (Already Configured)

| Limit | Value | Purpose |
|-------|-------|---------|
| **Trading Budget** | $10,000 | Maximum capital the AI can use |
| **Cash Risk Fraction** | 30% | Max 30% of cash per trade |
| **Max Position** | 30% | Max 30% portfolio in one stock |
| **Default Mode** | Simulate | No real orders unless user changes |

### Why These Limits?

**Trading Budget ($10k):**
- Prevents accidental large trades
- Even if paper trading, keeps numbers reasonable
- Easy to understand for demo purposes

**Cash Risk Fraction (30%):**
- AI won't bet everything on one trade
- Prevents all-in scenarios
- Standard risk management practice

**Max Position (30%):**
- Limits single-stock exposure
- Encourages diversification
- Prevents "YOLO" trades

---

## 🚨 Warning Banners

### What They Look Like

**When deployed to cloud, users will see:**

```
┌─────────────────────────────────────────┐
│ 🌐 CLOUD DEPLOYMENT - DEMO MODE         │  ← Yellow warning
│                                          │
│ This is a shared demo. All users share   │
│ the same API keys and see the same data. │
└─────────────────────────────────────────┘
```

**If paper trading enabled:**
```
┌─────────────────────────────────────────┐
│ ✅ PAPER TRADING MODE                    │  ← Green info
│                                          │
│ No real money is being used. All trades  │
│ are simulated with fake money.           │
└─────────────────────────────────────────┘
```

**If somehow live trading:**
```
┌─────────────────────────────────────────┐
│ 🚨 LIVE TRADING MODE - REAL MONEY       │  ← Red danger!
│                                          │
│ This deployment uses real money! Do not  │
│ share this URL publicly!                 │
└─────────────────────────────────────────┘
```

### Why Show These?

1. **Transparency:** Users know it's a shared environment
2. **Safety:** Prevents confusion about whose trades they're seeing
3. **Accountability:** Everyone knows trades affect the shared account

---

## 🎯 Deployment Checklist for Safe Demo

### Before Deploying:

- [ ] ✅ Using **paper trading** (not live)
- [ ] ✅ Password protection **enabled**
- [ ] ✅ Trading budget **capped at $10k**
- [ ] ✅ `.env` file **NOT in Git**
- [ ] ✅ API keys stored in **Streamlit Secrets**
- [ ] ✅ Default mode is **Simulate**
- [ ] ✅ Telegram notifications **enabled** (you get alerts)

### What to Share:

**Email/Slack message:**
```
🦊 Fox of Wallstreet - Live Trading Demo

🔗 URL: https://your-app.streamlit.app
🔒 Password: fox2024 (or your custom password)

⚠️ Important:
- This is a SHARED demo environment
- All users see the same data and trades
- Using PAPER trading (fake money)
- For demonstration only

🎮 How to use:
1. Go to Models page, load a model
2. Go to Trade Dashboard
3. Run AI Analysis
4. Try different trading modes (Simulate is safest)

Questions? Reply to this message!
```

---

## 🔒 Security Layers Explained

```
┌─────────────────────────────────────────┐
│ Layer 1: Password Protection            │
│ - Requires password to access           │
│ - Prevents random visitors              │
└─────────────────────────────────────────┘
            ↓ (after login)
┌─────────────────────────────────────────┐
│ Layer 2: Paper Trading                  │
│ - All trades use fake money             │
│ - No financial risk                     │
└─────────────────────────────────────────┘
            ↓ (user action)
┌─────────────────────────────────────────┐
│ Layer 3: Trading Limits                 │
│ - Max $10k budget                       │
│ - Max 30% per trade                     │
│ - Prevents overtrading                  │
└─────────────────────────────────────────┘
            ↓ (execution)
┌─────────────────────────────────────────┐
│ Layer 4: Mode Defaults                  │
│ - Default: Simulate mode                │
│ - User must switch to real trading      │
│ - Extra confirmation required           │
└─────────────────────────────────────────┘
            ↓ (notification)
┌─────────────────────────────────────────┐
│ Layer 5: Telegram Alerts                │
│ - You get notified of all trades        │
│ - Monitor demo usage                    │
│ - Detect unexpected activity            │
└─────────────────────────────────────────┘
```

---

## 🎬 Demo Script (For Presentations)

### 1-Minute Quick Demo:

```
"Let me show you our AI trading system..."

[Login with password]
"First, we load a trained AI model..."
  → Click Models, select model, Load

"The AI analyzes market data in real-time..."
  → Go to Trade Dashboard
  → Click Run AI Analysis

"Here's what the AI is thinking..."
  → Point to decision card
  → Show confidence level
  → Explain feature importance

"In simulate mode, we can test..."
  → Execute a trade
  → Show virtual portfolio update
  → Show P&L tracking

"For real trading, we'd switch to Secure mode..."
  → Show mode selector
  → Explain price verification
  → Mention Alpaca integration

"Questions?"
```

### Key Talking Points:

- **Same pipeline:** Training and live use identical code
- **Price verification:** Prevents slippage
- **Multiple modes:** Simulate → Secure → Autopilot
- **Risk management:** Built-in limits and alerts

---

## 🚫 What NOT to Do

**DON'T:**
- ❌ Share URL without password protection
- ❌ Use live trading for public demos
- ❌ Share API keys in screenshots
- ❌ Leave auto-refresh on in Autopilot mode unattended
- ❌ Deploy from master branch (use demo branch)

**DO:**
- ✅ Always use paper trading for demos
- ✅ Set a strong demo password
- ✅ Monitor Telegram notifications
- ✅ Use Simulate mode as default
- ✅ Deploy from a dedicated demo branch

---

## 📊 Monitoring Your Demo

### What You Should Watch:

1. **Telegram Notifications**
   - Get alerts for every trade
   - Know who's using the demo

2. **Alpaca Dashboard**
   - View all paper trades
   - Monitor activity

3. **Streamlit Cloud Logs**
   ```
   Your App → Manage App → Logs
   ```

4. **Usage Analytics** (Streamlit Cloud)
   ```
   Your App → Settings → Analytics
   ```

---

## 🆘 Emergency Shutdown

If something goes wrong:

**Option 1: Delete the App**
```
Streamlit Cloud → Your App → ⋮ → Delete
```

**Option 2: Revoke API Keys**
```
Alpaca Dashboard → API Keys → Regenerate
```
This instantly invalidates the old keys.

**Option 3: Change Password**
```
Update DEMO_PASSWORD in Streamlit Secrets
Restart the app
```

---

## ✅ Final Pre-Deployment Checklist

```bash
# 1. Verify you're on demo branch
git branch
# → * demo/live-trader

# 2. Verify .env is NOT tracked
git status
# → Should not show .env

# 3. Verify paper trading
grep "ALPACA_PAPER" .env
# → ALPACA_PAPER=true

# 4. Test locally
streamlit run apps/live_trader/app.py
# → Login screen appears
# → Password works
# → App loads correctly

# 5. Deploy to Streamlit Cloud
# → Select demo branch
# → Add secrets
# → Deploy

# 6. Test deployed version
# → Visit URL
# → Login with password
# → Load model
# → Run simulation

# 7. Share with colleague
# → Send URL + password
# → Confirm they can access
```

---

## Questions?

- **Password not working?** Check DEMO_PASSWORD in Streamlit Secrets
- **App not loading?** Check Streamlit Cloud logs
- **Keys not recognized?** Verify secrets are spelled correctly
- **Want to disable password?** Comment out `require_auth()` in app.py

**See also:** [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide
