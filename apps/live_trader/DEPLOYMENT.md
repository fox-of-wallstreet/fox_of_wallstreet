# Deployment Guide

## Table of Contents
1. [Refresh Interval Best Practices](#refresh-interval-best-practices)
2. [API Key Security](#api-key-security)
3. [Deployment Options](#deployment-options)
4. [Demo Deployment Setup](#demo-deployment-setup)

---

## Refresh Interval Best Practices

### Why Mode-Specific Defaults?

Different trading modes have different risk profiles and use cases:

| Mode | Default | Range | Rationale |
|------|---------|-------|-----------|
| **🔍 Simulate** | 30s | 10-300s | Fast for demo/testing, no financial risk |
| **🛡️ Secure** | 60s | 30-600s | Balanced - you confirm each trade |
| **🤖 Autopilot** | 300s | 60-3600s | Conservative - real money at stake |

### Industry Standards

**High-Frequency Trading (HFT):** Sub-second (not applicable here)
**Algorithmic Trading:** 1-15 minutes between decisions
**Swing Trading:** 1-4 hours or daily
**Position Trading:** Daily or weekly

**Our AI Model Recommendations:**

| Model Type | Recommended Refresh | Why |
|------------|---------------------|-----|
| Hourly (1h) | 300-3600s (5-60 min) | Match bar interval |
| Daily (1d) | 86400s (24 hours) | Daily rebalancing |
| Demo/Testing | 10-60s | Fast feedback loop |

### Why 300s (5 min) for Autopilot?

1. **Rate Limits:** Alpaca has API limits (200 requests/minute free tier)
2. **Slippage Protection:** Prices don't change dramatically in 5 minutes
3. **Overtrading Prevention:** Prevents excessive fees and emotional trading
4. **Market Hours:** Only trade during market hours anyway

---

## API Key Security

### ⚠️ CRITICAL: Never Commit API Keys!

**Bad (Don't do this):**
```python
# config/settings.py
ALPACA_API_KEY = "PK1234567890abcdef"  # ❌ NEVER!
```

**Good (Do this):**
```python
# config/settings.py
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")  # ✅ Load from env
```

### Security Levels by Deployment Type

#### Level 1: Local Development (Safest)
```bash
# .env file (gitignored)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
Keys never leave your machine.

#### Level 2: Private Server/VPS
```bash
# SSH into server, set env vars
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret

# Or use .env file with restricted permissions
chmod 600 .env
```

#### Level 3: Docker Deployment
```dockerfile
# Use build args or runtime env
docker run -e ALPACA_API_KEY=$ALPACA_API_KEY ...
```

#### Level 4: Cloud Platform (Streamlit Cloud, Heroku, etc.)
Use **Secrets Management** (see below).

---

## Deployment Options

### Option 1: Streamlit Community Cloud (Free, Easiest)

**Best for:** Demos, sharing with collaborators

**Steps:**
1. Push code to GitHub (WITHOUT .env file!)
2. Connect repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in Streamlit Cloud dashboard:
   ```
   Settings → Secrets → Add:
   
   ALPACA_API_KEY = "your_key"
   ALPACA_SECRET_KEY = "your_secret"
   ALPACA_PAPER = "true"
   TELEGRAM_TOKEN = "your_token"  # optional
   TELEGRAM_CHAT_ID = "your_chat_id"  # optional
   ```

**Pros:**
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Built-in secrets management
- ✅ Easy to share URL

**Cons:**
- ❌ 1GB RAM limit (may need to optimize)
- ❌ Sleeps after inactivity (wakes on visit)
- ❌ US-only regions

**Security Note:** Streamlit Cloud secrets are encrypted and injected at runtime. They're not in your code.

---

### Option 2: Private VPS (AWS, DigitalOcean, Linode)

**Best for:** Production trading, full control

**Steps:**
```bash
# 1. Provision server (Ubuntu 22.04)
# 2. Clone repo
git clone https://github.com/yourrepo/fox_of_wallstreet.git
cd fox_of_wallstreet

# 3. Install dependencies
pip install -r requirements.txt
pip install -r apps/live_trader/requirements.txt

# 4. Set environment variables (secure method)
sudo nano /etc/systemd/system/live-trader.env
# Add:
# ALPACA_API_KEY=your_key
# ALPACA_SECRET_KEY=your_secret

# 5. Create systemd service
sudo nano /etc/systemd/system/live-trader.service
```

**Service file:**
```ini
[Unit]
Description=Fox of Wallstreet Live Trader
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/fox_of_wallstreet
EnvironmentFile=/etc/systemd/system/live-trader.env
ExecStart=/usr/bin/python -m streamlit run apps/live_trader/app.py --server.port=8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 6. Start service
sudo systemctl daemon-reload
sudo systemctl enable live-trader
sudo systemctl start live-trader

# 7. Set up reverse proxy (nginx) for HTTPS
```

**Pros:**
- ✅ Full control
- ✅ 24/7 uptime
- ✅ Can run in any region
- ✅ No resource limits

**Cons:**
- ❌ Requires sysadmin knowledge
- ❌ Costs $5-20/month
- ❌ You manage security

---

### Option 3: Docker Deployment

**Best for:** Consistent environments, easy scaling

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY apps/live_trader/requirements.txt ./apps/live_trader/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r apps/live_trader/requirements.txt

# Copy app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run
ENTRYPOINT ["streamlit", "run", "apps/live_trader/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
# Build
docker build -t fox-trader .

# Run with env vars
docker run -d \
  -e ALPACA_API_KEY=$ALPACA_API_KEY \
  -e ALPACA_SECRET_KEY=$ALPACA_SECRET_KEY \
  -e ALPACA_PAPER=true \
  -p 8501:8501 \
  --name fox-trader \
  fox-trader
```

**With docker-compose:**
```yaml
version: '3.8'
services:
  trader:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env  # Gitignored file with secrets
    restart: unless-stopped
    volumes:
      - ./data:/app/data  # Persist trade history
```

---

### Option 4: Heroku / Railway / Render

**Best for:** Quick deployment, managed hosting

**Heroku:**
```bash
# 1. Create Heroku app
heroku create fox-trader

# 2. Set config vars
heroku config:set ALPACA_API_KEY=your_key
heroku config:set ALPACA_SECRET_KEY=your_secret

# 3. Deploy
git push heroku main
```

**Railway/Render:** Similar process - connect GitHub repo, add env vars in dashboard.

---

## Demo Deployment Setup

### For Demo Purposes (Show to Others)

**Recommended Approach: Streamlit Cloud with Paper Trading**

1. **Create a demo branch:**
```bash
git checkout -b demo
```

2. **Ensure `.env` is gitignored:**
```bash
echo ".env" >> .gitignore
git rm --cached .env  # If already tracked
```

3. **Add demo-specific settings:**
```python
# config/settings_demo.py
# Override for demo
LIVE_TRADING_BUDGET = 10000  # Cap at $10k
MAX_POSITION_PCT = 0.5  # Max 50% in one position
```

4. **Push to GitHub:**
```bash
git add .
git commit -m "Demo branch"
git push origin demo
```

5. **Deploy to Streamlit Cloud:**
   - Use `demo` branch
   - Add API keys to Streamlit Secrets
   - Share the URL

6. **Monitor usage:**
   - Set Telegram notifications for trades
   - Check Alpaca dashboard for activity
   - Set daily P&L limits

---

## API Key Best Practices

### 1. **Use Paper Trading for Demos**
```bash
ALPACA_PAPER=true  # Always for demos!
```

### 2. **Rotate Keys Regularly**
- Alpaca: Dashboard → API Keys → Regenerate
- Update in deployment secrets
- Old keys automatically invalidated

### 3. **Restrict IP Addresses**
If your deployment has static IP:
- Alpaca Dashboard → Allowed IPs
- Add your server's IP
- Blocks requests from other IPs

### 4. **Set Trading Limits**
```python
# config/settings.py
LIVE_TRADING_BUDGET = 10000  # Max $10k
MAX_POSITION_PCT = 0.3  # Max 30% in one stock
MAX_DAILY_TRADES = 10  # Prevent overtrading
```

### 5. **Enable Telegram Alerts**
Get notified of every trade in real-time.

### 6. **Monitor Logs**
```bash
# Check for suspicious activity
journalctl -u live-trader -f
```

---

## Security Checklist

Before deploying:

- [ ] `.env` is in `.gitignore`
- [ ] No hardcoded API keys in code
- [ ] Using paper trading for demo
- [ ] Trading budget is capped
- [ ] Telegram alerts configured
- [ ] HTTPS enabled (for cloud deployments)
- [ ] Access logging enabled
- [ ] Regular key rotation scheduled

---

## Quick Reference: Deployment Decision Tree

```
Who needs access?
├── Just me (local) → Keep local, use .env
├── Team/internal → Streamlit Cloud + Secrets
├── Public demo → Streamlit Cloud + Paper trading
└── Production trading → VPS/Docker + systemd

What's your budget?
├── Free → Streamlit Cloud, Heroku, Railway
├── $5-10/month → DigitalOcean, Linode VPS
└── $20+/month → AWS, GCP, Azure

How technical are you?
├── Beginner → Streamlit Cloud (easiest)
├── Intermediate → Docker + VPS
└── Advanced → Kubernetes + monitoring
```

---

## Need Help?

- Streamlit Cloud: [docs.streamlit.io](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- Docker: [docs.docker.com](https://docs.docker.com)
- Alpaca API: [alpaca.markets/docs](https://alpaca.markets/docs/)
