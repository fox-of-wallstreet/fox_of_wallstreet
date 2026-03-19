"""
Settings Page - Configure API keys and preferences
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv, set_key, find_dotenv

# Password protection
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.auth import require_auth
require_auth()

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Settings")

# Load current env
load_dotenv()

# Find .env file
env_path = find_dotenv(usecwd=True)
if not env_path:
    # Create .env in project root if not found
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("# Fox of Wallstreet Environment Variables\n")

st.header("🔑 Alpaca API Configuration")

# Detect if we're in a cloud deployment
deployment_type = "local"
if os.getenv("STREAMLIT_SERVER_ADDRESS") or os.getenv("STREAMLIT_SHARING_MODE"):
    deployment_type = "streamlit_cloud"
elif os.getenv("DYNO"):
    deployment_type = "heroku"
elif os.getenv("RAILWAY_ENVIRONMENT"):
    deployment_type = "railway"

if deployment_type != "local":
    st.success(f"🌐 Running on {deployment_type.replace('_', ' ').title()}")
    st.info("🔒 API keys are securely stored in environment variables (not in code)")

st.write("Enter your Alpaca API credentials for live trading.")
st.info("💡 You can also set these in your `.env` file in the project root.")

with st.form("alpaca_settings"):
    api_key = st.text_input(
        "API Key",
        value=os.getenv("ALPACA_API_KEY", ""),
        type="password",
        help="Your Alpaca API Key",
    )
    
    api_secret = st.text_input(
        "Secret Key",
        value=os.getenv("ALPACA_SECRET_KEY", ""),
        type="password",
        help="Your Alpaca Secret Key",
    )
    
    paper_trading = st.toggle(
        "Paper Trading Mode",
        value=os.getenv("ALPACA_PAPER", "true").lower() != "false",
        help="Enable for practice trading with fake money",
    )
    
    submitted = st.form_submit_button("💾 Save Alpaca Settings", type="primary")
    
    if submitted:
        try:
            # Save to .env file
            set_key(env_path, "ALPACA_API_KEY", api_key)
            set_key(env_path, "ALPACA_SECRET_KEY", api_secret)
            set_key(env_path, "ALPACA_PAPER", str(paper_trading).lower())
            
            st.success("✅ Alpaca settings saved to .env file!")
            st.info("🔄 Restart the app to apply changes.")
        except Exception as e:
            st.error(f"❌ Error saving settings: {e}")
            st.info("💡 You can manually edit the `.env` file in the project root.")

st.divider()

st.header("📱 Telegram Notifications (Optional)")
st.write("Get trade alerts via Telegram bot.")

with st.form("telegram_settings"):
    telegram_token = st.text_input(
        "Bot Token",
        value=os.getenv("TELEGRAM_TOKEN", ""),
        type="password",
        help="Your Telegram Bot Token from @BotFather",
    )
    
    telegram_chat = st.text_input(
        "Chat ID",
        value=os.getenv("TELEGRAM_CHAT_ID", ""),
        help="Your Telegram Chat ID (use @userinfobot to find it)",
    )
    
    st.info("📋 **How to set up Telegram notifications:**\n"
            "1. Message @BotFather to create a bot\n"
            "2. Copy the bot token here\n"
            "3. Message @userinfobot to get your Chat ID\n"
            "4. Save settings\n"
            "5. Start your bot by sending /start to it")
    
    submitted = st.form_submit_button("💾 Save Telegram Settings", type="primary")
    
    if submitted:
        try:
            set_key(env_path, "TELEGRAM_TOKEN", telegram_token)
            set_key(env_path, "TELEGRAM_CHAT_ID", telegram_chat)
            
            st.success("✅ Telegram settings saved!")
            st.info("🔄 Restart the app to apply changes.")
        except Exception as e:
            st.error(f"❌ Error saving settings: {e}")

st.divider()

st.header("📊 Current Configuration")

# Read from settings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import settings

cols = st.columns(2)

with cols[0]:
    st.subheader("Trading Parameters")
    st.write(f"**Symbol:** {settings.SYMBOL}")
    st.write(f"**Timeframe:** {settings.TIMEFRAME}")
    st.write(f"**Initial Balance:** ${settings.INITIAL_BALANCE:,.2f}")
    st.write(f"**Cash Risk Fraction:** {settings.CASH_RISK_FRACTION}")
    st.write(f"**Stop Loss:** {settings.STOP_LOSS_PCT:.1%}")
    st.write(f"**Take Profit:** {settings.TAKE_PROFIT_PCT:.1%}")

with cols[1]:
    st.subheader("Features")
    st.write(f"**News Features:** {'Enabled' if settings.USE_NEWS_FEATURES else 'Disabled'}")
    st.write(f"**Macro Features:** {'Enabled' if settings.USE_MACRO_FEATURES else 'Disabled'}")
    st.write(f"**Time Features:** {'Enabled' if settings.USE_TIME_FEATURES else 'Disabled'}")
    st.write(f"**Total Features:** {settings.EXPECTED_MARKET_FEATURES}")
    
    # Show current API status
    st.subheader("API Status")
    has_alpaca = bool(os.getenv("ALPACA_API_KEY")) and bool(os.getenv("ALPACA_SECRET_KEY"))
    has_telegram = bool(os.getenv("TELEGRAM_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID"))
    is_paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    
    if has_alpaca:
        if is_paper:
            st.success("✅ Alpaca API configured (PAPER TRADING)")
            st.info("💡 Using paper money - safe for testing!")
        else:
            st.error("🚨 Alpaca API configured (LIVE TRADING)")
            st.warning("⚠️ Real money will be used for trades!")
    else:
        st.error("❌ Alpaca API not configured")
    
    if has_telegram:
        st.success("✅ Telegram configured")
    else:
        st.info("ℹ️ Telegram not configured (optional)")
    
    # Show trading limits
    st.subheader("Risk Limits")
    st.write(f"**Trading Budget:** ${settings.LIVE_TRADING_BUDGET:,.2f}")
    st.write(f"**Cash Risk Fraction:** {settings.CASH_RISK_FRACTION:.0%}")
    st.write(f"**Max Position:** {settings.MAX_POSITION_PCT:.0%}")

st.divider()

st.header("📝 Manual .env Configuration")

st.write("Alternatively, you can edit your `.env` file directly:")

st.code("""# .env file in project root
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true

# Optional: Telegram notifications
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
""", language="bash")

st.info("After editing .env, restart the Streamlit app to apply changes.")
