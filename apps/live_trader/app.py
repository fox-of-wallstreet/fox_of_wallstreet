"""
Live Trader App - Main Entry Point

Real-time AI trading dashboard for production use.
"""

import os
import sys

# Add parent project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st

# Password protection for demo deployments
# Set DEMO_PASSWORD in .env or Streamlit Secrets to enable
# Leave empty to disable (local development)
sys.path.append(os.path.dirname(__file__))
from utils.auth import require_auth, show_logout_button
require_auth()  # Comment out to disable password protection

# Page configuration
st.set_page_config(
    page_title="Fox of Wallstreet - Live Trader",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "loaded_model": None,
        "scaler": None,
        "model_info": None,
        "trading_mode": "simulate",
        "alpaca_client": None,
        "last_ai_decision": None,
        "portfolio": None,
        "activity_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar navigation
with st.sidebar:
    st.title("🦊 Fox of Wallstreet")
    st.subheader("Live AI Trader")
    
    st.divider()
    
    # Navigation
    st.page_link("app.py", label="🚀 Trade", icon="🚀")
    st.page_link("pages/01_trade.py", label="📊 Trade Dashboard", icon="📊")
    st.page_link("pages/02_models.py", label="🧠 Models", icon="🧠")
    st.page_link("pages/03_history.py", label="📜 History", icon="📜")
    st.page_link("pages/04_settings.py", label="⚙️ Settings", icon="⚙️")
    
    st.divider()
    
    # Trading mode selector - HIGHLY VISIBLE
    st.markdown("### 🎮 Trading Mode")
    
    mode_colors = {
        "simulate": "🟢",
        "secure": "🟡", 
        "autopilot": "🔴",
    }
    
    mode = st.radio(
        "Select mode:",
        options=["simulate", "secure", "autopilot"],
        format_func=lambda x: {
            "simulate": "🔍 Simulate (no real orders)",
            "secure": "🛡️ Secure (confirm each trade)",
            "autopilot": "🤖 Autopilot (AI executes)",
        }[x],
        index=["simulate", "secure", "autopilot"].index(st.session_state["trading_mode"]),
        key="mode_selector",
        label_visibility="collapsed",
    )
    st.session_state["trading_mode"] = mode
    
    # Show current mode with color
    mode_color = mode_colors.get(mode, "⚪")
    st.markdown(f"**{mode_color} Current: {mode.upper()}**")
    
    st.divider()
    
    # Model status
    st.subheader("Model Status")
    if st.session_state["loaded_model"] is None:
        st.error("❌ No model loaded")
        st.info("Go to Models page to load a model")
    else:
        st.success(f"✅ {st.session_state['model_info'].get('name', 'Unknown')[:20]}...")
    
    st.divider()
    
    # Quick links
    st.subheader("Quick Links")
    st.link_button("📈 Open Alpaca", "https://app.alpaca.markets/")
    st.link_button("⚙️ Settings", "./settings")
    
    # Deployment info
    st.divider()
    deployment_env = os.getenv("STREAMLIT_SHARING_MODE", "local")
    if deployment_env != "local":
        st.caption(f"🌐 Cloud Deployment")
    else:
        st.caption(f"💻 Local Development")
    
    # Paper vs Live indicator
    is_paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    if not is_paper:
        st.error("🚨 LIVE TRADING MODE")
    
    # Logout button (if password protection enabled)
    show_logout_button()

# Main content
st.title("🦊 Live Trader")
st.write("Real-time AI-powered trading dashboard")

# Shared deployment warning (shown when deployed to cloud)
deployment_env = os.getenv("STREAMLIT_SHARING_MODE", "local")
is_paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"

if deployment_env != "local":
    # This is a cloud deployment - show warning
    if is_paper:
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #1e3a5f; margin-bottom: 1rem; border-left: 4px solid #4a9eff;'>
            <h4 style='margin:0; color: #4a9eff;'>🌐 SHARED DEMO ENVIRONMENT</h4>
            <p style='margin:0.5rem 0 0 0; color: #cccccc; font-size: 0.9em;'>
                This is a cloud demo. All users share the same API keys and trading account. 
                Using <b>PAPER TRADING</b> (fake money). 
                <a href="./settings" style="color: #4a9eff;">View Settings</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #4a0e0e; margin-bottom: 1rem; border-left: 4px solid #ff4a4a;'>
            <h4 style='margin:0; color: #ff4a4a;'>🚨 LIVE TRADING - SHARED ENVIRONMENT</h4>
            <p style='margin:0.5rem 0 0 0; color: #ffcccc; font-size: 0.9em;'>
                <b>WARNING:</b> This deployment uses REAL MONEY! All users share the same trading account.
                Do NOT share this URL publicly!
            </p>
        </div>
        """, unsafe_allow_html=True)

# Mode indicator banner
current_mode = st.session_state.get("trading_mode", "simulate")
mode_banners = {
    "simulate": ("🔍 SIMULATE MODE", "No real orders will be placed. Practice trading with virtual portfolio.", "info"),
    "secure": ("🛡️ SECURE MODE", "All trades require manual confirmation. Check the box to execute.", "warning"),
    "autopilot": ("🤖 AUTOPILOT MODE", "AI will execute trades automatically after price verification.", "error"),
}
badge, desc, style = mode_banners.get(current_mode, mode_banners["simulate"])

st.markdown(f"""
<div style='padding: 1rem; border-radius: 0.5rem; background-color: {"#0e4a2e" if style == "info" else "#5a4a0e" if style == "warning" else "#4a0e0e"}; margin-bottom: 1rem;'>
    <h3 style='margin:0; color: white;'>{badge}</h3>
    <p style='margin:0; color: #cccccc;'>{desc}</p>
</div>
""", unsafe_allow_html=True)

# Quick start guide
if st.session_state["loaded_model"] is None:
    st.info("""
    ### 👋 Welcome!
    
    To start trading:
    1. Go to **Models** page to load a trained AI model
    2. Configure your **Settings** (Alpaca API keys)
    3. Return to this page and click **Run AI Analysis**
    4. Review the AI's decision and execute
    
    **New?** Start in **Simulate** mode to see how the AI trades without risk.
    """)
else:
    # Show quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Trading Mode",
            value=st.session_state["trading_mode"].upper(),
        )
    
    with col2:
        st.metric(
            label="Model",
            value=st.session_state["model_info"]["name"][:15] + "..." 
            if len(st.session_state["model_info"]["name"]) > 15 
            else st.session_state["model_info"]["name"],
        )
    
    with col3:
        action_space = st.session_state["model_info"].get("action_space", "Unknown")
        st.metric(
            label="Action Space",
            value=action_space,
        )
    
    with col4:
        st.metric(
            label="Status",
            value="Ready" if st.session_state["loaded_model"] else "Not Loaded",
        )
    
    st.divider()
    
    # Quick action
    st.subheader("Quick Actions")
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("🚀 Go to Trade Dashboard", type="primary", use_container_width=True):
            st.switch_page("pages/01_trade.py")
    
    with cols[1]:
        if st.button("🧠 Switch Model", use_container_width=True):
            st.switch_page("pages/02_models.py")
    
    with cols[2]:
        if st.button("⚙️ Configure Settings", use_container_width=True):
            st.switch_page("pages/04_settings.py")
