"""
Trade Dashboard - Main trading interface with AI inference
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st

# Password protection
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.auth import require_auth, show_logout_button
require_auth()
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from config import settings

st.set_page_config(
    page_title="Trade Dashboard",
    page_icon="📊",
    layout="wide",
)

# Auto-refresh for autonomous mode
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Import components
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from components.decision_card import render_decision_card, render_action_buttons, render_portfolio_card
from components.position_sizing import render_position_sizing_card, render_position_size_calculator
from components.pnl_tracker import render_pnl_dashboard, render_pnl_tracker_card, add_trade_to_history
from utils.feature_fetcher import (
    run_ai_inference,
    action_to_name,
    get_feature_highlights,
    fetch_recent_prices,
)
from utils.alpaca_client import AlpacaTrader
from utils.telegram import get_notifier

# Initialize Alpaca client (cached per session)
@st.cache_resource
def get_alpaca_client():
    """Get or create Alpaca trading client."""
    return AlpacaTrader()

alpaca = get_alpaca_client()
telegram = get_notifier()

st.title("📊 Trade Dashboard")

# Check if model is loaded
if st.session_state.get("loaded_model") is None:
    st.error("⚠️ No model loaded. Please go to Models page first.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Go to Models", type="primary", use_container_width=True):
            st.switch_page("pages/02_models.py")
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("app.py")
    st.stop()

# Initialize portfolio state if not present
if "portfolio" not in st.session_state or st.session_state["portfolio"] is None:
    st.session_state["portfolio"] = {
        "cash": settings.INITIAL_BALANCE,
        "position": 0.0,
        "entry_price": 0.0,
        "last_action": 0,
    }

# Get model info
model_info = st.session_state.get("model_info", {})
action_space = model_info.get("action_space", "discrete_3")

# Get symbol from model metadata (not settings!)
trading_symbol = model_info.get("symbol", settings.SYMBOL)

# Show warning if model symbol differs from settings
if trading_symbol != settings.SYMBOL:
    st.warning(
        f"⚠️ Model trained on **{trading_symbol}**, but settings has **{settings.SYMBOL}**. "
        f"Using **{trading_symbol}** for price fetching."
    )

# Auto-refresh settings - Mode-specific defaults
st.sidebar.divider()
st.sidebar.subheader("🤖 Auto-Trading")

# Mode-specific defaults (Industry best practices)
MODE_REFRESH_DEFAULTS = {
    "simulate": {"default": 30, "min": 10, "max": 300, "step": 10},    # Fast for demo
    "secure": {"default": 60, "min": 30, "max": 600, "step": 30},     # Balanced
    "autopilot": {"default": 300, "min": 60, "max": 3600, "step": 60}, # Conservative
}

current_mode = st.session_state.get("trading_mode", "simulate")
mode_config = MODE_REFRESH_DEFAULTS[current_mode]

# Mode-specific help text
mode_help = {
    "simulate": "⚡ Fast refresh for testing (no risk)",
    "secure": "⏱️ Standard refresh - you'll confirm each trade",
    "autopilot": "🛡️ Conservative refresh - careful with real money!",
}

if HAS_AUTOREFRESH:
    auto_refresh_enabled = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=st.session_state.get("auto_refresh", False),
        help=f"Mode: {current_mode.upper()}. {mode_help[current_mode]}"
    )
    st.session_state["auto_refresh"] = auto_refresh_enabled
    
    if auto_refresh_enabled:
        # Use mode-specific slider config
        refresh_key = f"refresh_interval_{current_mode}"
        current_interval = st.session_state.get(refresh_key, mode_config["default"])
        
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=mode_config["min"],
            max_value=mode_config["max"],
            value=current_interval,
            step=mode_config["step"],
            key=refresh_key,
        )
        
        # Set up auto-refresh
        st_autorefresh(interval=refresh_interval * 1000, key=f"trade_autorefresh_{current_mode}")
        
        # Show mode-specific guidance
        if current_mode == "autopilot":
            st.sidebar.warning(f"⏱️ Auto-refresh: {refresh_interval}s\n🚨 Real trades will execute!")
        elif current_mode == "secure":
            st.sidebar.info(f"⏱️ Auto-refresh: {refresh_interval}s\n✋ You'll confirm each trade")
        else:
            st.sidebar.info(f"⏱️ Auto-refresh: {refresh_interval}s (simulate mode)")
        
        # Auto-run AI if enabled and we haven't run recently
        if st.session_state.get("auto_refresh") and "loaded_model" in st.session_state:
            last_run = st.session_state.get("last_auto_run", datetime.min)
            time_since_run = (datetime.now() - last_run).total_seconds()
            
            if time_since_run > refresh_interval * 0.8:  # 80% of interval
                st.session_state["auto_run_triggered"] = True
else:
    st.sidebar.info("ℹ️ Install streamlit-autorefresh for auto-trading:\n`pip install streamlit-autorefresh`")

# Logout button
show_logout_button()

# Main layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Market Snapshot")
    
    # Fetch current price
    @st.cache_data(ttl=60)
    def get_price_data(symbol, timeframe):
        try:
            data = fetch_recent_prices(symbol, timeframe, lookback_days=5)
            return data
        except Exception as e:
            st.error(f"Error fetching price: {e}")
            return None
    
    price_data = get_price_data(trading_symbol, settings.TIMEFRAME)
    
    if price_data is not None:
        latest_price = float(price_data['Close'].iloc[-1])
        prev_price = float(price_data['Close'].iloc[-2]) if len(price_data) > 1 else latest_price
        change = latest_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
        
        # Price display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"{trading_symbol} Price",
                value=f"${latest_price:.2f}",
                delta=f"${change:+.2f} ({change_pct:+.2f}%)",
            )
        with col2:
            st.metric(
                label="Last Updated",
                value=datetime.now().strftime("%H:%M:%S"),
            )
        with col3:
            st.metric(
                label="Timeframe",
                value=settings.TIMEFRAME,
            )
        
        # Price chart
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data['Date'],
            y=price_data['Close'],
            mode='lines',
            name='Close',
            line=dict(color='#00ff88', width=2),
        ))
        
        # Add entry price line if in position
        portfolio = st.session_state["portfolio"]
        if portfolio["position"] > 0 and portfolio["entry_price"] > 0:
            fig.add_hline(
                y=portfolio["entry_price"],
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Entry: ${portfolio['entry_price']:.2f}",
            )
        
        fig.update_layout(
            title=f"{trading_symbol} Price",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch price data")
        latest_price = 0.0

with right_col:
    st.subheader("AI Decision Center")
    
    # Model info
    st.write(f"**Active Model:** `{model_info.get('name', 'Unknown')[:25]}...`")
    st.write(f"**Action Space:** `{action_space}`")
    st.write(f"**Trading Mode:** `{st.session_state.get('trading_mode', 'simulate').upper()}`")
    
    st.divider()
    
    # Run AI Analysis button
    run_analysis = st.button("🧠 Run AI Analysis", type="primary", use_container_width=True)
    
    # Check for auto-run trigger
    auto_run_triggered = st.session_state.pop("auto_run_triggered", False)
    
    if run_analysis or auto_run_triggered:
        with st.spinner("🧠 AI is analyzing market conditions..."):
            try:
                # Build portfolio features
                portfolio = st.session_state["portfolio"]
                portfolio_value = portfolio["cash"] + (portfolio["position"] * latest_price)
                
                cash_ratio = portfolio["cash"] / (settings.INITIAL_BALANCE + 1e-8)
                position_size = (portfolio["position"] * latest_price) / (settings.INITIAL_BALANCE + 1e-8)
                inventory_fraction = (portfolio["position"] * latest_price) / (portfolio_value + 1e-8) if portfolio_value > 0 else 0.0
                
                if portfolio["position"] > 0 and portfolio["entry_price"] > 0:
                    unrealized_pnl = (latest_price - portfolio["entry_price"]) / portfolio["entry_price"]
                else:
                    unrealized_pnl = 0.0
                
                max_action = 4 if action_space == "discrete_5" else 2
                last_action_norm = portfolio["last_action"] / max_action
                
                portfolio_features = np.array([
                    cash_ratio,
                    position_size,
                    inventory_fraction,
                    unrealized_pnl,
                    last_action_norm,
                ], dtype=np.float32)
                
                # Run inference
                result = run_ai_inference(
                    model=st.session_state["loaded_model"],
                    scaler=st.session_state["scaler"],
                    symbol=trading_symbol,
                    timeframe=settings.TIMEFRAME,
                    portfolio_features=portfolio_features,
                    n_stack=settings.N_STACK,
                )
                
                # Store result
                st.session_state["last_ai_decision"] = result
                
                # Log activity
                log_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": f"AI Analysis: {action_to_name(result['action'], action_space)} (conf: {result['confidence']:.1f}%)",
                }
                if "activity_log" not in st.session_state:
                    st.session_state["activity_log"] = []
                st.session_state["activity_log"].append(log_entry)
                
                st.success("Analysis complete!")
                
                # Track auto-run timestamp
                st.session_state["last_auto_run"] = datetime.now()
                
                # In auto-refresh mode, show badge
                if auto_run_triggered:
                    st.info("🤖 Auto-run triggered this analysis")
                
            except Exception as e:
                st.error(f"Inference failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.divider()
    
    # Display AI decision if available
    if st.session_state.get("last_ai_decision"):
        decision = st.session_state["last_ai_decision"]
        
        # Get feature highlights
        highlights = get_feature_highlights(decision.get("features", {}), top_n=5)
        
        render_decision_card(
            action=decision["action"],
            action_name=action_to_name(decision["action"], action_space),
            confidence=decision["confidence"],
            latest_price=decision["latest_price"],
            feature_highlights=highlights,
        )
        
        # Position Sizing Preview
        st.divider()
        sizing = render_position_sizing_card(
            action=decision["action"],
            action_name=action_to_name(decision["action"], action_space),
            available_cash=portfolio["cash"],
            current_price=decision["latest_price"],
            current_position=portfolio["position"],
            entry_price=portfolio["entry_price"],
            action_space=action_space,
        )
        
        # Store for execute handlers
        current_decision = decision
    else:
        st.info("No AI decision yet. Click 'Run AI Analysis' above.")
        current_decision = None
    
    st.divider()
    
    # Execute actions
    def on_execute(action: int, is_override: bool = False):
        """Handle action execution with Alpaca integration."""
        mode = st.session_state.get("trading_mode", "simulate")
        portfolio = st.session_state["portfolio"]
        
        action_name = action_to_name(action, action_space)
        
        # ===== SIMULATE MODE =====
        if mode == "simulate":
            # Just update virtual portfolio
            if "BUY" in action_name:
                fraction = 1.0 if "100" in action_name else 0.5
                investment = portfolio["cash"] * fraction * settings.CASH_RISK_FRACTION
                shares = investment / latest_price if latest_price > 0 else 0
                
                old_position = portfolio["position"]
                old_cost = portfolio["position"] * portfolio["entry_price"] if portfolio["position"] > 0 else 0
                
                portfolio["position"] += shares
                portfolio["cash"] -= investment
                
                if portfolio["position"] > 0:
                    portfolio["entry_price"] = (old_cost + investment) / portfolio["position"]
                
                portfolio["last_action"] = action
                
                msg = f"SIMULATED: {action_name} - {shares:.4f} shares @ ${latest_price:.2f}"
                
            elif "SELL" in action_name:
                fraction = 1.0 if "100" in action_name else 0.5
                shares_to_sell = portfolio["position"] * fraction
                proceeds = shares_to_sell * latest_price
                
                # Track realized P&L
                st.session_state["trade_history"] = add_trade_to_history(
                    trade_history=st.session_state.get("trade_history", []),
                    action=action_name,
                    entry_price=portfolio["entry_price"],
                    exit_price=latest_price,
                    shares=shares_to_sell,
                )
                
                portfolio["position"] -= shares_to_sell
                portfolio["cash"] += proceeds
                
                if portfolio["position"] <= 0.001:
                    portfolio["position"] = 0.0
                    portfolio["entry_price"] = 0.0
                
                portfolio["last_action"] = action
                
                msg = f"SIMULATED: {action_name} - {shares_to_sell:.4f} shares @ ${latest_price:.2f}"
                
            else:  # HOLD
                msg = "SIMULATED: HOLD position maintained"
            
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": msg,
            }
            st.session_state["activity_log"].append(log_entry)
            st.success(f"📝 {msg}")
            st.rerun()
            return
        
        # ===== SECURE / AUTOPILOT MODES =====
        # These require Alpaca connection
        
        if not alpaca.is_connected():
            st.error("❌ Alpaca not connected. Please add API keys in Settings page.")
            return
        
        # Step 1: Sync portfolio from Alpaca
        with st.spinner("Syncing with Alpaca..."):
            alpaca_portfolio = alpaca.get_portfolio()
            portfolio["cash"] = alpaca_portfolio["cash"]
            portfolio["position"] = alpaca_portfolio["position"]
            portfolio["entry_price"] = alpaca_portfolio["entry_price"]
        
        # Step 2: Price Verification (for BUY/SELL, not HOLD)
        if action != 2:  # Not HOLD
            with st.spinner("Verifying price..."):
                is_valid, alpaca_price, verify_msg = alpaca.verify_price(
                    latest_price, 
                    max_diff_pct=settings.SLIPPAGE_PCT * 2  # 0.1% default
                )
                
                if not is_valid:
                    st.error(f"❌ Price verification failed: {verify_msg}")
                    st.warning("Trade blocked to protect against price slippage.")
                    
                    # Log the failure
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "message": f"BLOCKED: {action_name} - {verify_msg}",
                    }
                    st.session_state["activity_log"].append(log_entry)
                    return
                else:
                    st.info(f"✅ {verify_msg}")
        
        # Step 3: SECURE mode - Show confirmation
        if mode == "secure" and not is_override:
            confirm_msg = f"""
            **Confirm Trade:**
            
            - Action: {action_name}
            - Symbol: {trading_symbol}
            - Current Price: ${latest_price:.2f}
            - Cash Available: ${portfolio['cash']:.2f}
            - Current Position: {portfolio['position']:.4f} shares
            
            Proceed with this order?
            """
            
            if not st.checkbox("✅ I confirm this trade", key=f"confirm_{action}"):
                st.info("Trade waiting for confirmation. Check the box above to proceed.")
                return
        
        # Step 4: Submit order to Alpaca
        with st.spinner(f"Submitting {action_name} order to Alpaca..."):
            result = alpaca.submit_order(
                action=action,
                action_space=action_space,
                current_price=latest_price,
                current_shares=portfolio["position"],
                available_cash=portfolio["cash"],
            )
        
        # Step 5: Handle result
        if result["success"]:
            if result["status"] == "submitted":
                st.success(f"✅ ORDER SUBMITTED: {result['message']}")
                st.info(f"Order ID: {result['order_id']}")
                
                # Send Telegram notification for real orders (SECURE/AUTOPILOT only)
                if telegram.enabled:
                    telegram.notify_order(
                        symbol=trading_symbol,
                        action=action_name,
                        quantity=result.get('quantity', 0),
                        price=latest_price,
                        mode=mode,
                    )
                
                # Update local portfolio estimate
                if "BUY" in action_name:
                    fraction = 1.0 if "100" in action_name else 0.5
                    investment = portfolio["cash"] * fraction * settings.CASH_RISK_FRACTION
                    shares = investment / latest_price
                    
                    old_cost = portfolio["position"] * portfolio["entry_price"]
                    portfolio["position"] += shares
                    portfolio["cash"] -= investment
                    if portfolio["position"] > 0:
                        portfolio["entry_price"] = (old_cost + investment) / portfolio["position"]
                    
                elif "SELL" in action_name:
                    fraction = 1.0 if "100" in action_name else 0.5
                    shares_to_sell = portfolio["position"] * fraction
                    proceeds = shares_to_sell * latest_price
                    
                    # Track realized P&L before updating position
                    st.session_state["trade_history"] = add_trade_to_history(
                        trade_history=st.session_state.get("trade_history", []),
                        action=action_name,
                        entry_price=portfolio["entry_price"],
                        exit_price=latest_price,
                        shares=shares_to_sell,
                    )
                    
                    portfolio["position"] -= shares_to_sell
                    portfolio["cash"] += proceeds
                    if portfolio["position"] <= 0.001:
                        portfolio["position"] = 0.0
                        portfolio["entry_price"] = 0.0
                
                portfolio["last_action"] = action
                
                log_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "message": f"ALPACA: {action_name} - Order {result['order_id']}",
                }
                st.session_state["activity_log"].append(log_entry)
                
            else:  # skipped
                st.info(f"ℹ️ Order skipped: {result['message']}")
        else:
            st.error(f"❌ Order failed: {result['message']}")
            
            # Send error notification
            if telegram.enabled:
                telegram.notify_error(
                    symbol=trading_symbol,
                    error_msg=result['message'],
                )
        
        st.rerun()
    
    def on_override(action: int):
        """Handle manual override - bypasses AI recommendation."""
        on_execute(action, is_override=True)
    
    # Show action buttons
    render_action_buttons(
        action_space=action_space,
        mode=st.session_state.get("trading_mode", "simulate"),
        on_execute=on_execute,
        on_override=on_override,
    )

# Portfolio section
st.divider()

portfolio = st.session_state["portfolio"]
render_portfolio_card(
    cash=portfolio["cash"],
    position=portfolio["position"],
    entry_price=portfolio["entry_price"],
    latest_price=latest_price if price_data is not None else 0,
    symbol=trading_symbol,
)

# P&L Tracker (like Trading 212)
st.divider()

# Initialize trade history if not present
if "trade_history" not in st.session_state:
    st.session_state["trade_history"] = []

render_pnl_dashboard(
    portfolio=portfolio,
    trade_history=st.session_state["trade_history"],
    current_price=latest_price if price_data is not None else 0,
    symbol=trading_symbol,
)

# Position Size Calculator (for planning)
render_position_size_calculator()

# Activity log
st.divider()
st.subheader("📜 Activity Log")

if st.session_state.get("activity_log"):
    # Show last 10 entries
    for log in reversed(st.session_state["activity_log"][-10:]):
        st.write(f"`{log['timestamp']}` - {log['message']}")
    
    if st.button("Clear Log"):
        st.session_state["activity_log"] = []
        st.rerun()
else:
    st.info("No activity yet. Run AI analysis to see events here.")
