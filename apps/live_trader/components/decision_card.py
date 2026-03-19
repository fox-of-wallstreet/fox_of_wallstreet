"""
AI Decision Card Component

Displays the AI's trading decision with confidence, key features, and action buttons.
"""

import streamlit as st
import numpy as np


def get_action_color(action_name: str) -> str:
    """Get color for action type."""
    if "BUY" in action_name:
        return "green"
    elif "SELL" in action_name:
        return "red"
    else:
        return "gray"


def get_action_emoji(action_name: str) -> str:
    """Get emoji for action type."""
    if "BUY" in action_name:
        return "🟢"
    elif "SELL" in action_name:
        return "🔴"
    else:
        return "🟡"


def render_decision_card(
    action: int,
    action_name: str,
    confidence: float,
    latest_price: float,
    feature_highlights: list,
    portfolio_value: float = 10000.0,
):
    """
    Render the AI decision card.
    
    Args:
        action: Action index (0-4)
        action_name: Human-readable action name
        confidence: Confidence score 0-100
        latest_price: Current stock price
        feature_highlights: List of (feature, value, interpretation) tuples
        portfolio_value: Current portfolio value
    """
    color = get_action_color(action_name)
    emoji = get_action_emoji(action_name)
    
    # Main decision box
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {'#1a5f1a' if color == 'green' else '#5f1a1a' if color == 'red' else '#333'} 0%, {'#2d8a2d' if color == 'green' else '#8a2d2d' if color == 'red' else '#555'} 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h2 style="color: white; margin: 0; text-align: center;">
            {emoji} AI RECOMMENDATION: {action_name}
        </h2>
        <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 10px 0;">
            Confidence: {confidence:.1f}% | Price: ${latest_price:.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence meter
    st.progress(confidence / 100.0, text=f"Confidence: {confidence:.1f}%")
    
    if confidence < 50:
        st.warning("⚠️ Low confidence - AI is uncertain about this decision")
    elif confidence > 80:
        st.success("✅ High confidence - AI is very certain")
    
    # Feature highlights
    if feature_highlights:
        st.subheader("Key Signals")
        
        for feature, value, interpretation in feature_highlights:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{feature}**")
            with col2:
                st.write(f"{value:.3f}")
            with col3:
                st.caption(interpretation)


def render_action_buttons(
    action_space: str,
    mode: str,
    on_execute=None,
    on_override=None,
):
    """
    Render action buttons based on action space.
    
    Args:
        action_space: "discrete_3" or "discrete_5"
        mode: "simulate", "secure", or "autopilot"
        on_execute: Callback for execute button
        on_override: Callback for override button
    """
    st.subheader("Execute Action")
    
    # Mode indicator
    mode_colors = {
        "simulate": "🔍",
        "secure": "🛡️",
        "autopilot": "🤖",
    }
    st.write(f"**Mode:** {mode_colors.get(mode, '❓')} {mode.upper()}")
    
    if mode == "simulate":
        st.info("📝 SIMULATE mode: No real orders will be placed")
    elif mode == "secure":
        st.warning("🛡️ SECURE mode: You must confirm each trade")
    elif mode == "autopilot":
        st.error("🤖 AUTOPILOT mode: AI will execute trades automatically!")
    
    # Action buttons
    if action_space == "discrete_3":
        cols = st.columns(3)
        
        with cols[0]:
            if st.button("🔴 SELL ALL", use_container_width=True, type="secondary"):
                if on_execute:
                    on_execute(0)
                    
        with cols[1]:
            if st.button("🟡 HOLD", use_container_width=True, type="secondary"):
                if on_execute:
                    on_execute(2)
                    
        with cols[2]:
            if st.button("🟢 BUY ALL", use_container_width=True, type="primary"):
                if on_execute:
                    on_execute(1)
    
    else:  # discrete_5
        cols = st.columns(5)
        
        with cols[0]:
            if st.button("🔴 SELL 100%", use_container_width=True):
                if on_execute:
                    on_execute(0)
                    
        with cols[1]:
            if st.button("🟠 SELL 50%", use_container_width=True):
                if on_execute:
                    on_execute(1)
                    
        with cols[2]:
            if st.button("🟡 HOLD", use_container_width=True):
                if on_execute:
                    on_execute(2)
                    
        with cols[3]:
            if st.button("🟢 BUY 50%", use_container_width=True, type="primary"):
                if on_execute:
                    on_execute(3)
                    
        with cols[4]:
            if st.button("🔵 BUY 100%", use_container_width=True, type="primary"):
                if on_execute:
                    on_execute(4)
    
    # Override option
    st.divider()
    
    with st.expander("⚠️ Manual Override"):
        st.write("Force a different action than AI recommendation:")
        
        if action_space == "discrete_3":
            override_action = st.selectbox(
                "Select action:",
                options=[(0, "SELL ALL"), (1, "BUY ALL"), (2, "HOLD")],
                format_func=lambda x: x[1],
            )
        else:
            override_action = st.selectbox(
                "Select action:",
                options=[
                    (0, "SELL 100%"),
                    (1, "SELL 50%"),
                    (2, "HOLD"),
                    (3, "BUY 50%"),
                    (4, "BUY 100%"),
                ],
                format_func=lambda x: x[1],
            )
        
        if st.button("Execute Override", type="secondary"):
            if on_override:
                on_override(override_action[0])
                st.warning(f"Manual override executed: {override_action[1]}")


def render_portfolio_card(
    cash: float,
    position: float,
    entry_price: float,
    latest_price: float,
    symbol: str = "TSLA",
):
    """
    Render portfolio status card.
    """
    position_value = position * latest_price
    portfolio_value = cash + position_value
    
    if position > 0 and entry_price > 0:
        unrealized_pnl = (latest_price - entry_price) / entry_price * 100
        pnl_dollar = (latest_price - entry_price) * position
    else:
        unrealized_pnl = 0.0
        pnl_dollar = 0.0
    
    st.subheader("Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Cash",
            value=f"${cash:,.2f}",
        )
    
    with col2:
        st.metric(
            label=f"📈 Position ({symbol})",
            value=f"{position:.4f} shares" if position > 0 else "None",
            delta=f"@${position_value:,.2f}" if position > 0 else None,
        )
    
    with col3:
        st.metric(
            label="💎 Total Value",
            value=f"${portfolio_value:,.2f}",
        )
    
    with col4:
        if position > 0:
            st.metric(
                label="📊 Unrealized P&L",
                value=f"${pnl_dollar:,.2f}",
                delta=f"{unrealized_pnl:+.2f}%",
                delta_color="normal" if unrealized_pnl >= 0 else "inverse",
            )
        else:
            st.metric(
                label="📊 Unrealized P&L",
                value="-",
            )
    
    if position > 0:
        st.caption(f"Entry price: ${entry_price:.2f} | Current: ${latest_price:.2f}")
