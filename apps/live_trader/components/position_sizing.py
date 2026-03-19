"""
Position Sizing Calculator Component

Helps users understand trade sizing and risk before executing.
"""

import streamlit as st
import pandas as pd
from typing import Dict


def calculate_position_sizing(
    available_cash: float,
    current_price: float,
    current_position: float,
    entry_price: float,
    action: int,
    action_space: str,
    cash_risk_fraction: float = 0.65,
) -> Dict:
    """
    Calculate position sizing for a trade.
    
    Returns dict with:
    - investment_amount
    - shares_to_trade
    - new_position
    - new_cash
    - position_value
    - portfolio_value
    - risk_percent
    """
    result = {
        "investment_amount": 0.0,
        "shares_to_trade": 0.0,
        "new_position": current_position,
        "new_cash": available_cash,
        "position_value": current_position * current_price,
        "portfolio_value": available_cash + (current_position * current_price),
        "risk_percent": 0.0,
        "action_type": "HOLD",
    }
    
    if action_space == "discrete_3":
        if action == 1:  # Buy All
            result["action_type"] = "BUY"
            result["investment_amount"] = available_cash * cash_risk_fraction
            result["shares_to_trade"] = result["investment_amount"] / current_price if current_price > 0 else 0
            
            # Weighted average entry price
            old_cost = current_position * entry_price
            new_cost = result["investment_amount"]
            total_shares = current_position + result["shares_to_trade"]
            
            result["new_position"] = total_shares
            result["new_cash"] = available_cash - result["investment_amount"]
            result["new_entry_price"] = (old_cost + new_cost) / total_shares if total_shares > 0 else 0
            
        elif action == 0:  # Sell All
            result["action_type"] = "SELL"
            result["shares_to_trade"] = current_position
            result["investment_amount"] = current_position * current_price
            result["new_position"] = 0.0
            result["new_cash"] = available_cash + result["investment_amount"]
            result["new_entry_price"] = 0.0
            
        else:  # Hold
            result["action_type"] = "HOLD"
            
    else:  # discrete_5
        if action == 4:  # Buy 100%
            result["action_type"] = "BUY"
            result["investment_amount"] = available_cash * cash_risk_fraction
            result["shares_to_trade"] = result["investment_amount"] / current_price if current_price > 0 else 0
            
            old_cost = current_position * entry_price
            new_cost = result["investment_amount"]
            total_shares = current_position + result["shares_to_trade"]
            
            result["new_position"] = total_shares
            result["new_cash"] = available_cash - result["investment_amount"]
            result["new_entry_price"] = (old_cost + new_cost) / total_shares if total_shares > 0 else 0
            
        elif action == 3:  # Buy 50%
            result["action_type"] = "BUY"
            result["investment_amount"] = available_cash * 0.5 * cash_risk_fraction
            result["shares_to_trade"] = result["investment_amount"] / current_price if current_price > 0 else 0
            
            old_cost = current_position * entry_price
            new_cost = result["investment_amount"]
            total_shares = current_position + result["shares_to_trade"]
            
            result["new_position"] = total_shares
            result["new_cash"] = available_cash - result["investment_amount"]
            result["new_entry_price"] = (old_cost + new_cost) / total_shares if total_shares > 0 else 0
            
        elif action == 0:  # Sell 100%
            result["action_type"] = "SELL"
            result["shares_to_trade"] = current_position
            result["investment_amount"] = current_position * current_price
            result["new_position"] = 0.0
            result["new_cash"] = available_cash + result["investment_amount"]
            result["new_entry_price"] = 0.0
            
        elif action == 1:  # Sell 50%
            result["action_type"] = "SELL"
            result["shares_to_trade"] = current_position * 0.5
            result["investment_amount"] = result["shares_to_trade"] * current_price
            result["new_position"] = current_position * 0.5
            result["new_cash"] = available_cash + result["investment_amount"]
            result["new_entry_price"] = entry_price  # Unchanged
            
        else:  # Hold
            result["action_type"] = "HOLD"
    
    # Calculate risk % of portfolio
    if result["portfolio_value"] > 0:
        result["risk_percent"] = (result["investment_amount"] / result["portfolio_value"]) * 100
    
    return result


def render_position_sizing_card(
    action: int,
    action_name: str,
    available_cash: float,
    current_price: float,
    current_position: float,
    entry_price: float,
    action_space: str,
):
    """
    Render position sizing calculator card.
    
    **How to use:**
    - Shows exactly what the trade will do
    - Calculates shares, cash impact, new position
    - Helps understand risk before executing
    
    **How to interpret:**
    - Investment: How much cash will be used
    - Shares: How many shares traded
    - New Position: Your holdings after trade
    - Risk %: What % of portfolio is at risk
    """
    from config import settings
    
    st.subheader("📊 Trade Preview")
    
    sizing = calculate_position_sizing(
        available_cash=available_cash,
        current_price=current_price,
        current_position=current_position,
        entry_price=entry_price,
        action=action,
        action_space=action_space,
        cash_risk_fraction=settings.CASH_RISK_FRACTION,
    )
    
    # Current state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Cash", f"${available_cash:,.2f}")
    with col2:
        st.metric("Current Position", f"{current_position:.4f} shares")
    with col3:
        position_value = current_position * current_price
        st.metric("Position Value", f"${position_value:,.2f}")
    
    st.divider()
    
    # Trade details
    if sizing["action_type"] == "HOLD":
        st.info("🟡 HOLD - No trade will be executed")
        return sizing
    
    st.write(f"**Action:** {action_name}")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            "Investment",
            f"${sizing['investment_amount']:,.2f}",
            help="Cash that will be used for this trade"
        )
    with cols[1]:
        st.metric(
            "Shares",
            f"{sizing['shares_to_trade']:.4f}",
            help="Number of shares to buy/sell"
        )
    with cols[2]:
        if sizing["action_type"] == "BUY":
            delta = sizing['new_position'] - current_position
            st.metric(
                "New Position",
                f"{sizing['new_position']:.4f}",
                f"+{delta:.4f}",
                help="Your total shares after this trade"
            )
        else:
            delta = sizing['shares_to_trade']
            st.metric(
                "Remaining",
                f"{sizing['new_position']:.4f}",
                f"-{delta:.4f}",
                help="Shares remaining after selling"
            )
    with cols[3]:
        st.metric(
            "Risk %",
            f"{sizing['risk_percent']:.1f}%",
            help="Percentage of portfolio at risk"
        )
    
    # Post-trade state
    st.divider()
    st.write("**After Trade:**")
    
    post_cols = st.columns(3)
    with post_cols[0]:
        cash_delta = sizing['new_cash'] - available_cash
        st.metric(
            "Remaining Cash",
            f"${sizing['new_cash']:,.2f}",
            f"${cash_delta:+,.2f}"
        )
    with post_cols[1]:
        if sizing['new_position'] > 0:
            st.metric(
                "Avg Entry Price",
                f"${sizing.get('new_entry_price', entry_price):.2f}",
                help="Weighted average cost per share"
            )
    with post_cols[2]:
        new_portfolio_value = sizing['new_cash'] + (sizing['new_position'] * current_price)
        st.metric(
            "Portfolio Value",
            f"${new_portfolio_value:,.2f}",
            help="Total value (cash + position)"
        )
    
    return sizing


def render_position_size_calculator():
    """
    Standalone position size calculator (for manual planning).
    
    **How to use:**
    - Enter your portfolio details
    - Select action
    - See calculated trade size
    - Use for planning before loading model
    """
    st.subheader("🧮 Position Size Calculator")
    
    with st.expander("Manual Calculator", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            account_size = st.number_input(
                "Account Size ($)",
                min_value=1000.0,
                value=10000.0,
                step=1000.0,
            )
            current_price = st.number_input(
                "Stock Price ($)",
                min_value=1.0,
                value=250.0,
                step=1.0,
            )
        
        with col2:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=1,
                max_value=100,
                value=65,
                help="Percentage of available cash to use",
            )
            action_type = st.selectbox(
                "Action",
                options=["Buy 100%", "Buy 50%", "Sell 50%", "Sell 100%", "Hold"],
            )
        
        if st.button("Calculate", type="primary"):
            if action_type == "Hold":
                st.info("No trade - position unchanged")
            else:
                is_buy = "Buy" in action_type
                fraction = 1.0 if "100%" in action_type else 0.5
                
                investment = account_size * (risk_per_trade / 100) * fraction
                shares = investment / current_price
                
                st.success(f"**Trade Details:**")
                st.write(f"- Investment: ${investment:,.2f}")
                st.write(f"- Shares: {shares:.4f}")
                st.write(f"- Remaining Cash: ${account_size - investment:,.2f}")
                
                if is_buy:
                    st.write(f"- New Position: {shares:.4f} shares")
                else:
                    st.write(f"- Proceeds: ${investment:,.2f}")
