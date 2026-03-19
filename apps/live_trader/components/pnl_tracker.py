"""
P&L Tracker Component

Tracks realized and unrealized profit/loss like Trading 212.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List


def calculate_unrealized_pnl(
    current_position: float,
    entry_price: float,
    current_price: float,
) -> Dict:
    """
    Calculate unrealized P&L for open position.
    
    **How to interpret:**
    - Unrealized = Paper gain/loss on open position
    - Changes with every price tick
    - Becomes "Realized" when you sell
    
    Returns:
    - unrealized_dollar: $ gain/loss
    - unrealized_pct: % gain/loss
    - break_even_price: Price needed to break even
    """
    if current_position <= 0 or entry_price <= 0:
        return {
            "unrealized_dollar": 0.0,
            "unrealized_pct": 0.0,
            "break_even_price": 0.0,
            "is_profit": False,
        }
    
    position_cost = current_position * entry_price
    position_value = current_position * current_price
    
    unrealized_dollar = position_value - position_cost
    unrealized_pct = (unrealized_dollar / position_cost) * 100
    
    return {
        "unrealized_dollar": unrealized_dollar,
        "unrealized_pct": unrealized_pct,
        "break_even_price": entry_price,  # For simple positions
        "is_profit": unrealized_dollar >= 0,
        "position_cost": position_cost,
        "position_value": position_value,
    }


def calculate_realized_pnl(trade_history: List[Dict]) -> Dict:
    """
    Calculate realized P&L from completed trades.
    
    **How to interpret:**
    - Realized = Actual cash gain/loss from closed trades
    - Locked in when you sell
    - Cumulative total of all completed round-trips
    
    Args:
        trade_history: List of completed trades with entry/exit info
        
    Returns:
    - total_realized_dollar
    - total_realized_pct
    - winning_trades
    - losing_trades
    - win_rate
    """
    if not trade_history:
        return {
            "total_realized_dollar": 0.0,
            "total_realized_pct": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
        }
    
    total_pnl = 0.0
    winning = 0
    losing = 0
    wins = []
    losses = []
    
    for trade in trade_history:
        pnl = trade.get("realized_pnl", 0.0)
        total_pnl += pnl
        
        if pnl > 0:
            winning += 1
            wins.append(pnl)
        elif pnl < 0:
            losing += 1
            losses.append(pnl)
    
    total_trades = winning + losing
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
    
    return {
        "total_realized_dollar": total_pnl,
        "total_realized_pct": 0.0,  # Would need initial capital reference
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": win_rate,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "total_trades": total_trades,
    }


def render_pnl_dashboard(
    portfolio: Dict,
    trade_history: List[Dict],
    current_price: float,
    symbol: str = "TSLA",
):
    """
    Render Trading 212-style P&L dashboard.
    
    **How to use:**
    - Shows at-a-glance profit/loss summary
    - Real-time unrealized P&L updates with price
    - Cumulative realized P&L from closed trades
    
    **How to interpret:**
    - 🔵 Unrealized: Paper profit/loss on open position
    - 🟢 Realized: Actual cash profit from closed trades
    - Win Rate: % of trades that were profitable
    """
    st.subheader("💰 Profit & Loss")
    
    # Get P&L calculations
    unrealized = calculate_unrealized_pnl(
        portfolio.get("position", 0.0),
        portfolio.get("entry_price", 0.0),
        current_price,
    )
    
    realized = calculate_realized_pnl(trade_history)
    
    # Main P&L Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Total Return (Realized + Unrealized)
        total_pnl_dollar = realized["total_realized_dollar"] + unrealized["unrealized_dollar"]
        
        st.metric(
            label="Total Return",
            value=f"${total_pnl_dollar:+,.2f}",
            delta=None,
        )
        st.caption("Realized + Unrealized")
    
    with col2:
        # Unrealized P&L (current position)
        if portfolio.get("position", 0) > 0:
            color = "🟢" if unrealized["is_profit"] else "🔴"
            st.metric(
                label=f"{color} Unrealized P&L",
                value=f"${unrealized['unrealized_dollar']:+,.2f}",
                delta=f"{unrealized['unrealized_pct']:+.2f}%",
                delta_color="normal" if unrealized["is_profit"] else "inverse",
            )
            st.caption(f"Open position: {portfolio['position']:.4f} {symbol}")
        else:
            st.metric(
                label="🔵 Unrealized P&L",
                value="$0.00",
            )
            st.caption("No open position")
    
    with col3:
        # Realized P&L (closed trades)
        is_profit = realized["total_realized_dollar"] >= 0
        color = "🟢" if is_profit else "🔴"
        st.metric(
            label=f"{color} Realized P&L",
            value=f"${realized['total_realized_dollar']:+,.2f}",
        )
        st.caption(f"From {realized['total_trades']} closed trades")
    
    st.divider()
    
    # Position Details (if holding)
    if portfolio.get("position", 0) > 0:
        st.write("**Position Details:**")
        
        pos_cols = st.columns(4)
        with pos_cols[0]:
            st.metric("Shares", f"{portfolio['position']:.4f}")
        with pos_cols[1]:
            st.metric("Entry Price", f"${portfolio['entry_price']:.2f}")
        with pos_cols[2]:
            st.metric("Current Price", f"${current_price:.2f}")
        with pos_cols[3]:
            st.metric("Position Value", f"${unrealized['position_value']:,.2f}")
        
        # Break-even info
        st.caption(
            f"Break-even at ${portfolio['entry_price']:.2f} | "
            f"Position cost: ${unrealized['position_cost']:,.2f}"
        )
    
    st.divider()
    
    # Trading Statistics
    if realized["total_trades"] > 0:
        st.write("**Trading Statistics:**")
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric(
                "Win Rate",
                f"{realized['win_rate']:.1f}%",
                help="Percentage of profitable trades"
            )
        with stat_cols[1]:
            st.metric(
                "Winning Trades",
                realized["winning_trades"],
            )
        with stat_cols[2]:
            st.metric(
                "Losing Trades", 
                realized["losing_trades"],
            )
        with stat_cols[3]:
            win_loss_ratio = (
                abs(realized["avg_win"] / realized["avg_loss"]) 
                if realized["avg_loss"] != 0 else 0
            )
            st.metric(
                "Avg Win/Loss Ratio",
                f"{win_loss_ratio:.2f}",
                help="Average winner vs average loser"
            )
        
        # Trade breakdown
        with st.expander("Trade History Detail"):
            if trade_history:
                df = pd.DataFrame(trade_history)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No completed trades yet")


def render_pnl_tracker_card(
    portfolio: Dict,
    current_price: float,
    symbol: str = "TSLA",
):
    """
    Compact P&L card for sidebar or header.
    
    **How to use:**
    - Quick view of current profit/loss
    - Updates in real-time with price changes
    """
    unrealized = calculate_unrealized_pnl(
        portfolio.get("position", 0.0),
        portfolio.get("entry_price", 0.0),
        current_price,
    )
    
    position_value = portfolio.get("position", 0) * current_price
    total_value = portfolio.get("cash", 0) + position_value
    
    col1, col2 = st.columns(2)
    
    with col1:
        if portfolio.get("position", 0) > 0:
            color = "🟢" if unrealized["is_profit"] else "🔴"
            st.write(
                f"{color} P&L: **${unrealized['unrealized_dollar']:+.2f}** "
                f"({unrealized['unrealized_pct']:+.2f}%)"
            )
        else:
            st.write("🔵 No position")
    
    with col2:
        st.write(f"💎 Total: **${total_value:,.2f}**")


def add_trade_to_history(
    trade_history: List[Dict],
    action: str,
    entry_price: float,
    exit_price: float,
    shares: float,
    timestamp: datetime = None,
) -> List[Dict]:
    """
    Add a completed trade to history.
    
    **How to use:**
    - Call this when a position is closed
    - Calculates realized P&L automatically
    
    Args:
        trade_history: Existing list of trades
        action: "BUY" or "SELL"
        entry_price: Average entry price
        exit_price: Exit price
        shares: Number of shares
        timestamp: Trade close time
        
    Returns:
        Updated trade_history list
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Calculate realized P&L
    realized_pnl = (exit_price - entry_price) * shares
    realized_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    
    trade = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "action": action,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "realized_pnl": realized_pnl,
        "realized_pct": realized_pct,
    }
    
    trade_history.append(trade)
    return trade_history
