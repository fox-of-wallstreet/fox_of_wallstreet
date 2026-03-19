"""
Explore Page - Trade-by-trade explorer
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Explore Trades",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Trade Explorer")

from shared.utils.model_discovery import list_available_models, format_model_display_name

models = list_available_models()
valid_models = [m for m in models if m["has_ledger"]]

if not valid_models:
    st.error("No models with backtest data found.")
    st.stop()

# Model selection
options = {format_model_display_name(m): m for m in valid_models}
selected_name = st.selectbox("Select Model", options=list(options.keys()))
selected_model = options[selected_name]

st.divider()

# Filters
st.subheader("Filter Trades")

col1, col2, col3 = st.columns(3)

with col1:
    action_filter = st.multiselect(
        "Action Type",
        options=["BUY_100", "BUY_50", "SELL_100", "SELL_50", "HOLD"],
        default=[],
    )

with col2:
    min_pnl = st.number_input("Min P&L ($)", value=-1000.0)

with col3:
    max_pnl = st.number_input("Max P&L ($)", value=1000.0)

# Sample trade table
st.divider()
st.subheader("Trades")

sample_trades = pd.DataFrame([
    {
        "Date": "2025-11-05",
        "Action": "BUY_50",
        "Price": "$245.30",
        "Portfolio": "$10,000",
        "P&L": "-",
    },
    {
        "Date": "2025-11-08",
        "Action": "SELL_50",
        "Price": "$267.80",
        "Portfolio": "$10,450",
        "P&L": "+$450",
    },
    {
        "Date": "2025-11-12",
        "Action": "BUY_100",
        "Price": "$242.10",
        "Portfolio": "$9,926",
        "P&L": "-",
    },
])

st.dataframe(sample_trades, use_container_width=True, hide_index=True)

st.caption("Click a trade to see details")

# Trade detail
st.divider()
st.subheader("Trade Detail")
st.info("Select a trade above to see detailed analysis including:")
st.write("- Entry/exit price chart")
st.write("- Feature values at trade time")
st.write("- Context: what was the AI thinking?")
