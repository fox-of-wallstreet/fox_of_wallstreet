"""
Analyze Page - Deep dive into single model
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analyze Model",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Model Analysis")

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

# Metrics grid
st.subheader("Performance Summary")

metrics_cols = st.columns(4)
metrics_data = [
    ("Total Return", "+12.4%"),
    ("Sharpe Ratio", "1.82"),
    ("Max Drawdown", "-8.2%"),
    ("Win Rate", "58%"),
    ("Profit Factor", "1.45"),
    ("Expectancy", "$23.40"),
    ("Avg Trade", "$45.20"),
    ("Total Trades", "45"),
]

for i, (label, value) in enumerate(metrics_data):
    with metrics_cols[i % 4]:
        st.metric(label, value)

st.divider()

# Charts
st.subheader("Equity Curve")
st.info("Interactive equity curve with trade markers would appear here")

st.subheader("Drawdown Analysis")
st.info("Underwater curve showing drawdown periods")

st.subheader("Monthly Returns")
st.info("Calendar heatmap of monthly returns")
