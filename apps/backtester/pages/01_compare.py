"""
Compare Page - Side-by-side model comparison
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Compare Models",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Comparison")

from shared.utils.model_discovery import list_available_models, format_model_display_name

models = list_available_models()
valid_models = [m for m in models if m["has_ledger"]]

if not valid_models:
    st.error("No models with backtest data found. Run backtesting first.")
    st.stop()

# Model selection
st.subheader("Select Models to Compare")

col1, col2 = st.columns(2)

with col1:
    base_options = {format_model_display_name(m): m for m in valid_models}
    base_name = st.selectbox("Base Model", options=list(base_options.keys()), index=0)
    base_model = base_options[base_name]

with col2:
    compare_options = {format_model_display_name(m): m for m in valid_models if m != base_model}
    if compare_options:
        compare_name = st.selectbox("Comparison Model", options=list(compare_options.keys()), index=0)
        compare_model = compare_options[compare_name]
    else:
        st.info("Need another model to compare")
        compare_model = None

# Load and display
if base_model:
    st.divider()
    st.subheader("Performance Metrics")
    
    # Placeholder metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Total Return", "+12.4%", "+9.2% vs Buy&Hold")
    with cols[1]:
        st.metric("Sharpe Ratio", "1.8", "+0.7 vs benchmark")
    with cols[2]:
        st.metric("Max Drawdown", "-8.2%", "Better by -6.8%")
    
    st.info("This is a scaffold. Real implementation would:")
    st.write("1. Load backtest_ledger.csv for each model")
    st.write("2. Calculate metrics (return, Sharpe, MaxDD, win rate)")
    st.write("3. Show equity curve overlay")
    st.write("4. Show drawdown comparison")
    st.write("5. Statistical significance test")
