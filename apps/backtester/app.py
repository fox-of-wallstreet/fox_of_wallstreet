"""
Backtester App - Main Entry Point

Historical analysis and model comparison tool.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Fox of Wallstreet - Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
def init_session_state():
    defaults = {
        "base_model": None,
        "compare_models": [],
        "date_range": None,
        "selected_trade": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar
with st.sidebar:
    st.title("📊 Backtester")
    st.subheader("Model Analysis & Comparison")
    
    st.divider()
    
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/01_compare.py", label="📈 Compare", icon="📈")
    st.page_link("pages/02_analyze.py", label="🔬 Analyze", icon="🔬")
    st.page_link("pages/03_explore.py", label="🔍 Explore", icon="🔍")
    st.page_link("pages/04_report.py", label="📄 Report", icon="📄")

# Main content
st.title("📊 Backtester")
st.write("Historical analysis and model comparison tool")

st.info("""
### Welcome to the Backtester!

This app helps you:
- **Compare** multiple models side-by-side
- **Analyze** individual model performance
- **Explore** trade-by-trade details
- **Generate** shareable reports

### Quick Start
1. Go to **Compare** to see models side-by-side
2. Go to **Analyze** for deep dive into one model
3. Go to **Explore** to inspect individual trades
4. Go to **Report** to export findings
""")

# Show available models
st.divider()
st.subheader("Available Models")

from shared.utils.model_discovery import list_available_models

models = list_available_models()

if models:
    import pandas as pd
    
    table_data = []
    for m in models:
        table_data.append({
            "Model": m["name"][:50],
            "Trained": m["timestamp"].strftime("%Y-%m-%d") if m["timestamp"] else "?",
            "Model": "✅" if m["has_model"] else "❌",
            "Ledger": "✅" if m["has_ledger"] else "❌",
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.success(f"Found {len(models)} models in artifacts/")
else:
    st.warning("No models found. Run training and backtesting first.")
