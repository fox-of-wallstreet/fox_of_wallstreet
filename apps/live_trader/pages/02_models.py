"""
Models Page - Browse and load trained models
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import streamlit as st

# Password protection
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.auth import require_auth, show_logout_button
require_auth()

# Import components
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from components.model_selector import (
    render_model_selector,
    render_loaded_model_status,
    load_model_to_session,
)

st.set_page_config(
    page_title="Models",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Model Management")

# Two columns: selection and status
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Select a Model")
    st.write("Choose a trained model from your artifacts folder.")
    
    # Render selector
    selected_model = render_model_selector()
    
    # Handle load
    if selected_model:
        success = load_model_to_session(selected_model)
        if success:
            st.rerun()

with col2:
    st.header("Current Model")
    render_loaded_model_status()

# Model comparison section
st.divider()
st.header("📊 Available Models")

from shared.utils.model_discovery import list_available_models

models = list_available_models()

if models:
    # Create a table view
    import pandas as pd
    
    table_data = []
    for m in models:
        table_data.append({
            "Name": m["name"][:40] + "..." if len(m["name"]) > 40 else m["name"],
            "Symbol": m["parsed"].get("symbol", "?"),
            "Timeframe": m["parsed"].get("timeframe", "?"),
            "Action": m["parsed"].get("action", "?"),
            "Date": m["timestamp"].strftime("%Y-%m-%d %H:%M") if m["timestamp"] else "Unknown",
            "Model": "✅" if m["has_model"] else "❌",
            "Backtest": "✅" if m["has_ledger"] else "❌",
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No models found. Train a model first using `python scripts/train.py`")

# Help section
st.divider()
st.header("ℹ️ About Models")

st.write("""
**Model Files:**
- `model.zip` - The trained PPO model weights
- `scaler.pkl` - Feature scaler fitted on training data
- `metadata.json` - Training configuration and hyperparameters
- `backtest_ledger.csv` - Historical trades from backtest

**Compatibility:**
Models are validated against your current `config/settings.py`. 
If settings don't match, you'll see a warning but can still load the model.

**Action Spaces:**
- **discrete_3**: Simpler - Buy All, Sell All, or Hold
- **discrete_5**: More granular - Buy/Sell at 50% or 100% increments
""")
