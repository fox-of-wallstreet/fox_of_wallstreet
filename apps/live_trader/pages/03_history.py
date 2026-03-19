"""
History Page - View past trades and activity
"""

import os
import sys
import streamlit as st

# Password protection
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.auth import require_auth
require_auth()

st.set_page_config(
    page_title="History",
    page_icon="📜",
    layout="wide",
)

st.title("📜 Trading History")

st.info("This page will show:")
st.write("- List of all executed trades with timestamps")
st.write("- P&L for each trade")
st.write("- Cumulative performance chart")
st.write("- Export to CSV functionality")
st.write("- Trade annotations/notes")

# Placeholder
st.divider()
st.subheader("Recent Activity")

# Sample data structure
import pandas as pd
from datetime import datetime, timedelta

sample_data = [
    {
        "Time": datetime.now() - timedelta(hours=2),
        "Action": "BUY_50",
        "Price": "$245.30",
        "Shares": "20",
        "Value": "$4,906",
        "Status": "Filled",
    },
    {
        "Time": datetime.now() - timedelta(hours=5),
        "Action": "SELL_100",
        "Price": "$267.80",
        "Shares": "40",
        "Value": "$10,712",
        "Status": "Filled",
    },
    {
        "Time": datetime.now() - timedelta(days=1),
        "Action": "BUY_100",
        "Price": "$242.10",
        "Shares": "41",
        "Value": "$9,926",
        "Status": "Filled",
    },
]

df = pd.DataFrame(sample_data)
st.table(df)

st.caption("Note: This is sample data. Real implementation will read from activity log.")
