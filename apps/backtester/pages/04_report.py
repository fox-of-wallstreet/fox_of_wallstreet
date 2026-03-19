"""
Report Page - Generate and export reports
"""

import streamlit as st

st.set_page_config(
    page_title="Generate Report",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Report Generation")

st.info("Generate shareable reports from your analysis")

st.divider()

st.subheader("Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    st.button("📊 Export CSV", use_container_width=True)
    st.caption("Raw trade data")

with col2:
    st.button("📈 Export PDF", use_container_width=True)
    st.caption("Formatted report with charts")

with col3:
    st.button("🌐 Export HTML", use_container_width=True)
    st.caption("Interactive web report")

st.divider()

st.subheader("Report Preview")

st.write("""
### Model Performance Report

**Model:** ppo_TSLA_1h_discrete_5_news_macro_time_20260318

**Summary:**
- Total Return: +12.4%
- Sharpe Ratio: 1.82
- Max Drawdown: -8.2%
- Win Rate: 58%

**Key Insights:**
- Model outperformed buy-and-hold by 9.2%
- Best performance in trending markets
- Struggled during high volatility periods
- Average hold time: 18 bars

**Trade Distribution:**
- Winning trades: 26 (58%)
- Losing trades: 19 (42%)
- Average winner: +$89
- Average loser: -$34

**Recommendations:**
- Consider wider stop loss during high volatility
- Model shows strong momentum capture
- Suitable for sideways to trending markets
""")

st.info("This is a preview. Real implementation will generate dynamic reports.")
