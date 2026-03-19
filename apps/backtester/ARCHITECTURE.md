# Backtester App - Architecture Document

## Purpose
Historical analysis and model comparison tool. This app analyzes past backtest results and compares models. It does NOT train new models or execute live trades.

## Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│   (Separate - produces artifacts + backtest_ledger.csv)         │
│   • scripts/train.py → produces model                           │
│   • scripts/backtest.py → produces ledger + reports/            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ produces
┌─────────────────────────────────────────────────────────────────┐
│                    ARTIFACTS STORAGE                            │
│   • artifacts/ppo_TSLA_1h_20260318_1445/                        │
│     ├── model.zip                                               │
│     ├── scaler.pkl                                              │
│     ├── metadata.json                                           │
│     ├── backtest_ledger.csv          ← KEY INPUT                │
│     ├── backtest_summary.json                                   │
│     └── reports/                                                │
│         ├── figures/                                            │
│         │   ├── equity_vs_benchmark.png                         │
│         │   ├── actions_overlay.png                             │
│         │   └── drawdown_curve.png                              │
│         └── tables/                                             │
│             └── equity_timeseries.csv                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ consumed by
┌─────────────────────────────────────────────────────────────────┐
│  📊 BACKTESTER APP (This Application)                           │
│                                                                 │
│  Responsibilities:                                              │
│  • Browse and compare trained models                            │
│  • Visualize backtest results                                   │
│  • Analyze trade patterns                                       │
│  • Compare model vs model vs buy-and-hold                       │
│  • Generate comparison reports                                  │
│                                                                 │
│  Explicitly NOT responsible for:                                │
│  • Training new models                                          │
│  • Running new backtests (uses existing ledger)                 │
│  • Live trading                                                 │
│  • Feature engineering                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Principles

### 1. Read-Only Analysis
This app is **read-only** with respect to models and artifacts:
- Loads existing backtest_ledger.csv files
- Never modifies artifacts/
- Never trains or backtests

### 2. Multi-Model Comparison
Core value proposition: compare 2+ models side-by-side:
- Performance metrics comparison
- Equity curve overlay
- Trade timing analysis
- Feature impact comparison

### 3. Interactive Exploration
Users can dig into details:
- Click on equity curve → see what happened that day
- Filter trades by P&L, duration, feature values
- Zoom into specific time periods
- Compare trades made by different models on same day

## Data Flow Architecture

### Data Sources
```python
# Primary: backtest_ledger.csv
trade_history = pd.read_csv("artifacts/.../backtest_ledger.csv")
# Columns: Date, Action, Price, Portfolio_Value, Position_Before, Position_After

# Secondary: backtest_summary.json
summary = json.load("artifacts/.../backtest_summary.json")
# Contains: final_return, total_trades, avg_hold_time, etc.

# Tertiary: metadata.json
metadata = json.load("artifacts/.../metadata.json")
# Contains: hyperparams, feature list, action space

# For charts: equity_timeseries.csv
equity = pd.read_csv("artifacts/.../reports/tables/equity_timeseries.csv")
```

### State Management
```python
session_state = {
    # Selection State
    "base_model": str,           # artifact folder name
    "compare_models": list,      # additional models to compare
    "date_range": (start, end),  # filter period
    
    # Analysis State
    "selected_trade": dict | None,  # clicked trade for details
    "active_tab": str,              # compare / analyze / explore
    "filters": {
        "min_pnl": float,
        "max_pnl": float,
        "action_types": list,
    },
    
    # Cached Data
    "loaded_ledgers": dict,      # artifact_name → df
    "calculated_metrics": dict,  # artifact_name → metrics
}
```

## Component Hierarchy

```
app.py (entry point)
│
├── 📄 pages/
│   ├── 01_compare.py        ← Side-by-side model comparison (default)
│   ├── 02_analyze.py        ← Deep dive into single model
│   ├── 03_explore.py        ← Trade-by-trade explorer
│   └── 04_report.py         ← Generate/share reports
│
├── 🧩 components/
│   ├── model_selector.py        ← Multi-select dropdown
│   ├── metrics_grid.py          ← Key stats cards
│   ├── equity_chart.py          ← Plotly overlay chart
│   ├── trade_timeline.py        ← Buy/sell markers
│   ├── trade_table.py           ← Sortable/filterable trades
│   ├── drawdown_chart.py        ← Underwater curve
│   ├── feature_comparison.py    ← Feature importance across models
│   └── period_selector.py       ← Date range picker
│
└── 🔧 utils/
    ├── ledger_loader.py       ← Parse backtest_ledger.csv
    ├── metrics_calculator.py  ← Sharpe, max DD, win rate, etc.
    ├── trade_analyzer.py      ← Cycle detection, durations
    ├── comparison_engine.py   ← Model vs Model stats
    └── report_generator.py    ← PDF/HTML export
```

## Key Views

### View 1: Model Comparison (Default)
```
┌─────────────────────────────────────────────────────────────────┐
│  [Select Base Model ▼]  vs  [Add Comparison ▼]  [+ Add More]   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Model A     │  │ Model B     │  │ Buy & Hold  │             │
│  │ +12.4% 🥇   │  │ +8.1% 🥈    │  │ +3.2%       │             │
│  │ ----------- │  │ ----------- │  │ ----------- │             │
│  │ Sharpe: 1.8 │  │ Sharpe: 1.5 │  │ Sharpe: 0.8 │             │
│  │ MaxDD: -8%  │  │ MaxDD: -12% │  │ MaxDD: -15% │             │
│  │ Trades: 45  │  │ Trades: 62  │  │ Trades: 1   │             │
│  │ Win%: 58%   │  │ Win%: 52%   │  │ Win%: 100%  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  [Equity Curve Overlay Chart]                                   │
│  [Trade Count Over Time]                                        │
│  [Drawdown Comparison]                                          │
└─────────────────────────────────────────────────────────────────┘
```

### View 2: Single Model Analysis
```
┌─────────────────────────────────────────────────────────────────┐
│  Model: ppo_0918  [View Metadata]  [Open Report Folder]        │
│                                                                 │
│  Performance Summary:                                           │
│  [Total Return] [Sharpe Ratio] [Max Drawdown] [Win Rate]       │
│  [Avg Trade]    [Profit Factor][Expectancy]   [Calmar]         │
│                                                                 │
│  Equity Curve with Trade Markers:                               │
│  [📈 Interactive Chart - click trades for details]             │
│                                                                 │
│  Monthly Returns Heatmap:                                       │
│  [Calendar view of returns]                                     │
│                                                                 │
│  Feature Values During Trades:                                  │
│  [Box plots: RSI when buying vs selling]                       │
└─────────────────────────────────────────────────────────────────┘
```

### View 3: Trade Explorer
```
┌─────────────────────────────────────────────────────────────────┐
│  Filters: [Action ▼] [Min P&L] [Max P&L] [Duration ▼] [Apply]  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Trade # | Date | Action | Entry | Exit | P&L | Duration │  │
│  │--------|------|--------|-------|------|-----|----------│  │
│  │ 1      | Nov5 | BUY50  | $245  | $267 | +$45| 18 bars  │  │
│  │ 2      | Nov8 | SELL100| $267  | $255 | -$24| 12 bars  │  │
│  │ ...    | ...  | ...    | ...   | ...  | ... | ...      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [◀ Prev] [Next ▶]  Showing 1-20 of 45 trades                  │
│                                                                 │
│  Selected Trade Details:                                        │
│  [Price chart segment with entry/exit marked]                   │
│  [Feature values at entry]                                      │
│  [What was AI thinking? - feature context]                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Technical Decisions

### 1. Data Loading Strategy
```python
@st.cache_data
def load_ledger(artifact_path: str) -> pd.DataFrame:
    """Cache ledgers to avoid re-reading CSVs"""
    return pd.read_csv(f"{artifact_path}/backtest_ledger.csv")

@st.cache_data
def calculate_metrics(ledger_df: pd.DataFrame) -> dict:
    """Cache expensive metric calculations"""
    return {
        "total_return": compute_return(ledger_df),
        "sharpe_ratio": compute_sharpe(ledger_df),
        "max_drawdown": compute_max_dd(ledger_df),
        # ...
    }
```

### 2. Metric Calculations
Standard metrics computed from ledger:
```python
class MetricsCalculator:
    def total_return(self, ledger) -> float:
        start = ledger['Portfolio_Value'].iloc[0]
        end = ledger['Portfolio_Value'].iloc[-1]
        return (end - start) / start
    
    def sharpe_ratio(self, ledger, risk_free=0.02) -> float:
        returns = ledger['Portfolio_Value'].pct_change().dropna()
        excess_returns = returns - risk_free/252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def max_drawdown(self, ledger) -> float:
        cummax = ledger['Portfolio_Value'].cummax()
        drawdown = (ledger['Portfolio_Value'] - cummax) / cummax
        return drawdown.min()
    
    def win_rate(self, ledger) -> float:
        trades = self.extract_trades(ledger)
        winners = [t for t in trades if t['pnl'] > 0]
        return len(winners) / len(trades)
```

### 3. Comparison Engine
```python
class ModelComparison:
    def compare_two(self, model_a: str, model_b: str) -> dict:
        """Returns statistical comparison"""
        metrics_a = calculate_metrics(load_ledger(model_a))
        metrics_b = calculate_metrics(load_ledger(model_b))
        
        return {
            "return_diff": metrics_a['total_return'] - metrics_b['total_return'],
            "sharpe_diff": metrics_a['sharpe_ratio'] - metrics_b['sharpe_ratio'],
            "trade_overlap": self.calculate_overlap(model_a, model_b),
            "correlation": self.calculate_correlation(model_a, model_b),
        }
```

### 4. Trade Cycle Extraction
From ledger, reconstruct complete trades:
```python
def extract_cycles(ledger: pd.DataFrame) -> list[dict]:
    """
    Convert flat ledger into trade cycles:
    [{entry_date, exit_date, entry_price, exit_price, pnl, duration}, ...]
    """
    cycles = []
    entry = None
    
    for _, row in ledger.iterrows():
        if row['Position_Before'] == 0 and row['Position_After'] > 0:
            entry = row  # Trade opened
        elif entry is not None and row['Position_After'] == 0:
            # Trade closed
            cycles.append({
                'entry_date': entry['Date'],
                'exit_date': row['Date'],
                'entry_price': entry['Price'],
                'exit_price': row['Price'],
                'pnl': row['Portfolio_Value'] - entry['Portfolio_Value'],
                'duration_bars': row['step'] - entry['step'],
            })
            entry = None
    
    return cycles
```

## Interactive Features

### 1. Click-to-Inspect
```python
# In Plotly chart
fig.update_traces(
    customdata=df[['Date', 'Action', 'Pnl']],
    hovertemplate='%{customdata[0]}<br>Action: %{customdata[1]}<br>P&L: %{customdata[2]}',
)

# Capture click events
selected_point = plotly_events(fig)
if selected_point:
    st.session_state['selected_trade'] = selected_point[0]
```

### 2. Linked Brushing
Select date range on equity chart → trade table filters automatically:
```python
# Date range from chart selection
start_date, end_date = st.plotly_chart(fig, use_container_width=True, 
                                        on_select="rerun")

# Filter table
filtered_trades = trades[(trades['Date'] >= start_date) & 
                         (trades['Date'] <= end_date)]
```

### 3. Feature Analysis
Compare feature distributions:
```python
def analyze_feature_on_trades(ledger: pd.DataFrame, feature: str, 
                               features_df: pd.DataFrame) -> dict:
    """
    What was RSI when model bought vs sold?
    """
    buy_mask = ledger['Action'].str.contains('BUY')
    sell_mask = ledger['Action'].str.contains('SELL')
    
    return {
        'buy_mean': features_df.loc[buy_mask, feature].mean(),
        'sell_mean': features_df.loc[sell_mask, feature].mean(),
        'buy_distribution': features_df.loc[buy_mask, feature].describe(),
        'sell_distribution': features_df.loc[sell_mask, feature].describe(),
    }
```

## Report Generation

### Export Options
```python
def generate_pdf_report(model_names: list) -> bytes:
    """Generate PDF with all charts and tables"""
    # Use reportlab or weasyprint
    pass

def generate_html_report(model_names: list) -> str:
    """Generate shareable HTML"""
    # Static HTML with embedded Plotly charts
    pass

def export_trades_csv(model_name: str) -> str:
    """Export filtered trades"""
    return ledger.to_csv()
```

## Files to Create

```
apps/backtester/
├── ARCHITECTURE.md          ← This file
├── requirements.txt         ← streamlit, plotly, reportlab
├── app.py                   ← Entry point
├── pages/
│   ├── 01_compare.py
│   ├── 02_analyze.py
│   ├── 03_explore.py
│   └── 04_report.py
├── components/
│   ├── __init__.py
│   ├── model_selector.py
│   ├── metrics_grid.py
│   ├── equity_chart.py
│   ├── trade_timeline.py
│   ├── trade_table.py
│   ├── drawdown_chart.py
│   ├── feature_comparison.py
│   └── period_selector.py
└── utils/
    ├── __init__.py
    ├── ledger_loader.py
    ├── metrics_calculator.py
    ├── trade_analyzer.py
    ├── comparison_engine.py
    └── report_generator.py
```

## Dependencies on Core Project

```python
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings  # For paths, symbols
# Note: Do NOT import TradingEnv or processor for analysis
#       Only need ledger files which are self-contained
```

## Success Criteria

- [ ] Load and display any backtest_ledger.csv
- [ ] Compare 2+ models side-by-side
- [ ] Interactive equity curve with zoom/pan
- [ ] Trade table with filtering
- [ ] Calculate standard metrics (Sharpe, MaxDD, etc.)
- [ ] Export reports (CSV, PDF)
- [ ] Click trade to see details
- [ ] Feature analysis visualization

## Integration with Live Trader

Share components between apps:
```
apps/
├── 📁 shared/
│   ├── components/
│   │   ├── price_chart.py
│   │   ├── model_badge.py
│   │   └── metrics_card.py
│   └── utils/
│       ├── model_discovery.py  # Find all artifacts
│       └── formatting.py       # Number formatting
│
├── 📁 live_trader/
│   └── ...
│
└── 📁 backtester/
    └── ...
```

## Future Enhancements

1. **Walk-Forward Analysis**: Show performance over time windows
2. **Feature Ablation**: Compare model with/without specific features
3. **Monte Carlo**: Shuffle trade order for confidence intervals
4. **Regime Detection**: Auto-detect market regimes, show performance per regime
5. **Trade Annotation**: Add comments to specific trades
