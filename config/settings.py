import os
from datetime import datetime

# ==========================================
# 🎛️ THE CONTROL ROOM (Single Source of Truth)
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "intermediate")
ARTIFACTS_BASE_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_BASE_DIR, exist_ok=True)

# ==========================================
# 1. ASSET & TIMEFRAME
# ==========================================
SYMBOL = "TSLA"
TIMEFRAME = "1h"   # Options: "1h", "1d"

# ==========================================
# 2. ACTION SPACE
# ==========================================
ACTION_SPACE_TYPE = "discrete_5"   # Options: "discrete_3", "discrete_5"
INITIAL_BALANCE = 10_000.0          # Start capital. Typical: 1_000 to 100_000.
CASH_RISK_FRACTION = 0.65           # Position sizing per buy. Typical: 0.30 to 1.00. Lower = safer.

# ==========================================
# 3. REWARD & RISK
# ==========================================
REWARD_STRATEGY = "pure_pnl"       # "pure_pnl" (neutral) or "absolute_asymmetric" (losses penalized more).
BANKRUPTCY_THRESHOLD_PCT = 0.50     # Episode stops at this drawdown floor. 0.50 = allow 50% capital loss.
BANKRUPTCY_PENALTY = 10.0           # Extra negative reward on bankruptcy. Typical: 5 to 30.
STOP_LOSS_PCT = 0.20                # Forced close when unrealized PnL <= -value. Typical: 0.05 to 0.30.
TAKE_PROFIT_PCT = 0.30              # Forced close when unrealized PnL >= value. Use large value to effectively disable TP.
SLIPPAGE_PCT = 0.0005               # Execution friction. Typical intraday stress range: 0.0005 to 0.002.
INVALID_ACTION_PENALTY = 0.08       # Penalty for impossible actions (e.g., sell with no shares). Typical: 0.01 to 0.10.
MIN_INVESTMENT_FRACTION = 0.001     # Minimum meaningful buy as fraction of INITIAL_BALANCE ($10 on $10k). Buys below this get invalid-action penalty.
MAX_BARS_NORMALIZATION = 100        # Normalizer for bars_in_trade feature. Typical: 50 to 300.

# ==========================================
# 4. DATES
# ==========================================
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE = "2025-10-31"
TEST_START_DATE = "2025-11-01"
TEST_END_DATE = "2026-03-06"

# ==========================================
# 5. FEATURE FLAGS & FEATURE LIST
# ==========================================
USE_NEWS_FEATURES = True
USE_TIME_FEATURES = True
USE_MACRO_FEATURES = True

_BASE_FEATURES = [
    "Log_Return",
    "Volume_Z_Score",
    "RSI",
    "MACD_Hist",
    "ATR_Pct",
    "Dist_MA_Slow",
    "Realized_Vol_Short",
    "Vol_Regime",
    "AVWAP_Dist",
    "AVWAP_Dist_ATR",
]

_NEWS_FEATURES  = ["Sentiment_Mean", "News_Intensity"]                        if USE_NEWS_FEATURES  else []
_TIME_FEATURES  = ["Sin_Time", "Cos_Time"]                                    if USE_TIME_FEATURES  else []
_MACRO_FEATURES = ["QQQ_Ret", "Rel_Strength_QQQ", "VIX_Z", "TNX_Z"]          if USE_MACRO_FEATURES else []

FEATURES_LIST = _BASE_FEATURES + _MACRO_FEATURES + _NEWS_FEATURES + _TIME_FEATURES
EXPECTED_MARKET_FEATURES = len(FEATURES_LIST)

# ==========================================
# 6. TECHNICAL INDICATOR PARAMS
# ==========================================
RSI_WINDOW        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
VOLATILITY_WINDOW = 20
MA_SLOW_WINDOW    = 50    # For Dist_MA_Slow
VOL_SHORT_WINDOW  = 10    # For Realized_Vol_Short
VOL_LONG_WINDOW   = 30    # For Vol_Regime denominator
NEWS_EMA_SPAN     = 5     # Kept for build_news_sentiment internal use

# ==========================================
# AVWAP PARAMS
# ==========================================
AVWAP_PIVOT_LEFT_H  = 5    # Hourly: bars to the left for pivot detection
AVWAP_PIVOT_RIGHT_H = 5    # Hourly: bars to the right for pivot confirmation
AVWAP_PIVOT_LEFT_D  = 3    # Daily: bars to the left for pivot detection
AVWAP_PIVOT_RIGHT_D = 3    # Daily: bars to the right for pivot confirmation
AVWAP_ATR_K_H       = 0.75 # Hourly ATR significance multiplier
AVWAP_ATR_K_D       = 1.0  # Daily ATR significance multiplier

# ==========================================
# 7. TRAINING
# ==========================================
TOTAL_TIMESTEPS = 500_000  # Best observed for TSLA 1h discrete_5 (Mogens)
LEARNING_RATE   = 3e-4
ENT_COEF        = 0.01
N_STACK         = 5
RANDOM_SEED     = 42

# ==========================================
# 8. OPTUNA
# ==========================================
USE_OPTUNA_BEST_PARAMS = True
OPTUNA_STUDY_NAME      = f"ppo_{SYMBOL.lower()}_{TIMEFRAME}"
OPTUNA_DB_PATH         = os.path.join(BASE_DIR, "artifacts", "optuna_study.db")
OPTUNA_TRIALS          = 20
OPTUNA_EVAL_TIMESTEPS  = 75_000  # ~25% of TOTAL_TIMESTEPS — enough for meaningful trial differentiation

# ==========================================
# 9. EXPERIMENT NAMING (Hybrid: readable + timestamp)
# Encodes your key experiment choices directly in the folder name.
# Two runs never collide. No need to remember what "_v4" was.
# ==========================================
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

EXPERIMENT_NAME = (
    f"ppo_{SYMBOL}_{TIMEFRAME}_{ACTION_SPACE_TYPE}"
    f"_{'news' if USE_NEWS_FEATURES  else 'nonews'}"
    f"_{'macro' if USE_MACRO_FEATURES else 'nomacro'}"
    f"_{'time' if USE_TIME_FEATURES  else 'notime'}"
    f"_{_RUN_TIMESTAMP}"
)

ARTIFACT_DIR = os.path.join(ARTIFACTS_BASE_DIR, EXPERIMENT_NAME)

# ==========================================
# 10. RAW CSV CHECKPOINTS
# ==========================================
RAW_PRICES_CSV = os.path.join(RAW_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_prices.csv")
RAW_NEWS_CSV   = os.path.join(RAW_DATA_DIR, f"{SYMBOL.lower()}_news.csv")
RAW_MACRO_CSV  = os.path.join(RAW_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_macro.csv")

# Macro symbols and output column names used by scripts/macro_engine.py
MACRO_SYMBOL_MAP = {
    "QQQ":  "QQQ_Close",
    "^VIX": "VIX_Close",
    "^TNX": "TNX_Close",
}

# ==========================================
# 11. INTERMEDIATE CSV CHECKPOINTS
# ==========================================
NEWS_SENTIMENT_CSV = os.path.join(INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_news_sentiment.csv")
MERGED_DATA_CSV    = os.path.join(INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_merged.csv")
TRAIN_FEATURES_CSV = os.path.join(INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_train_features.csv")
TEST_FEATURES_CSV  = os.path.join(INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_test_features.csv")
TRAIN_FEATURES_SIGNATURE_JSON = os.path.join(
    INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_train_features_signature.json"
)
TEST_FEATURES_SIGNATURE_JSON = os.path.join(
    INTERMEDIATE_DATA_DIR, f"{SYMBOL.lower()}_{TIMEFRAME}_test_features_signature.json"
)

# ==========================================
# 12. ARTIFACT OUTPUTS
# ==========================================
MODEL_PATH          = os.path.join(ARTIFACT_DIR, "model")
SCALER_PATH         = os.path.join(ARTIFACT_DIR, "scaler.pkl")
METADATA_PATH       = os.path.join(ARTIFACT_DIR, "metadata.json")
BACKTEST_LEDGER_PATH = os.path.join(ARTIFACT_DIR, "backtest_ledger.csv")

# ==========================================
# 13. VALIDATION CONSTANTS
# ==========================================
VALID_TIMEFRAMES        = {"1h", "1d"}
VALID_ACTION_SPACES     = {"discrete_3", "discrete_5"}
VALID_REWARD_STRATEGIES = {"absolute_asymmetric", "pure_pnl"}
MIN_TRAIN_ROWS          = 100
