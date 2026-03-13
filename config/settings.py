import os
<<<<<<< HEAD
from core import tools
=======
from datetime import datetime
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

# ==========================================
# 🎛️ THE CONTROL ROOM (Single Source of Truth)
# ==========================================

<<<<<<< HEAD
# 1. ASSET & TIMEFRAME
SYMBOL = "TSLA"
TIMEFRAME = "1h"       # Options: "1h" (Hourly) or "1d" (Daily)

# 2. ACTION SPACE (The AI's Trading Style)
# Options:
# "discrete_3" (0=Sell All, 1=Buy All, 2=Hold)
# "discrete_5" (0=Sell All, 1=Sell Half, 2=Hold, 3=Buy Half, 4=Buy All)
ACTION_SPACE_TYPE = "discrete_5"

# 3. REWARD FUNCTION (The AI's Psychology)
# Options:
# "absolute_asymmetric" (Punishes losses 2x harder than gains - capital preservation)
# "pure_pnl" (Rewards/punishes 1:1 based strictly on profit/loss)
REWARD_STRATEGY = "absolute_asymmetric"

# 4. DATA SPLIT DESIGN
# Flexible to balance the amount of training data for 1h and 1d.
TRAIN_END_DATE = "2025-10-31"
TEST_START_DATE = "2025-11-01"
TEST_END_DATE = "2026-03-11"

if TIMEFRAME == "1h":
    TRAIN_START_DATE = "2023-01-01"
elif TIMEFRAME == "1d":
    TRAIN_START_DATE = "2018-01-01"
else:
    raise ValueError(f"Unsupported TIMEFRAME: {TIMEFRAME}")


# 5. TRAINING HYPERPARAMETERS
TOTAL_TIMESTEPS = 250_000
CASH_RISK_FRACTION = 0.75 # how much of the portfolio is deployed (0.99 old default -> very risky trading)
STOP_LOSS_PCT = 0.50     # Maximum tolerated loss before forced exit. -> Reference variables for standard boundaries
TAKE_PROFIT_PCT = 0.50   # Automatic exit when profit reached +20%.
MAX_BARS_NORMALIZATION = 100  # For hourly swing trading but works for daily data as well

# 6. FEATURE ENGINEERING PARAMETERS
if TIMEFRAME == "1h":
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    VOLATILITY_WINDOW = 20

    SHORT_VOL_WINDOW = 24      # 24 hours
    LONG_VOL_WINDOW = 120      # 5 trading days of hourly bars
    MA_FAST_WINDOW = 20
    MA_SLOW_WINDOW = 50

elif TIMEFRAME == "1d":
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    VOLATILITY_WINDOW = 20

    SHORT_VOL_WINDOW = 5       # 5 trading days
    LONG_VOL_WINDOW = 20       # 1 trading month
    MA_FAST_WINDOW = 20
    MA_SLOW_WINDOW = 50

else:
    raise ValueError(f"Unsupported TIMEFRAME: {TIMEFRAME}")

# 7. ENVIRONMENT DEFAULTS
INITIAL_BALANCE = 10000.0
SLIPPAGE_PCT = 0.0005

TRADE_PENALTY_FULL = 0.01
TRADE_PENALTY_HALF = 0.005
INVALID_ACTION_PENALTY = 0.05
BANKRUPTCY_PENALTY = 10.0

MIN_POSITION_THRESHOLD = 1e-8
MAX_BARS_IN_TRADE_NORM = 100.0

# 8. PPO DEFAULT HYPERPARAMETERS
PPO_LEARNING_RATE = 0.0001 #0.0007 #how strongly the neural network weights change after each update
PPO_BATCH_SIZE = 128 #number of sampled transitions used in one update step
PPO_GAMMA = 0.96 #0.91 #how much the agent values future rewards relative to immediate rewards, near 1 agent becomes more long-term
PPO_ENT_COEF = 0.001 #0.0012 # Entropy Coefficient,controls how much PPO is rewarded for keeping its action distribution exploratory/uncertain


# 9. Optional: Create an evaluation plot when running backtest.py
PLOT_BACKTEST = True

# 10. Experiment naming settings
RANDOM_SEED = 42
EXPERIMENT_VERSION = 1

OPTUNA_STUDY_VERSION = "v1"

# ==========================================
# 📦 ARTIFACT TRACKING (Auto-Naming Vault)
# ==========================================

EXPERIMENT_NAME = tools.build_experiment_name()

=======
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

<<<<<<< HEAD
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
BACKTEST_SUMMARY_PATH = os.path.join(ARTIFACT_DIR, "backtest_summary.json")
=======
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
CASH_RISK_FRACTION = 0.70           # Position sizing per buy. Typical: 0.30 to 1.00. Lower = safer.

# ==========================================
# 3. REWARD & RISK
# ==========================================
REWARD_STRATEGY = "pure_pnl"       # "pure_pnl" (neutral) or "absolute_asymmetric" (losses penalized more).
BANKRUPTCY_THRESHOLD_PCT = 0.50     # Episode stops at this drawdown floor. 0.50 = allow 50% capital loss.
BANKRUPTCY_PENALTY = 10.0           # Extra negative reward on bankruptcy. Typical: 5 to 30.
STOP_LOSS_PCT = 0.20                # Forced close when unrealized PnL <= -value. Typical: 0.05 to 0.30.
TAKE_PROFIT_PCT = 0.30              # Forced close when unrealized PnL >= value. Use large value to effectively disable TP.
SLIPPAGE_PCT = 0.0005               # Execution friction. Typical intraday stress range: 0.0005 to 0.002.
INVALID_ACTION_PENALTY = 0.05       # Penalty for impossible actions (e.g., sell with no shares). Typical: 0.01 to 0.10.
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
    "BB_Pct",
    "ATR_Pct",
]

_NEWS_FEATURES  = ["Sentiment_EMA", "News_Intensity"]                          if USE_NEWS_FEATURES  else []
_TIME_FEATURES  = ["Sin_Time", "Cos_Time", "Mins_to_Close"]                    if USE_TIME_FEATURES  else []
_MACRO_FEATURES = ["QQQ_Ret", "ARKK_Ret", "Rel_Strength_QQQ", "VIX_Level", "TNX_Level"] if USE_MACRO_FEATURES else []

FEATURES_LIST = _BASE_FEATURES + _MACRO_FEATURES + _NEWS_FEATURES + _TIME_FEATURES
EXPECTED_MARKET_FEATURES = len(FEATURES_LIST)

# ==========================================
# 6. TECHNICAL INDICATOR PARAMS
# ==========================================
RSI_WINDOW       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
VOLATILITY_WINDOW = 20
NEWS_EMA_SPAN    = 5

# ==========================================
# 7. TRAINING
# ==========================================
TOTAL_TIMESTEPS = 1_000_000
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
OPTUNA_EVAL_TIMESTEPS  = 20_000

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
    "QQQ": "QQQ_Close",
    "ARKK": "ARKK_Close",
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
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
