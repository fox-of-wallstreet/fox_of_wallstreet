import os
from core import tools

# ==========================================
# 🎛️ THE CONTROL ROOM (Team Settings)
# ==========================================

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", EXPERIMENT_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
BACKTEST_SUMMARY_PATH = os.path.join(ARTIFACT_DIR, "backtest_summary.json")
