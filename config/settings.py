import os

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

# 4. DATE SWAPPING (Train on New, Test on Old?)
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE   = "2025-10-31"

TEST_START_DATE  = "2025-11-01"
TEST_END_DATE    = "2026-03-10"

# 5. TRAINING HYPERPARAMETERS
TOTAL_TIMESTEPS = 1_000_000
CASH_RISK_FRACTION = 0.99
STOP_LOSS_PCT = 0.10     # Reference variables for standard boundaries
TAKE_PROFIT_PCT = 0.20
MAX_BARS_NORMALIZATION = 100  # For hourly swing trading

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
PPO_LEARNING_RATE = 3e-4
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
PPO_ENT_COEF = 0.01

# ==========================================
# 📦 ARTIFACT TRACKING (Auto-Naming Vault)
# ==========================================
ACTION_TAG = "d5" if ACTION_SPACE_TYPE == "discrete_5" else "d3"
REWARD_TAG = "asym" if REWARD_STRATEGY == "absolute_asymmetric" else "pnl"
ENV_TAG = f"pen{int(TRADE_PENALTY_FULL * 1000)}"
PPO_TAG = f"lr{str(PPO_LEARNING_RATE).replace('.', '')}"

# Example: ppo_TSLA_1h_d5_asym_pen10_lr00003_v1
EXPERIMENT_NAME = f"ppo_{SYMBOL}_{TIMEFRAME}_{ACTION_TAG}_{REWARD_TAG}_{ENV_TAG}_{PPO_TAG}_v1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts", EXPERIMENT_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
