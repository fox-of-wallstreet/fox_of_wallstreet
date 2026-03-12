"""
Train one PPO trading agent and save its artifacts.
"""

import os
import sys
import json
from datetime import datetime, timezone

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators, prepare_features
from core.environment import TradingEnv
from core.tools import fnline, get_features_list, get_stack_size


def run_training():
    """
    Train a PPO agent on the configured training period and save model metadata.
    """
    print(fnline(), f"🚀 INITIATING TRAINING: {settings.EXPERIMENT_NAME}")

    # 1. Load raw market + news data
    csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{fnline()} ❌ Cannot find {csv_path}. Run data_engine.py first!"
        )

    df_all = pd.read_csv(csv_path)
    df_all["Date"] = pd.to_datetime(df_all["Date"], utc=True)

    # 2. Slice to training dates
    train_mask = (
        (df_all["Date"] >= pd.to_datetime(settings.TRAIN_START_DATE, utc=True))
        & (df_all["Date"] <= pd.to_datetime(settings.TRAIN_END_DATE, utc=True))
    )
    train_df = df_all.loc[train_mask].copy().reset_index(drop=True)

    raw_rows_loaded = len(train_df)

    print(
        fnline(),
        f"📅 Training Data: {raw_rows_loaded} rows from "
        f"{settings.TRAIN_START_DATE} to {settings.TRAIN_END_DATE}"
    )

    if train_df.empty:
        raise ValueError(
            f"{fnline()} ❌ Training dataframe is empty after date filtering."
        )

    # 3. Feature engineering
    train_df = add_technical_indicators(train_df)
    rows_after_feature_engineering = len(train_df)

    if train_df.empty:
        raise ValueError(
            f"{fnline()} ❌ Training dataframe is empty after preprocessing. "
            "Check date split and rolling windows."
        )

    print(
        fnline(),
        f"📈 Rows after feature engineering: {rows_after_feature_engineering}"
    )

    # Current train.py does not use a separate validation split yet
    train_rows = len(train_df)
    validation_rows = 0
    total_rows_used = train_rows + validation_rows

    features_list = get_features_list()
    stack_size = get_stack_size()

    # 4. Scale features using the training set and save the scaler
    scaled_features = prepare_features(train_df, features_list, is_training=True)

    # 5. Build vectorized RL environment
    base_env = TradingEnv(df=train_df, features=scaled_features)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(vec_env, n_stack=stack_size)

    # 6. Train PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=settings.PPO_LEARNING_RATE,
        batch_size=settings.PPO_BATCH_SIZE,
        gamma=settings.PPO_GAMMA,
        ent_coef=settings.PPO_ENT_COEF,
        seed=settings.RANDOM_SEED,
    )

    model.learn(total_timesteps=settings.TOTAL_TIMESTEPS)

    # 7. Save trained model
    model.save(settings.MODEL_PATH)
    print(fnline(), f"🧠 Model saved to {settings.MODEL_PATH}")

    # 8. Save metadata receipt
    timesteps_per_train_row = (
        settings.TOTAL_TIMESTEPS / train_rows if train_rows > 0 else None
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "experiment_name": settings.EXPERIMENT_NAME,
            "symbol": settings.SYMBOL,
            "timeframe": settings.TIMEFRAME,
            "action_space_type": settings.ACTION_SPACE_TYPE,
            "reward_strategy": settings.REWARD_STRATEGY,
            "random_seed": settings.RANDOM_SEED,
            "experiment_version": settings.EXPERIMENT_VERSION,
        },
        "data_split": {
            "train_start_date": settings.TRAIN_START_DATE,
            "train_end_date": settings.TRAIN_END_DATE,
            "test_start_date": settings.TEST_START_DATE,
            "test_end_date": settings.TEST_END_DATE,
        },
        "dataset_statistics": {
            "raw_rows_loaded": raw_rows_loaded,
            "rows_after_feature_engineering": rows_after_feature_engineering,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "total_rows_used": total_rows_used,
        },
        "training": {
            "total_timesteps": settings.TOTAL_TIMESTEPS,
            "timesteps_per_train_row": timesteps_per_train_row,
            "cash_risk_fraction": settings.CASH_RISK_FRACTION,
            "stop_loss_pct": settings.STOP_LOSS_PCT,
            "take_profit_pct": settings.TAKE_PROFIT_PCT,
            "max_bars_normalization": settings.MAX_BARS_NORMALIZATION,
        },
        "feature_engineering": {
            "rsi_window": settings.RSI_WINDOW,
            "macd_fast": settings.MACD_FAST,
            "macd_slow": settings.MACD_SLOW,
            "macd_signal": settings.MACD_SIGNAL,
            "volatility_window": settings.VOLATILITY_WINDOW,
            "short_vol_window": settings.SHORT_VOL_WINDOW,
            "long_vol_window": settings.LONG_VOL_WINDOW,
            "ma_fast_window": settings.MA_FAST_WINDOW,
            "ma_slow_window": settings.MA_SLOW_WINDOW,
            "features_used": features_list,
            "n_features": len(features_list),
            "stack_size": stack_size,
        },
        "environment": {
            "initial_balance": settings.INITIAL_BALANCE,
            "slippage_pct": settings.SLIPPAGE_PCT,
            "trade_penalty_full": settings.TRADE_PENALTY_FULL,
            "trade_penalty_half": settings.TRADE_PENALTY_HALF,
            "invalid_action_penalty": settings.INVALID_ACTION_PENALTY,
            "bankruptcy_penalty": settings.BANKRUPTCY_PENALTY,
            "min_position_threshold": settings.MIN_POSITION_THRESHOLD,
            "max_bars_in_trade_norm": settings.MAX_BARS_IN_TRADE_NORM,
        },
        "ppo_hyperparameters": {
            "learning_rate": settings.PPO_LEARNING_RATE,
            "batch_size": settings.PPO_BATCH_SIZE,
            "gamma": settings.PPO_GAMMA,
            "ent_coef": settings.PPO_ENT_COEF,
        },
        "artifacts": {
            "artifact_dir": settings.ARTIFACT_DIR,
            "model_path": settings.MODEL_PATH,
            "scaler_path": settings.SCALER_PATH,
            "metadata_path": settings.METADATA_PATH,
        },
    }

    with open(settings.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(fnline(), f"🧾 Metadata receipt generated at {settings.METADATA_PATH}")
    print(fnline(), f"✅ Training complete. All artifacts secured in Vault: {settings.ARTIFACT_DIR}")


if __name__ == "__main__":
    run_training()
