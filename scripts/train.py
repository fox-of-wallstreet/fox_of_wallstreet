'''
Missing module docstring.
'''

import os
import sys
import json
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators, prepare_features
from core.environment import TradingEnv

def run_training():
    '''
    Missing function or method docstring.
    '''
    print(f"🚀 INITIATING TRAINING: {settings.EXPERIMENT_NAME}")

    # 1. Load Data
    csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Cannot find {csv_path}. Run data_engine.py first!")

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Slice to Training Dates
    mask = (df['Date'] >= pd.to_datetime(settings.TRAIN_START_DATE, utc=True)) & (df['Date'] <= pd.to_datetime(settings.TRAIN_END_DATE, utc=True))
    train_df = df.loc[mask].copy().reset_index(drop=True)
    print(f"📅 Training Data: {len(train_df)} rows from {settings.TRAIN_START_DATE} to {settings.TRAIN_END_DATE}")

    # 2. Process Features & Scale (Saves the Scaler)
    train_df = add_technical_indicators(train_df)

    if train_df.empty:
      raise ValueError("Training dataframe is empty after preprocessing. Check date split and rolling windows.")

    base_features = [
        'Log_Return',
        'Volume_Z_Score',
        'RSI',
        'MACD_Hist',
        'BB_Pct',
        'ATR_Pct',
        'Realized_Vol_Short',
        'Realized_Vol_Long',
        'Vol_Regime',
        'Dist_MA_Fast',
        'Dist_MA_Slow',
        'QQQ_Ret',
        'ARKK_Ret',
        'Rel_Strength_QQQ',
        'VIX_Z',
        'TNX_Z',
        'Sentiment_EMA',
        'News_Intensity'
    ]

    if settings.TIMEFRAME == "1h":
        features_list = base_features + ['Sin_Time', 'Cos_Time', 'Mins_to_Close']
    elif settings.TIMEFRAME == "1d":
        features_list = base_features
    else:
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

    scaled_features = prepare_features(train_df, features_list, is_training=True)

    # 3. Build Environment with 5-Hour Memory Buffer
    base_env = TradingEnv(df=train_df, features=scaled_features)
    vec_env = DummyVecEnv([lambda: base_env])

    if settings.TIMEFRAME == "1h":
        stack_size = 5
    elif settings.TIMEFRAME == "1d":
        stack_size = 10
    else:
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

    env = VecFrameStack(vec_env, n_stack=stack_size)
    #env = VecFrameStack(vec_env, n_stack=5)

    # 4. Train Brain
    model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=settings.PPO_LEARNING_RATE,
    batch_size=settings.PPO_BATCH_SIZE,
    gamma=settings.PPO_GAMMA,
    ent_coef=settings.PPO_ENT_COEF
)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01)
    model.learn(total_timesteps=settings.TOTAL_TIMESTEPS)

    # 5. Save Model
    model.save(settings.MODEL_PATH)
    print(f"🧠 Model saved to {settings.MODEL_PATH}")

    # 6. Generate MLOps Metadata Receipt
    metadata = {
        "Experiment_Name": settings.EXPERIMENT_NAME,
        "Symbol": settings.SYMBOL,
        "Timeframe": settings.TIMEFRAME,
        "Action_Space": settings.ACTION_SPACE_TYPE,
        "Reward_Strategy": settings.REWARD_STRATEGY,
        "Train_Dates": f"{settings.TRAIN_START_DATE} to {settings.TRAIN_END_DATE}",
        "Test_Dates": f"{settings.TEST_START_DATE} to {settings.TEST_END_DATE}",
        "Total_Timesteps": settings.TOTAL_TIMESTEPS,
        "Cash_Risk_Fraction": settings.CASH_RISK_FRACTION,
        "Features_Used": features_list
    }

    with open(settings.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"🧾 Metadata receipt generated at {settings.METADATA_PATH}")
    print(f"✅ Training Complete. All artifacts secured in Vault: {settings.ARTIFACT_DIR}")

if __name__ == "__main__":
    run_training()
