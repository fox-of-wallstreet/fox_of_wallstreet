import os
import sys
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators, prepare_features
from core.environment import TradingEnv


# ==========================================
# 1. Global Data Setup
# ==========================================
print("📥 Loading local CSV for optimization...")

csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ {csv_path} not found! Run data_engine.py first.")

df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])

mask = (
    (df["Date"] >= pd.to_datetime(settings.TRAIN_START_DATE)) &
    (df["Date"] <= pd.to_datetime(settings.TRAIN_END_DATE))
)
train_df = df.loc[mask].copy().reset_index(drop=True)

if train_df.empty:
    raise ValueError(
        f"❌ Training dataframe is empty before preprocessing. "
        f"Check TRAIN_START_DATE ({settings.TRAIN_START_DATE}) and TRAIN_END_DATE ({settings.TRAIN_END_DATE})."
    )

train_df = add_technical_indicators(train_df)

if train_df.empty:
    raise ValueError("❌ Training dataframe is empty after preprocessing. Check rolling windows and timeframe settings.")

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
    stack_size = 5
elif settings.TIMEFRAME == "1d":
    features_list = base_features
    stack_size = 10
else:
    raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

scaled_features = prepare_features(train_df, features_list, is_training=True)


# ==========================================
# 2. Optuna Search Space
# ==========================================
def sample_ppo_params(trial: optuna.Trial):
    """Suggest PPO hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
    }


# ==========================================
# 3. Objective Function
# ==========================================
def objective(trial: optuna.Trial):
    hyperparams = sample_ppo_params(trial)

    base_env = TradingEnv(df=train_df, features=scaled_features)
    base_env = Monitor(base_env)

    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=stack_size)

    model = PPO("MlpPolicy", env, verbose=0, **hyperparams)

    try:
        model.learn(total_timesteps=20000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3, deterministic=True)
        return mean_reward
    except Exception as e:
        print(f"⚠️ Trial failed: {e}")
        return -1000.0


# ==========================================
# 4. Run Optimization
# ==========================================
def run_optimization():
    print("🧠 Starting Optuna hyperparameter search...")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_warmup_steps=5)

    db_path = "sqlite:///ppo_study.db"
    study_name = f"ppo_{settings.SYMBOL.lower()}_{settings.TIMEFRAME}"

    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=20)

    print("\n🏆 OPTIMIZATION COMPLETE 🏆")
    print(f"✅ Study saved to {db_path}")
    print("Best Trial Score (Mean Reward):", study.best_value)
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    run_optimization()
