<<<<<<< HEAD
import os
import sys
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from sklearn.preprocessing import RobustScaler
=======
"""
Optuna Hyperparameter Optimization.
Searches for best PPO params using short training bursts.
Results are saved to settings.OPTUNA_DB_PATH for train.py to consume.
All configuration is driven by config/settings.py.
"""

import os
import sys

import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import pandas as pd
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
<<<<<<< HEAD
from core.processor import add_technical_indicators
from core.environment import TradingEnv
from core.tools import fnline, get_features_list, get_stack_size

# ! Set the number of timesteps for this ppo hyperparameter optimization training
OPT_TIMESTEPS = 100_000
N_TRAILS = 5 # number of trails to find best hyperparameters

# ==========================================
# 1. Helpers
# ==========================================
def scale_train_valid(train_df, valid_df, features_list):
    """
    Fit scaler on training subset only, then transform both train and validation.
    This avoids overwriting the real scaler artifact used by train/backtest/live.
    """
    scaler = RobustScaler()

    train_scaled = scaler.fit_transform(train_df[features_list])
    valid_scaled = scaler.transform(valid_df[features_list])

    train_scaled_df = pd.DataFrame(train_scaled, columns=features_list, index=train_df.index)
    valid_scaled_df = pd.DataFrame(valid_scaled, columns=features_list, index=valid_df.index)

    return train_scaled_df, valid_scaled_df


def make_env(df, features, stack_size):
    base_env = TradingEnv(df=df, features=features)
    base_env = Monitor(base_env)
    env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(env, n_stack=stack_size)
    return env


def get_study_name():
    action_tag = "d5" if settings.ACTION_SPACE_TYPE == "discrete_5" else "d3"
    reward_tag = "asym" if settings.REWARD_STRATEGY == "absolute_asymmetric" else "pnl"
    env_tag = f"pen{int(settings.TRADE_PENALTY_FULL * 1000)}"
    return f"ppo_{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_{action_tag}_{reward_tag}_{env_tag}_{settings.OPTUNA_STUDY_VERSION}"


def run_validation_backtest(model, env):
    """
    Run one deterministic validation episode and return:
    - final portfolio value
    - total return (%)
    - max drawdown (%)
    - trade count
    """
    obs = env.reset()
    done = [False]
    portfolio_values = []
    prev_position = env.get_attr("position")[0]
    trade_count = 0

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, infos = env.step(action)

        info = infos[0]
        portfolio_value = info["portfolio_value"]
        portfolio_values.append(portfolio_value)

        current_position = env.get_attr("position")[0]
        if current_position != prev_position:
            trade_count += 1
            prev_position = current_position

    final_value = portfolio_values[-1]
    initial_value = env.get_attr("initial_balance")[0]
    total_return = ((final_value - initial_value) / initial_value) * 100

    peak = portfolio_values[0]
    max_drawdown = 0.0
    for v in portfolio_values:
        peak = max(peak, v)
        dd = ((v - peak) / peak) * 100
        max_drawdown = min(max_drawdown, dd)

    return final_value, total_return, max_drawdown, trade_count


# ==========================================
# 2. Global Data Setup
# ==========================================
print(fnline(), "📥 Loading local CSV for optimization...")

csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ {csv_path} not found! Run data_engine.py first.")

df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])

mask = (
    (df["Date"] >= pd.to_datetime(settings.TRAIN_START_DATE)) &
    (df["Date"] <= pd.to_datetime(settings.TRAIN_END_DATE))
)
full_train_df = df.loc[mask].copy().reset_index(drop=True)

if full_train_df.empty:
    raise ValueError(
        f"❌ Training dataframe is empty before preprocessing. "
        f"Check TRAIN_START_DATE ({settings.TRAIN_START_DATE}) and TRAIN_END_DATE ({settings.TRAIN_END_DATE})."
    )

full_train_df = add_technical_indicators(full_train_df)

if full_train_df.empty:
    raise ValueError("❌ Training dataframe is empty after preprocessing. Check rolling windows and timeframe settings.")

# Chronological split: first 80% train, last 20% validation
split_idx = int(len(full_train_df) * 0.8)
train_df = full_train_df.iloc[:split_idx].copy().reset_index(drop=True)
valid_df = full_train_df.iloc[split_idx:].copy().reset_index(drop=True)

if train_df.empty or valid_df.empty:
    raise ValueError("❌ Train/validation split produced an empty subset.")

features_list = get_features_list()
stack_size = get_stack_size()

train_scaled, valid_scaled = scale_train_valid(train_df, valid_df, features_list)

print(fnline(), f"📊 Optimization train rows: {len(train_df)}")
print(fnline(), f"📊 Optimization valid rows: {len(valid_df)}")
print(fnline(), f"📊 Features used: {len(features_list)}")
print(fnline(), f"📊 Stack size: {stack_size}")


# ==========================================
# 3. Optuna Search Space
# ==========================================
def sample_ppo_params(trial: optuna.Trial):
    """Suggest PPO hyperparameters within safer bounds."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.98),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-3, log=True),
    }


# ==========================================
# 4. Objective Function
# ==========================================
def objective(trial: optuna.Trial):
    hyperparams = sample_ppo_params(trial)

    train_env = make_env(train_df, train_scaled, stack_size)
    valid_env = make_env(valid_df, valid_scaled, stack_size)

    model = PPO("MlpPolicy", train_env, verbose=0, **hyperparams)

    try:
        # Longer than 20k to reduce short-horizon bias
        model.learn(total_timesteps=OPT_TIMESTEPS)

        final_value, total_return, max_drawdown, trade_count = run_validation_backtest(model, valid_env)

        # Validation return with mild drawdown penalty
        #score = total_return - 0.7 * abs(max_drawdown) # favors conservative behavior of agent
        score = total_return - 0.3 * abs(max_drawdown) # Balance return and drawdown.
        #score = total_return - 0.1 * abs(max_drawdown) # more agressive obj. func regarding trading

        return score

    except Exception as e:
        print(fnline(), f"⚠️ Trial failed: {e}")
        return -1000.0


# ==========================================
# 5. Run Optimization
# ==========================================
def run_optimization():
    print(fnline(), "🧠 Starting Optuna hyperparameter search...")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_warmup_steps=5)

    db_path = "sqlite:///ppo_study.db"
    study_name = get_study_name()

    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=N_TRAILS)

    print(fnline(), "🏆 OPTIMIZATION COMPLETE 🏆")
    print(fnline(), f"✅ Study saved to {db_path}")
    print(fnline(), f"Study name: {study_name}")
    print(fnline(), f"Best Trial Score (validation score): {study.best_value}")
    print(fnline(), "Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(fnline(), f"    {key}: {value}")
=======
from core.processor import build_training_dataset
from core.environment import TradingEnv


def _load_train_data():
    """
    Load training data from checkpoint if available, otherwise rebuild.
    Reuses the same logic as train.py — no duplication.
    """
    if os.path.exists(settings.TRAIN_FEATURES_CSV):
        print("⚡ Loaded train features from checkpoint — skipping reprocessing.")
        return pd.read_csv(settings.TRAIN_FEATURES_CSV, parse_dates=["Date"])
    return build_training_dataset()


def _scale_for_optimization(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a fresh scaler in-memory for optimization runs.
    This avoids any dependency on a previously saved training scaler file.
    """
    features_list = settings.FEATURES_LIST
    missing = [col for col in features_list if col not in train_df.columns]
    if missing:
        raise ValueError(f"❌ Missing requested feature columns in optimization data: {missing}")

    data_to_scale = train_df[features_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data_to_scale)
    return pd.DataFrame(scaled, columns=features_list, index=train_df.index)


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space.
    Extend this function to search over more params (e.g. n_steps, clip_range).
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.9999, log=True),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
    }


def build_objective(train_df, scaled_features):
    """
    Factory that closes over the data so Optuna's objective
    doesn't reload or reprocess on every trial.
    """
    def objective(trial: optuna.Trial) -> float:
        hyperparams = sample_ppo_params(trial)

        base_env = TradingEnv(df=train_df, features=scaled_features)
        base_env = Monitor(base_env)
        env      = VecFrameStack(DummyVecEnv([lambda: base_env]), n_stack=settings.N_STACK)

        model = PPO("MlpPolicy", env, verbose=0, seed=settings.RANDOM_SEED, **hyperparams)

        try:
            model.learn(total_timesteps=settings.OPTUNA_EVAL_TIMESTEPS)
        except Exception as e:
            print(f"⚠️  Trial {trial.number} failed: {e}")
            return -1000.0

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3, deterministic=True)
        return mean_reward

    return objective


def run_optimization():
    print(f"🧠 Starting Optuna Search: {settings.OPTUNA_STUDY_NAME}")
    print(f"   Trials:          {settings.OPTUNA_TRIALS}")
    print(f"   Steps per trial: {settings.OPTUNA_EVAL_TIMESTEPS}")
    print(f"   DB:              {settings.OPTUNA_DB_PATH}")

    # -------------------------------------------------------
    # 1. Load data once — all trials share the same dataset
    # -------------------------------------------------------
    train_df        = _load_train_data()
    scaled_features = _scale_for_optimization(train_df)
    print(f"📅 Optimizing on: {len(train_df)} rows | "
          f"{settings.TRAIN_START_DATE} → {settings.TRAIN_END_DATE}")

    # -------------------------------------------------------
    # 2. Create or resume study
    # load_if_exists=True means you can stop and continue later
    # -------------------------------------------------------
    storage    = f"sqlite:///{settings.OPTUNA_DB_PATH}"
    study      = optuna.create_study(
        study_name     = settings.OPTUNA_STUDY_NAME,
        storage        = storage,
        load_if_exists = True,          # Append to existing — safe to re-run
        direction      = "maximize",
        sampler        = TPESampler(seed=settings.RANDOM_SEED),
        pruner         = MedianPruner(n_warmup_steps=5),
    )

    # -------------------------------------------------------
    # 3. Run trials
    # -------------------------------------------------------
    objective = build_objective(train_df, scaled_features)
    study.optimize(objective, n_trials=settings.OPTUNA_TRIALS)

    # -------------------------------------------------------
    # 4. Report results
    # -------------------------------------------------------
    print("\n🏆 OPTIMIZATION COMPLETE")
    print(f"   Best mean reward: {study.best_value:.4f}")
    print("   Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")
    print(f"\n💾 Results saved → {settings.OPTUNA_DB_PATH}")
    print("   Set USE_OPTUNA_BEST_PARAMS = True in settings.py to use these in train.py")
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485


if __name__ == "__main__":
    run_optimization()
