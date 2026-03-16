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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),  # clamped — >3e-4 destabilizes PPO
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma":         trial.suggest_float("gamma", 0.90, 0.97, log=False),   # restored to 0.90 — 0.865 caused overtrading
        "ent_coef":      trial.suggest_float("ent_coef", 5e-4, 0.01, log=True), # floor raised 1e-4→5e-4 (prevents early entropy collapse)
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


if __name__ == "__main__":
    run_optimization()
