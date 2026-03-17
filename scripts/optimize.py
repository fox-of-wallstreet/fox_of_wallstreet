"""
Optuna Hyperparameter Optimization.
Searches for best PPO params using short training bursts.
Results are saved to settings.OPTUNA_DB_PATH for train.py to consume.
All configuration is driven by config/settings.py.
"""

import os
import sys
import math

import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
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


def _extract_portfolio_value_from_info(info):
    """
    Try several plausible portfolio/equity keys from env info.
    Returns float or None.
    """
    if not isinstance(info, dict):
        return None

    candidate_keys = [
        "portfolio_value",
        "Portfolio_Value",
        "equity",
        "Equity",
        "account_value",
        "Account_Value",
        "net_worth",
        "Net_Worth",
    ]

    for key in candidate_keys:
        value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def _run_deterministic_rollout(model, env):
    """
    Run one deterministic evaluation episode and compute simple portfolio metrics.

    Preferred path:
    - use portfolio value extracted from env info

    Fallback path:
    - if no portfolio value is available, use cumulative reward as a weaker proxy
    """
    obs = env.reset()

    portfolio_values = []
    cumulative_reward = 0.0
    trade_count = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        action_int = int(action[0]) if isinstance(action, (np.ndarray, list, tuple)) else int(action)
        if action_int != 0:
            trade_count += 1

        obs, rewards, dones, infos = env.step(action)

        reward_value = float(rewards[0]) if isinstance(rewards, (np.ndarray, list, tuple)) else float(rewards)
        cumulative_reward += reward_value

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        dones_flag = dones[0] if isinstance(dones, (np.ndarray, list, tuple)) else dones
        done = bool(dones_flag)

        pv = _extract_portfolio_value_from_info(info)
        if pv is not None:
            portfolio_values.append(float(pv))

    # Preferred scoring path: portfolio-based
    if len(portfolio_values) >= 2:
        portfolio_values = np.asarray(portfolio_values, dtype=float)
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]

        total_return = ((final_value / (initial_value + 1e-8)) - 1.0) * 100.0
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = ((portfolio_values - running_max) / (running_max + 1e-8)) * 100.0
        max_drawdown = float(drawdowns.min())  # negative %

        # Balanced objective
        # score = total_return - 0.1 * abs(max_drawdown) # aggressive approach
        score = total_return - 0.3 * abs(max_drawdown)
        # score = total_return - 0.7 * abs(max_drawdown) - 0.02 * trade_count # conservative approach

        return {
            "score": float(score),
            "score_source": "portfolio_value",
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "trade_count": int(trade_count),
            "final_value": float(final_value),
            "cumulative_reward": float(cumulative_reward),
        }

    # Fallback scoring path: cumulative reward only
    return {
        "score": float(cumulative_reward),
        "score_source": "cumulative_reward_fallback",
        "total_return": None,
        "max_drawdown": None,
        "trade_count": int(trade_count),
        "final_value": None,
        "cumulative_reward": float(cumulative_reward),
    }


def cosine_lr(initial_lr: float):
    """Cosine decay schedule — mirrors train.py so Optuna trials are evaluated identically."""
    def schedule(progress_remaining: float) -> float:
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return initial_lr * (
            settings.LR_COSINE_MIN_FRACTION
            + (1.0 - settings.LR_COSINE_MIN_FRACTION) * cosine_factor
        )
    return schedule

# The OPTUNA Search space (if needed adjust here!)
def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space.
    Extend this function to search over more params (e.g. n_steps, clip_range).
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-4, 8e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.95, log=False),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.01, log=True),
    }


def build_objective(train_df, scaled_features):
    """
    Factory that closes over the data so Optuna's objective
    doesn't reload or reprocess on every trial.
    """
    def objective(trial: optuna.Trial) -> float:
        hyperparams = sample_ppo_params(trial)

        vec_env = DummyVecEnv([lambda: TradingEnv(df=train_df, features=scaled_features)])
        vec_env = VecMonitor(vec_env)
        env = VecFrameStack(vec_env, n_stack=settings.N_STACK)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=settings.RANDOM_SEED,
            **hyperparams,
        )

        try:
            model.learn(total_timesteps=settings.OPTUNA_EVAL_TIMESTEPS)
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed during training: {e}")
            return -1000.0

        try:
            metrics = _run_deterministic_rollout(model, env)
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed during evaluation: {e}")
            return -1000.0

        # Persist richer diagnostics in the Optuna DB
        for key, value in metrics.items():
            trial.set_user_attr(key, value)

        if metrics["score_source"] == "portfolio_value":
            print(
                f"Trial {trial.number:02d} | "
                f"score={metrics['score']:.3f} | "
                f"ret={metrics['total_return']:.3f}% | "
                f"mdd={metrics['max_drawdown']:.3f}% | "
                f"trades={metrics['trade_count']}"
            )
        else:
            print(
                f"Trial {trial.number:02d} | "
                f"score={metrics['score']:.3f} | "
                f"source={metrics['score_source']} | "
                f"trades={metrics['trade_count']}"
            )

        return float(metrics["score"])

    return objective


def run_optimization():
    print(f"🧠 Starting Optuna Search: {settings.OPTUNA_STUDY_NAME}")
    print(f"   Trials:          {settings.OPTUNA_TRIALS}")
    print(f"   Steps per trial: {settings.OPTUNA_EVAL_TIMESTEPS}")
    print(f"   DB:              {settings.OPTUNA_DB_PATH}")

    os.makedirs(os.path.dirname(settings.OPTUNA_DB_PATH), exist_ok=True)

    # -------------------------------------------------------
    # 1. Load data once — all trials share the same dataset
    # -------------------------------------------------------
    train_df = _load_train_data()
    scaled_features = _scale_for_optimization(train_df)
    print(
        f"📅 Optimizing on: {len(train_df)} rows | "
        f"{settings.TRAIN_START_DATE} → {settings.TRAIN_END_DATE}"
    )

    # -------------------------------------------------------
    # 2. Create or resume study
    # load_if_exists=True means you can stop and continue later
    # -------------------------------------------------------
    storage = f"sqlite:///{settings.OPTUNA_DB_PATH}"
    study = optuna.create_study(
        study_name=settings.OPTUNA_STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=settings.RANDOM_SEED),
        pruner=MedianPruner(n_warmup_steps=5),
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
    print(f"   Best objective score: {study.best_value:.4f}")
    print("   Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

    print("   Best trial diagnostics:")
    for key in [
        "score_source",
        "total_return",
        "max_drawdown",
        "trade_count",
        "final_value",
        "cumulative_reward",
    ]:
        if key in study.best_trial.user_attrs:
            print(f"      {key}: {study.best_trial.user_attrs[key]}")

    print(f"\n💾 Results saved → {settings.OPTUNA_DB_PATH}")
    print("   Set USE_OPTUNA_BEST_PARAMS = True in settings.py to use these in train.py")


if __name__ == "__main__":
    run_optimization()
