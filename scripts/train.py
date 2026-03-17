"""
PPO Training Pipeline.
Orchestrates: data loading → feature engineering → scaling → env → train → save.
All configuration is driven exclusively by config/settings.py.
"""

import os
import sys
import json
import math

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.experiment_journal import log_training_run
from core.processor import build_training_dataset, prepare_features
from core.environment import TradingEnv


def cosine_lr(initial_lr: float):
    """Returns an SB3-compatible LR schedule: cosine decay from initial_lr down to
    initial_lr * LR_COSINE_MIN_FRACTION. High exploration early, stable exploitation late."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 (start) → 0.0 (end)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return initial_lr * (settings.LR_COSINE_MIN_FRACTION + (1.0 - settings.LR_COSINE_MIN_FRACTION) * cosine_factor)
    return schedule


def _safe_mtime(path):
    return int(os.path.getmtime(path)) if path and os.path.exists(path) else None


def _train_dataset_signature():
    return {
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "train_start_date": settings.TRAIN_START_DATE,
        "train_end_date": settings.TRAIN_END_DATE,
        "features_list": settings.FEATURES_LIST,
        "use_news": settings.USE_NEWS_FEATURES,
        "use_macro": settings.USE_MACRO_FEATURES,
        "use_time": settings.USE_TIME_FEATURES,
        "rsi_window": settings.RSI_WINDOW,
        "macd_fast": settings.MACD_FAST,
        "macd_slow": settings.MACD_SLOW,
        "macd_signal": settings.MACD_SIGNAL,
        "volatility_window": settings.VOLATILITY_WINDOW,
        "news_ema_span": settings.NEWS_EMA_SPAN,
        "raw_prices_mtime": _safe_mtime(settings.RAW_PRICES_CSV),
        "raw_news_mtime": _safe_mtime(settings.RAW_NEWS_CSV),
        "raw_macro_mtime": _safe_mtime(settings.RAW_MACRO_CSV),
    }


def _load_train_checkpoint_if_compatible():
    if not os.path.exists(settings.TRAIN_FEATURES_CSV):
        return None
    if not os.path.exists(settings.TRAIN_FEATURES_SIGNATURE_JSON):
        print("⚠️ Train checkpoint signature missing; rebuilding train features.")
        return None

    with open(settings.TRAIN_FEATURES_SIGNATURE_JSON, "r") as f:
        saved_signature = json.load(f)

    current_signature = _train_dataset_signature()
    if saved_signature != current_signature:
        print("⚠️ Train checkpoint signature mismatch; rebuilding train features.")
        return None

    print("⚡ Loaded train features from compatible checkpoint.")
    return pd.read_csv(settings.TRAIN_FEATURES_CSV, parse_dates=["Date"])


def _write_train_signature():
    os.makedirs(os.path.dirname(settings.TRAIN_FEATURES_SIGNATURE_JSON), exist_ok=True)
    with open(settings.TRAIN_FEATURES_SIGNATURE_JSON, "w") as f:
        json.dump(_train_dataset_signature(), f, indent=2)


def _resolve_ppo_params():
    """Use Optuna best params when enabled, otherwise fallback to settings defaults."""
    params = {
        "learning_rate": settings.LEARNING_RATE,
        "ent_coef":      settings.ENT_COEF,
        "batch_size":    settings.BATCH_SIZE,
        "gamma":         settings.GAMMA,
    }

    if not settings.USE_OPTUNA_BEST_PARAMS:
        return params

    if not os.path.exists(settings.OPTUNA_DB_PATH):
        print("⚠️ USE_OPTUNA_BEST_PARAMS=True but Optuna DB not found; using settings defaults.")
        return params

    try:
        import optuna

        storage = f"sqlite:///{settings.OPTUNA_DB_PATH}"
        study = optuna.load_study(study_name=settings.OPTUNA_STUDY_NAME, storage=storage)
        best = study.best_trial.params

        # Only override keys known by train script.
        if "learning_rate" in best:
            params["learning_rate"] = float(best["learning_rate"])
        if "ent_coef" in best:
            params["ent_coef"] = float(best["ent_coef"])
        if "batch_size" in best:
            params["batch_size"] = int(best["batch_size"])
        if "gamma" in best:
            params["gamma"] = float(best["gamma"])

        print(f"✅ Loaded Optuna best params from {settings.OPTUNA_DB_PATH}")
    except Exception as exc:
        print(f"⚠️ Could not load Optuna best params ({exc}); using settings defaults.")

    return params


def run_training():
    print(f"🚀 INITIATING TRAINING: {settings.EXPERIMENT_NAME}")
    os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)

    # -------------------------------------------------------
    # 1. Build or load the training dataset
    # Calls the full processor pipeline:
    # raw prices + news → sentiment → merge → indicators → train slice → CSV
    # If checkpoints already exist, you can swap this for pd.read_csv()
    # to skip reprocessing on re-runs.
    # -------------------------------------------------------
    train_df = _load_train_checkpoint_if_compatible()
    if train_df is None:
        train_df = build_training_dataset()
        _write_train_signature()

    print(f"📅 Training data: {len(train_df)} rows | "
          f"{settings.TRAIN_START_DATE} → {settings.TRAIN_END_DATE}")

    # -------------------------------------------------------
    # 2. Scale features
    # Fits + saves the scaler to settings.SCALER_PATH.
    # Uses settings.FEATURES_LIST — no hardcoded list here.
    # -------------------------------------------------------
    scaled_features = prepare_features(train_df, is_training=True)

    # -------------------------------------------------------
    # 3. Build environment
    # N_STACK frames stacked → agent sees a short-term memory window.
    # -------------------------------------------------------
    base_env = TradingEnv(df=train_df, features=scaled_features)
    vec_env  = DummyVecEnv([lambda: base_env])
    vec_env  = VecMonitor(vec_env)
    env      = VecFrameStack(vec_env, n_stack=settings.N_STACK)

    # -------------------------------------------------------
    # 4. Train
    # All PPO params come from settings — change them there, not here.
    # -------------------------------------------------------
    ppo_params = _resolve_ppo_params()
    print(f"DEBUG PPO PARAMS USED: {ppo_params}")
    tb_log_dir = os.path.join(settings.ARTIFACT_DIR, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=ppo_params["learning_rate"],
        ent_coef=ppo_params["ent_coef"],
        batch_size=ppo_params["batch_size"],
        gamma=ppo_params["gamma"],
        seed=settings.RANDOM_SEED,
        tensorboard_log=tb_log_dir,
    )
    model.learn(total_timesteps=settings.TOTAL_TIMESTEPS,
                tb_log_name="ppo")

    # -------------------------------------------------------
    # 5. Save model
    # -------------------------------------------------------
    model.save(settings.MODEL_PATH)
    print(f"🧠 Model saved → {settings.MODEL_PATH}")

    # -------------------------------------------------------
    # 6. Save MLOps metadata receipt
    # Everything needed to reproduce this exact run.
    # -------------------------------------------------------
    metadata = {
        "experiment_name":    settings.EXPERIMENT_NAME,
        "symbol":             settings.SYMBOL,
        "timeframe":          settings.TIMEFRAME,
        "action_space":       settings.ACTION_SPACE_TYPE,
        "reward_strategy":    settings.REWARD_STRATEGY,
        "train_dates":        f"{settings.TRAIN_START_DATE} → {settings.TRAIN_END_DATE}",
        "test_dates":         f"{settings.TEST_START_DATE} → {settings.TEST_END_DATE}",
        "total_timesteps":    settings.TOTAL_TIMESTEPS,
        "learning_rate":      ppo_params["learning_rate"],
        "ent_coef":           ppo_params["ent_coef"],
        "batch_size":         ppo_params["batch_size"],
        "gamma":              ppo_params["gamma"],
        "n_stack":            settings.N_STACK,
        "random_seed":        settings.RANDOM_SEED,
        "cash_risk_fraction": settings.CASH_RISK_FRACTION,
        "features_used":      settings.FEATURES_LIST,
        "feature_count":      settings.EXPECTED_MARKET_FEATURES,
        "use_news":           settings.USE_NEWS_FEATURES,
        "use_macro":          settings.USE_MACRO_FEATURES,
        "use_time":           settings.USE_TIME_FEATURES,
    }

    with open(settings.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    log_training_run(metadata, settings.ARTIFACT_DIR)

    print(f"🧾 Metadata receipt → {settings.METADATA_PATH}")
    print(f"✅ Training complete. Artifacts in: {settings.ARTIFACT_DIR}")


if __name__ == "__main__":
    run_training()
