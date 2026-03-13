"""Lightweight experiment journal utilities for train/backtest run tracking."""

import os
import subprocess
from datetime import datetime

import pandas as pd

from config import settings


JOURNAL_CSV = os.path.join(settings.ARTIFACTS_BASE_DIR, "experiment_journal.csv")


def _now_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit_short():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=settings.BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_journal():
    if not os.path.exists(JOURNAL_CSV):
        return pd.DataFrame()
    return pd.read_csv(JOURNAL_CSV)


def _write_journal(df):
    os.makedirs(os.path.dirname(JOURNAL_CSV), exist_ok=True)
    df.to_csv(JOURNAL_CSV, index=False)


def _upsert_row(run_id, payload):
    df = _read_journal()
    if "run_id" not in df.columns or df.empty:
        df = pd.DataFrame([{"run_id": run_id, **payload}])
        _write_journal(df)
        return

    if (df["run_id"] == run_id).any():
        idx = df.index[df["run_id"] == run_id][-1]
        for key, value in payload.items():
            df.loc[idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([{"run_id": run_id, **payload}])], ignore_index=True)

    _write_journal(df)


def log_training_run(metadata, artifact_dir):
    run_id = metadata.get("experiment_name", os.path.basename(artifact_dir))
    payload = {
        "updated_at_utc": _now_utc(),
        "stage": "trained",
        "symbol": metadata.get("symbol"),
        "timeframe": metadata.get("timeframe"),
        "action_space": metadata.get("action_space"),
        "reward_strategy": metadata.get("reward_strategy"),
        "use_news": metadata.get("use_news"),
        "use_macro": metadata.get("use_macro"),
        "use_time": metadata.get("use_time"),
        "feature_count": metadata.get("feature_count"),
        "features_used": "|".join(metadata.get("features_used", [])),
        "cash_risk_fraction": metadata.get("cash_risk_fraction"),
        "total_timesteps": metadata.get("total_timesteps"),
        "learning_rate": metadata.get("learning_rate"),
        "ent_coef": metadata.get("ent_coef"),
        "n_stack": metadata.get("n_stack"),
        "random_seed": metadata.get("random_seed"),
        "train_dates": metadata.get("train_dates"),
        "test_dates": metadata.get("test_dates"),
        "artifact_dir": artifact_dir,
        "model_path": os.path.join(artifact_dir, "model.zip"),
        "scaler_path": os.path.join(artifact_dir, "scaler.pkl"),
        "metadata_path": os.path.join(artifact_dir, "metadata.json"),
        "ledger_path": os.path.join(artifact_dir, "backtest_ledger.csv"),
        "git_commit": _git_commit_short(),
    }
    _upsert_row(run_id, payload)


def log_backtest_result(run_id, final_portfolio_value, total_return_pct, logged_events, ledger_path):
    payload = {
        "updated_at_utc": _now_utc(),
        "stage": "backtested",
        "backtest_final_value": float(final_portfolio_value),
        "backtest_total_return_pct": float(total_return_pct),
        "backtest_logged_events": int(logged_events),
        "ledger_path": ledger_path,
    }
    _upsert_row(run_id, payload)
