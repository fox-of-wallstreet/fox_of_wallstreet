"""
Build a master comparison table across all experiment artifact folders.

Priority:
1. Use backtest_summary.json if present
2. Otherwise fall back to metadata.json + backtest_ledger.csv
"""

import os
import sys
import json
import pandas as pd

# Ensure Python can find your folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.tools import fnline


ARTIFACTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts"
)

MASTER_SUMMARY_CSV = os.path.join(ARTIFACTS_ROOT, "experiment_summary.csv")


def safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_bar_hours_from_metadata(metadata: dict) -> int:
    timeframe = (
        metadata.get("experiment", {}).get("timeframe")
        or metadata.get("timeframe")
    )

    if timeframe == "1h":
        return 1
    if timeframe == "1d":
        return 24
    return 1


def get_min_position_threshold(metadata: dict) -> float:
    return float(
        metadata.get("environment", {}).get("min_position_threshold", 1e-8)
    )


def recompute_from_ledger(ledger_path: str, metadata: dict) -> dict:
    """
    Fallback if backtest_summary.json is absent.
    Recompute transaction and cycle metrics from backtest_ledger.csv.
    """
    metrics = {
        "total_real_transactions": 0,
        "trades_per_100_bars": None,
        "n_completed_cycles": 0,
        "avg_holding_duration_bars": 0.0,
        "avg_pnl_per_cycle_dollars": 0.0,
        "avg_return_per_cycle_pct": 0.0,
        "metric_mode": None,
    }

    if not os.path.exists(ledger_path):
        return metrics

    df = pd.read_csv(ledger_path)
    if df.empty:
        return metrics

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    metrics["total_real_transactions"] = len(df)

    threshold = get_min_position_threshold(metadata)
    bar_hours = get_bar_hours_from_metadata(metadata)

    cycle_pnls = []
    cycle_returns = []
    cycle_durations = []

    entry_row = None

    for _, row in df.iterrows():
        pos_before = float(row["Position_Before"])
        pos_after = float(row["Position_After"])

        opened_from_flat = (pos_before <= threshold) and (pos_after > threshold)
        closed_to_flat = (pos_before > threshold) and (pos_after <= threshold)

        if opened_from_flat:
            entry_row = row

        elif closed_to_flat and entry_row is not None:
            exit_row = row

            entry_time = entry_row["Date"]
            exit_time = exit_row["Date"]

            entry_portfolio = float(entry_row["Portfolio_Value"])
            exit_portfolio = float(exit_row["Portfolio_Value"])

            pnl_dollars = exit_portfolio - entry_portfolio
            return_pct = (
                ((exit_portfolio - entry_portfolio) / entry_portfolio) * 100
                if entry_portfolio != 0 else 0.0
            )

            duration_bars = (
                (exit_time - entry_time).total_seconds() / (3600 * bar_hours)
            )

            cycle_pnls.append(pnl_dollars)
            cycle_returns.append(return_pct)
            cycle_durations.append(duration_bars)

            entry_row = None

    metrics["n_completed_cycles"] = len(cycle_pnls)

    if cycle_pnls:
        metrics["avg_holding_duration_bars"] = sum(cycle_durations) / len(cycle_durations)
        metrics["avg_pnl_per_cycle_dollars"] = sum(cycle_pnls) / len(cycle_pnls)
        metrics["avg_return_per_cycle_pct"] = sum(cycle_returns) / len(cycle_returns)

    action_space = (
        metadata.get("experiment", {}).get("action_space_type")
        or metadata.get("action_space_type")
    )
    metrics["metric_mode"] = (
        "trade_metrics" if action_space == "discrete_3" else "position_episode_metrics"
    )

    return metrics


def build_row_from_artifact(artifact_dir: str) -> dict | None:
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    summary_path = os.path.join(artifact_dir, "backtest_summary.json")
    ledger_path = os.path.join(artifact_dir, "backtest_ledger.csv")

    metadata = safe_load_json(metadata_path)
    summary = safe_load_json(summary_path)

    if not metadata and not summary and not os.path.exists(ledger_path):
        return None

    experiment = metadata.get("experiment", {})
    data_split = metadata.get("data_split", {})
    dataset_stats = metadata.get("dataset_statistics", {})
    training = metadata.get("training", {})
    ppo = metadata.get("ppo_hyperparameters", {})

    row = {
        "experiment_name": experiment.get("experiment_name", os.path.basename(artifact_dir)),
        "artifact_dir": os.path.basename(artifact_dir),
        "symbol": experiment.get("symbol"),
        "timeframe": experiment.get("timeframe"),
        "action_space_type": experiment.get("action_space_type"),
        "reward_strategy": experiment.get("reward_strategy"),
        "train_start_date": data_split.get("train_start_date"),
        "train_end_date": data_split.get("train_end_date"),
        "test_start_date": data_split.get("test_start_date"),
        "test_end_date": data_split.get("test_end_date"),
        "raw_rows_loaded": dataset_stats.get("raw_rows_loaded"),
        "rows_after_feature_engineering": dataset_stats.get("rows_after_feature_engineering"),
        "train_rows": dataset_stats.get("train_rows"),
        "validation_rows": dataset_stats.get("validation_rows"),
        "total_rows_used": dataset_stats.get("total_rows_used"),
        "total_timesteps": training.get("total_timesteps"),
        "cash_risk_fraction": training.get("cash_risk_fraction"),
        "ppo_learning_rate": ppo.get("learning_rate"),
        "ppo_batch_size": ppo.get("batch_size"),
        "ppo_gamma": ppo.get("gamma"),
        "ppo_ent_coef": ppo.get("ent_coef"),
        "has_metadata": os.path.exists(metadata_path),
        "has_backtest_summary": os.path.exists(summary_path),
        "has_ledger": os.path.exists(ledger_path),
    }

    if summary:
        core = summary.get("core_metrics", {})
        cycle = summary.get("cycle_metrics", {})
        period = summary.get("backtest_period", {})

        row.update({
            "backtest_start": period.get("start"),
            "backtest_end": period.get("end"),
            "bars_evaluated": period.get("bars_evaluated"),
            "final_portfolio_value": core.get("final_portfolio_value"),
            "total_return_pct": core.get("total_return_pct"),
            "total_real_transactions": core.get("total_real_transactions"),
            "trades_per_100_bars": core.get("trades_per_100_bars"),
            "n_completed_cycles": cycle.get("n_completed_cycles"),
            "avg_holding_duration_bars": cycle.get("avg_holding_duration_bars"),
            "avg_pnl_per_cycle_dollars": cycle.get("avg_pnl_per_cycle_dollars"),
            "avg_return_per_cycle_pct": cycle.get("avg_return_per_cycle_pct"),
            "metric_mode": summary.get("metric_mode"),
            "summary_source": "backtest_summary.json",
        })
    else:
        fallback = recompute_from_ledger(ledger_path, metadata)
        row.update({
            "backtest_start": None,
            "backtest_end": None,
            "bars_evaluated": None,
            "final_portfolio_value": None,
            "total_return_pct": None,
            "total_real_transactions": fallback["total_real_transactions"],
            "trades_per_100_bars": fallback["trades_per_100_bars"],
            "n_completed_cycles": fallback["n_completed_cycles"],
            "avg_holding_duration_bars": fallback["avg_holding_duration_bars"],
            "avg_pnl_per_cycle_dollars": fallback["avg_pnl_per_cycle_dollars"],
            "avg_return_per_cycle_pct": fallback["avg_return_per_cycle_pct"],
            "metric_mode": fallback["metric_mode"],
            "summary_source": "recomputed_from_ledger",
        })

    return row


def scan_artifacts() -> list[dict]:
    rows = []

    if not os.path.exists(ARTIFACTS_ROOT):
        return rows

    for name in sorted(os.listdir(ARTIFACTS_ROOT)):
        artifact_dir = os.path.join(ARTIFACTS_ROOT, name)

        if not os.path.isdir(artifact_dir):
            continue

        row = build_row_from_artifact(artifact_dir)
        if row is not None:
            rows.append(row)

    return rows


def main():
    print(fnline(), "📊 Building master experiment summary table...")

    rows = scan_artifacts()

    if not rows:
        print(fnline(), "⚠️ No valid experiment artifacts found.")
        return

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["symbol", "timeframe", "action_space_type", "total_timesteps"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    df.to_csv(MASTER_SUMMARY_CSV, index=False)

    print(fnline(), f"✅ Master summary saved to: {MASTER_SUMMARY_CSV}")
    print(fnline(), f"📦 Experiments included: {len(df)}")

    preview_cols = [
        "experiment_name",
        "timeframe",
        "action_space_type",
        "reward_strategy",
        "total_timesteps",
        "total_return_pct",
        "total_real_transactions",
        "summary_source",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print()
    print(df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
