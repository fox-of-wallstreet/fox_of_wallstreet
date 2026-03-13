<<<<<<< HEAD
"""
Run an out-of-sample backtest for a trained PPO trading agent, save a trade
ledger, compute summary metrics from that ledger, and optionally generate
a backtest action plot.
"""
=======
"""Backtest pipeline aligned with the modular processor architecture."""
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

import os
import sys
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.experiment_journal import log_backtest_result
from core.environment import TradingEnv
<<<<<<< HEAD
from core.tools import fnline, get_features_list, get_stack_size


def get_bar_hours() -> int:
    """
    Return the approximate duration of one bar in hours.
    Used to convert holding durations into bar counts.
    """
    if settings.TIMEFRAME == "1h":
        return 1
    if settings.TIMEFRAME == "1d":
        return 24
    raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")


def get_action_map() -> dict:
    """
    Return action labels according to the configured action space.
    """
    if settings.ACTION_SPACE_TYPE == "discrete_3":
        return {
            0: "SELL_ALL",
            1: "BUY_ALL",
            2: "HOLD",
        }

    if settings.ACTION_SPACE_TYPE == "discrete_5":
        return {
            0: "SELL_100",
            1: "SELL_50",
            2: "HOLD",
            3: "BUY_50",
            4: "BUY_100",
        }

    raise ValueError(f"Unsupported ACTION_SPACE_TYPE: {settings.ACTION_SPACE_TYPE}")


def analyze_trade_ledger(ledger_path: str) -> dict:
    """
    Recompute metrics directly from the saved trade ledger.

    Universal metrics are valid for all configurations:
    - n_transactions

    Trade-cycle metrics:
    - For discrete_3: flat -> invested -> flat corresponds closely to one exact trade.
    - For discrete_5: flat -> scaled position -> flat is interpreted as one
      flat-to-flat position episode.

    Expected ledger columns:
    - Date
    - Action
    - Price
    - Portfolio_Value
    - Position_Before
    - Position_After
    """
    metrics = {
        "n_transactions": 0,
        "n_completed_cycles": 0,
        "avg_holding_duration_bars": 0.0,
        "avg_pnl_per_cycle_dollars": 0.0,
        "avg_return_per_cycle_pct": 0.0,
    }

    if not os.path.exists(ledger_path):
        return metrics

    df_trades = pd.read_csv(ledger_path)
    if df_trades.empty:
        return metrics

    df_trades["Date"] = pd.to_datetime(df_trades["Date"], utc=True)
    metrics["n_transactions"] = len(df_trades)

    threshold = settings.MIN_POSITION_THRESHOLD
    bar_hours = get_bar_hours()

    cycle_pnls = []
    cycle_returns = []
    cycle_durations = []

    entry_row = None

    for _, row in df_trades.iterrows():
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

            duration_timedelta = exit_time - entry_time
            duration_bars = duration_timedelta.total_seconds() / (3600 * bar_hours)

            cycle_pnls.append(pnl_dollars)
            cycle_returns.append(return_pct)
            cycle_durations.append(duration_bars)

            entry_row = None

    metrics["n_completed_cycles"] = len(cycle_pnls)

    if cycle_pnls:
        metrics["avg_pnl_per_cycle_dollars"] = sum(cycle_pnls) / len(cycle_pnls)
        metrics["avg_return_per_cycle_pct"] = sum(cycle_returns) / len(cycle_returns)
        metrics["avg_holding_duration_bars"] = sum(cycle_durations) / len(cycle_durations)

    return metrics


def plot_backtest_actions(test_df: pd.DataFrame, ledger_path: str, save_path: str) -> None:
    """
    Plot the stock close price over the backtest window and overlay executed
    buy/sell actions from the saved ledger.

    Supports both discrete_3 and discrete_5:
    - All BUY actions are plotted with upward green triangles.
    - All SELL actions are plotted with downward red triangles.
    - For discrete_5, strong actions use a larger marker size than light actions.
    """
    if not os.path.exists(ledger_path):
        print(fnline(), "⚠️ No ledger found. Skipping backtest plot.")
        return

    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        print(fnline(), "⚠️ Ledger is empty. Skipping backtest plot.")
        return

    df_plot = test_df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"], utc=True)

    ledger["Date"] = pd.to_datetime(ledger["Date"], utc=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df_plot["Date"], df_plot["Close"], label="Close Price")

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        buy_mask = ledger["Action"].eq("BUY_ALL")
        sell_mask = ledger["Action"].eq("SELL_ALL")

        buys = ledger.loc[buy_mask]
        sells = ledger.loc[sell_mask]

        if not buys.empty:
            ax.scatter(
                buys["Date"],
                buys["Price"],
                marker="^",
                s=110,
                label="Buy"
            )

        if not sells.empty:
            ax.scatter(
                sells["Date"],
                sells["Price"],
                marker="v",
                s=110,
                label="Sell"
            )

    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        buy_50 = ledger.loc[ledger["Action"].eq("BUY_50")]
        buy_100 = ledger.loc[ledger["Action"].eq("BUY_100")]
        sell_50 = ledger.loc[ledger["Action"].eq("SELL_50")]
        sell_100 = ledger.loc[ledger["Action"].eq("SELL_100")]

        if not buy_50.empty:
            ax.scatter(
                buy_50["Date"],
                buy_50["Price"],
                marker="^",
                s=70,
                label="Buy 50%"
            )

        if not buy_100.empty:
            ax.scatter(
                buy_100["Date"],
                buy_100["Price"],
                marker="^",
                s=130,
                label="Buy 100%"
            )

        if not sell_50.empty:
            ax.scatter(
                sell_50["Date"],
                sell_50["Price"],
                marker="v",
                s=70,
                label="Sell 50%"
            )

        if not sell_100.empty:
            ax.scatter(
                sell_100["Date"],
                sell_100["Price"],
                marker="v",
                s=130,
                label="Sell 100%"
            )

    ax.set_title(f"Backtest Actions - {settings.EXPERIMENT_NAME}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(fnline(), f"📉 Backtest action plot saved to Vault: {save_path}")


def run_backtest():
    """
    Load the trained PPO model, run it on the configured test period,
    save the backtest ledger, compute out-of-sample metrics, and optionally
    save a backtest action plot.
    """
    print(fnline(), f"🧪 STARTING FINAL EXAM: {settings.EXPERIMENT_NAME}")
=======
from core.processor import (
    add_technical_indicators,
    build_news_sentiment,
    load_raw_macro,
    load_raw_news,
    load_raw_prices,
    merge_prices_news_macro,
)

>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

def _get_bar_hours() -> int:
    if settings.TIMEFRAME == "1h":
        return 1
    if settings.TIMEFRAME == "1d":
        return 24
    raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

<<<<<<< HEAD
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # Slice to testing dates
    mask = (
        (df["Date"] >= pd.to_datetime(settings.TEST_START_DATE, utc=True))
        & (df["Date"] <= pd.to_datetime(settings.TEST_END_DATE, utc=True))
    )
    test_df = df.loc[mask].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError(
            f"❌ Test dataframe is empty! Check TEST_START_DATE ({settings.TEST_START_DATE}) "
            f"and TEST_END_DATE ({settings.TEST_END_DATE})."
        )

    print(fnline(), f"📅 Testing Data: {len(test_df)} rows loaded.")

    # Process features
    test_df = add_technical_indicators(test_df)

    if test_df.empty:
        raise ValueError(
            "❌ Test dataframe is empty after preprocessing. "
            "Check rolling windows and test date range."
        )

    features_list = get_features_list()
    stack_size = get_stack_size()

    # Load scaler fitted during training
    scaled_features = prepare_features(test_df, features_list, is_training=False)

    # Build environment
    base_env = TradingEnv(df=test_df, features=scaled_features)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(vec_env, n_stack=stack_size)

    # Load trained model
    model_path = settings.MODEL_PATH
    print(fnline(), f"🧠 Loading trained model from {model_path}.zip")

    if os.path.isfile(f"{model_path}.zip"):
        model = PPO.load(model_path, env=env)
    else:
        print(fnline(), f"❌ Cannot PPO.load({model_path}): No such file or directory.")
        return
=======

def _analyze_trade_ledger(ledger_path: str) -> dict:
    """
    Compute ledger-derived cycle metrics.
    For discrete_5 this is interpreted as flat-to-flat position episodes.
    """
    metrics = {
        "n_transactions": 0,
        "n_completed_cycles": 0,
        "avg_holding_duration_bars": 0.0,
        "avg_pnl_per_cycle_dollars": 0.0,
        "avg_return_per_cycle_pct": 0.0,
    }

    if not os.path.exists(ledger_path):
        return metrics

    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        return metrics

    required_cols = {
        "Date",
        "Portfolio_Value",
        "Position_Before",
        "Position_After",
    }
    if not required_cols.issubset(set(ledger.columns)):
        return metrics

    ledger["Date"] = pd.to_datetime(ledger["Date"], errors="coerce")
    ledger = ledger.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    metrics["n_transactions"] = int(len(ledger))

    threshold = 1e-10
    bar_hours = _get_bar_hours()

    cycle_pnls = []
    cycle_returns = []
    cycle_durations = []

    entry_row = None
    for _, row in ledger.iterrows():
        pos_before = float(row["Position_Before"])
        pos_after = float(row["Position_After"])

        opened_from_flat = (pos_before <= threshold) and (pos_after > threshold)
        closed_to_flat = (pos_before > threshold) and (pos_after <= threshold)

        if opened_from_flat:
            entry_row = row
        elif closed_to_flat and entry_row is not None:
            entry_portfolio = float(entry_row["Portfolio_Value"])
            exit_portfolio = float(row["Portfolio_Value"])
            pnl_dollars = exit_portfolio - entry_portfolio
            return_pct = (
                ((exit_portfolio - entry_portfolio) / (entry_portfolio + 1e-8)) * 100.0
            )

            duration_td = row["Date"] - entry_row["Date"]
            duration_bars = duration_td.total_seconds() / (3600.0 * bar_hours)

            cycle_pnls.append(pnl_dollars)
            cycle_returns.append(return_pct)
            cycle_durations.append(duration_bars)
            entry_row = None

    metrics["n_completed_cycles"] = int(len(cycle_pnls))
    if cycle_pnls:
        metrics["avg_pnl_per_cycle_dollars"] = float(sum(cycle_pnls) / len(cycle_pnls))
        metrics["avg_return_per_cycle_pct"] = float(sum(cycle_returns) / len(cycle_returns))
        metrics["avg_holding_duration_bars"] = float(sum(cycle_durations) / len(cycle_durations))

    return metrics


def _maybe_plot_backtest_actions(test_df: pd.DataFrame, ledger_path: str, save_path: str) -> None:
    if not getattr(settings, "PLOT_BACKTEST", False):
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠️ Plot requested but matplotlib is unavailable: {exc}")
        return

    if not os.path.exists(ledger_path):
        print("⚠️ No ledger found. Skipping backtest plot.")
        return

    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        print("⚠️ Ledger is empty. Skipping backtest plot.")
        return

    df_plot = test_df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")
    ledger["Date"] = pd.to_datetime(ledger["Date"], errors="coerce")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_plot["Date"], df_plot["Close"], label="Close Price")

    buys = ledger[ledger["Action"].str.contains("BUY", na=False)]
    sells = ledger[ledger["Action"].str.contains("SELL", na=False)]

    if not buys.empty:
        ax.scatter(buys["Date"], buys["Price"], marker="^", s=90, label="Buy")
    if not sells.empty:
        ax.scatter(sells["Date"], sells["Price"], marker="v", s=90, label="Sell")

    ax.set_title(f"Backtest Actions - {settings.EXPERIMENT_NAME}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📉 Backtest action plot saved to {save_path}")


def _ensure_reports_dirs(run_dir: str) -> dict:
    reports_dir = os.path.join(run_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    tables_dir = os.path.join(reports_dir, "tables")
    summary_dir = os.path.join(reports_dir, "summary")

    for d in (reports_dir, figures_dir, tables_dir, summary_dir):
        os.makedirs(d, exist_ok=True)

    return {
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "summary_dir": summary_dir,
    }


def _extract_cycle_returns(ledger: pd.DataFrame, threshold: float = 1e-10):
    required_cols = {"Date", "Portfolio_Value", "Position_Before", "Position_After"}
    if ledger.empty or not required_cols.issubset(set(ledger.columns)):
        return []

    ordered = ledger.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
    ordered = ordered.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    cycle_returns = []
    entry_row = None
    for _, row in ordered.iterrows():
        pos_before = float(row["Position_Before"])
        pos_after = float(row["Position_After"])
        opened_from_flat = (pos_before <= threshold) and (pos_after > threshold)
        closed_to_flat = (pos_before > threshold) and (pos_after <= threshold)

        if opened_from_flat:
            entry_row = row
        elif closed_to_flat and entry_row is not None:
            entry_portfolio = float(entry_row["Portfolio_Value"])
            exit_portfolio = float(row["Portfolio_Value"])
            ret_pct = ((exit_portfolio - entry_portfolio) / (entry_portfolio + 1e-8)) * 100.0
            cycle_returns.append(ret_pct)
            entry_row = None

    return cycle_returns


def _write_backtest_reports(equity_df: pd.DataFrame, ledger_path: str, reports_paths: dict) -> dict:
    generated = {
        "equity_timeseries_csv": None,
        "actions_overlay_png": None,
        "equity_vs_benchmark_png": None,
        "drawdown_curve_png": None,
        "trade_return_hist_png": None,
    }

    equity_csv = os.path.join(reports_paths["tables_dir"], "equity_timeseries.csv")
    equity_df.to_csv(equity_csv, index=False)
    generated["equity_timeseries_csv"] = equity_csv

    if equity_df.empty or "Date" not in equity_df.columns:
        return generated

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠️ Skipping plots: matplotlib unavailable ({exc})")
        return generated

    df_plot = equity_df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")
    df_plot = df_plot.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if df_plot.empty:
        return generated

    ledger = pd.read_csv(ledger_path) if os.path.exists(ledger_path) else pd.DataFrame()
    if not ledger.empty and "Date" in ledger.columns:
        ledger["Date"] = pd.to_datetime(ledger["Date"], errors="coerce")

    # 1) Actions overlay
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_plot["Date"], df_plot["Close"], label="Close Price")
    if not ledger.empty:
        buy_50 = ledger[ledger["Action"] == "BUY_50"]
        buy_100 = ledger[ledger["Action"] == "BUY_100"]
        sell_50 = ledger[ledger["Action"] == "SELL_50"]
        sell_100 = ledger[ledger["Action"] == "SELL_100"]
        forced = ledger[ledger["Action"] == "FORCED_SL_TP"]

        if not buy_50.empty:
            ax.scatter(buy_50["Date"], buy_50["Price"], marker="^", s=70, label="BUY_50")
        if not buy_100.empty:
            ax.scatter(buy_100["Date"], buy_100["Price"], marker="^", s=120, label="BUY_100")
        if not sell_50.empty:
            ax.scatter(sell_50["Date"], sell_50["Price"], marker="v", s=70, label="SELL_50")
        if not sell_100.empty:
            ax.scatter(sell_100["Date"], sell_100["Price"], marker="v", s=120, label="SELL_100")
        if not forced.empty:
            ax.scatter(forced["Date"], forced["Price"], marker="x", s=65, label="FORCED_SL_TP")

    ax.set_title(f"Backtest Actions - {settings.EXPERIMENT_NAME}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    actions_png = os.path.join(reports_paths["figures_dir"], "actions_overlay.png")
    plt.savefig(actions_png, dpi=150)
    plt.close()
    generated["actions_overlay_png"] = actions_png

    # 2) Equity vs benchmark
    initial_equity = float(df_plot["Portfolio_Value"].iloc[0])
    initial_price = float(df_plot["Close"].iloc[0])
    df_plot["Portfolio_Index"] = (df_plot["Portfolio_Value"] / (initial_equity + 1e-8)) * 100.0
    df_plot["BuyHold_Index"] = (df_plot["Close"] / (initial_price + 1e-8)) * 100.0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_plot["Date"], df_plot["Portfolio_Index"], label="Portfolio (Index=100)")
    ax.plot(df_plot["Date"], df_plot["BuyHold_Index"], label="Buy & Hold TSLA (Index=100)")
    ax.set_title("Equity vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    equity_png = os.path.join(reports_paths["figures_dir"], "equity_vs_benchmark.png")
    plt.savefig(equity_png, dpi=150)
    plt.close()
    generated["equity_vs_benchmark_png"] = equity_png

    # 3) Drawdown
    running_max = df_plot["Portfolio_Value"].cummax()
    drawdown_pct = ((df_plot["Portfolio_Value"] - running_max) / (running_max + 1e-8)) * 100.0
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df_plot["Date"], drawdown_pct, label="Drawdown %")
    ax.fill_between(df_plot["Date"], drawdown_pct, 0, alpha=0.2)
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(True)
    plt.tight_layout()
    drawdown_png = os.path.join(reports_paths["figures_dir"], "drawdown_curve.png")
    plt.savefig(drawdown_png, dpi=150)
    plt.close()
    generated["drawdown_curve_png"] = drawdown_png

    # 4) Trade return histogram
    if not ledger.empty:
        cycle_returns = _extract_cycle_returns(ledger)
        if cycle_returns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(cycle_returns, bins=20)
            ax.set_title("Cycle Return Distribution")
            ax.set_xlabel("Return per cycle (%)")
            ax.set_ylabel("Count")
            ax.grid(True)
            plt.tight_layout()
            hist_png = os.path.join(reports_paths["figures_dir"], "trade_return_hist.png")
            plt.savefig(hist_png, dpi=150)
            plt.close()
            generated["trade_return_hist_png"] = hist_png

    return generated


def _resolve_trained_artifact_paths():
    """Resolve model/ledger paths, handling timestamped experiment names across runs."""
    current_model_zip = f"{settings.MODEL_PATH}.zip"
    current_scaler = settings.SCALER_PATH
    current_metadata = settings.METADATA_PATH
    if os.path.exists(current_model_zip) and os.path.exists(current_scaler):
        return (
            settings.MODEL_PATH,
            settings.SCALER_PATH,
            settings.BACKTEST_LEDGER_PATH,
            current_metadata,
        )

    # Fallback: find latest compatible run directory produced by a prior training command.
    prefix = (
        f"ppo_{settings.SYMBOL}_{settings.TIMEFRAME}_{settings.ACTION_SPACE_TYPE}"
        f"_{'news' if settings.USE_NEWS_FEATURES else 'nonews'}"
        f"_{'macro' if settings.USE_MACRO_FEATURES else 'nomacro'}"
        f"_{'time' if settings.USE_TIME_FEATURES else 'notime'}_"
    )

    candidates = []
    for name in os.listdir(settings.ARTIFACTS_BASE_DIR):
        run_dir = os.path.join(settings.ARTIFACTS_BASE_DIR, name)
        model_zip = os.path.join(run_dir, "model.zip")
        scaler_pkl = os.path.join(run_dir, "scaler.pkl")
        if (
            os.path.isdir(run_dir)
            and name.startswith(prefix)
            and os.path.exists(model_zip)
            and os.path.exists(scaler_pkl)
        ):
            candidates.append((os.path.getmtime(model_zip), run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"❌ Trained model not found at {current_model_zip} and no compatible prior run found in "
            f"{settings.ARTIFACTS_BASE_DIR}. Run scripts/train.py first."
        )

    latest_run_dir = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    resolved_model_path = os.path.join(latest_run_dir, "model")
    resolved_scaler_path = os.path.join(latest_run_dir, "scaler.pkl")
    resolved_ledger_path = os.path.join(latest_run_dir, "backtest_ledger.csv")
    resolved_metadata_path = os.path.join(latest_run_dir, "metadata.json")
    print(f"ℹ️ Using latest compatible artifact run: {latest_run_dir}")
    return (
        resolved_model_path,
        resolved_scaler_path,
        resolved_ledger_path,
        resolved_metadata_path,
    )


def _validate_backtest_compatibility(metadata_path):
    """Fail fast when current runtime settings differ from training run settings."""
    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found at {metadata_path}; skipping compatibility check.")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    expected = {
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "action_space": settings.ACTION_SPACE_TYPE,
        "reward_strategy": settings.REWARD_STRATEGY,
        "n_stack": settings.N_STACK,
        "features_used": settings.FEATURES_LIST,
        "feature_count": settings.EXPECTED_MARKET_FEATURES,
        "use_news": settings.USE_NEWS_FEATURES,
        "use_macro": settings.USE_MACRO_FEATURES,
        "use_time": settings.USE_TIME_FEATURES,
        "cash_risk_fraction": settings.CASH_RISK_FRACTION,
    }

    mismatches = []
    for key, current_value in expected.items():
        trained_value = metadata.get(key)
        if trained_value != current_value:
            mismatches.append((key, trained_value, current_value))

    if mismatches:
        lines = [
            "❌ Backtest compatibility check failed.",
            "Current settings do not match the resolved training run metadata:",
        ]
        for key, trained, current in mismatches:
            lines.append(f"  - {key}: trained={trained} | current={current}")
        lines.append("Align config/settings.py with the trained run, or retrain with current settings.")
        raise ValueError("\n".join(lines))

    print("✅ Backtest compatibility check passed against training metadata.")


def _scale_with_resolved_scaler(df, scaler_path):
    """Scale features using scaler from resolved artifact run."""
    features_list = settings.FEATURES_LIST
    missing = [col for col in features_list if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing requested feature columns in test set: {missing}")

    data_to_scale = df[features_list].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(data_to_scale)
    return pd.DataFrame(scaled_data, columns=features_list, index=df.index)


def _safe_mtime(path):
    return int(os.path.getmtime(path)) if path and os.path.exists(path) else None


def _test_dataset_signature():
    return {
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "test_start_date": settings.TEST_START_DATE,
        "test_end_date": settings.TEST_END_DATE,
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


def _load_test_checkpoint_if_compatible():
    if not os.path.exists(settings.TEST_FEATURES_CSV):
        return None
    if not os.path.exists(settings.TEST_FEATURES_SIGNATURE_JSON):
        print("⚠️ Test checkpoint signature missing; rebuilding test features.")
        return None

    with open(settings.TEST_FEATURES_SIGNATURE_JSON, "r") as f:
        saved_signature = json.load(f)

    current_signature = _test_dataset_signature()
    if saved_signature != current_signature:
        print("⚠️ Test checkpoint signature mismatch; rebuilding test features.")
        return None

    print("⚡ Loaded test features from compatible checkpoint.")
    return pd.read_csv(settings.TEST_FEATURES_CSV, parse_dates=["Date"])


def _write_test_signature():
    os.makedirs(os.path.dirname(settings.TEST_FEATURES_SIGNATURE_JSON), exist_ok=True)
    with open(settings.TEST_FEATURES_SIGNATURE_JSON, "w") as f:
        json.dump(_test_dataset_signature(), f, indent=2)


def _build_or_load_test_dataset():
    """Load cached test features or build them from raw checkpoints."""
    cached = _load_test_checkpoint_if_compatible()
    if cached is not None:
        return cached

    # Rebuild test features from raw checkpoints using the same processor flow.
    prices_df = load_raw_prices()
    news_df = load_raw_news()
    macro_df = load_raw_macro()
    sentiment_df = build_news_sentiment(news_df, timeframe=settings.TIMEFRAME)
    merged_df = merge_prices_news_macro(prices_df, sentiment_df, macro_df)
    full_feature_df = add_technical_indicators(merged_df)

    test_start = pd.to_datetime(settings.TEST_START_DATE)
    test_end = pd.to_datetime(settings.TEST_END_DATE)
    test_df = full_feature_df[
        (full_feature_df["Date"] >= test_start) &
        (full_feature_df["Date"] <= test_end)
    ].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError(
            "❌ Test dataset is empty after processing and date filtering. "
            "Check TEST_START_DATE/TEST_END_DATE and raw checkpoints."
        )

    os.makedirs(os.path.dirname(settings.TEST_FEATURES_CSV), exist_ok=True)
    test_df.to_csv(settings.TEST_FEATURES_CSV, index=False)
    _write_test_signature()
    print(f"✅ Test features checkpoint saved to {settings.TEST_FEATURES_CSV}")
    return test_df


def run_backtest():
    """Run deterministic backtest using the trained model and saved scaler."""
    print(f"🧪 STARTING BACKTEST: {settings.EXPERIMENT_NAME}")

    model_base_path, scaler_path, ledger_path, metadata_path = _resolve_trained_artifact_paths()
    model_path = f"{model_base_path}.zip"
    run_id = os.path.basename(os.path.dirname(model_base_path))

    _validate_backtest_compatibility(metadata_path)

    test_df = _build_or_load_test_dataset()
    print(f"📅 Testing data: {len(test_df)} rows | {settings.TEST_START_DATE} → {settings.TEST_END_DATE}")

    # Use the scaler fitted in the same resolved training run.
    scaled_features = _scale_with_resolved_scaler(test_df, scaler_path)

    base_env = TradingEnv(df=test_df, features=scaled_features)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(vec_env, n_stack=settings.N_STACK)

    model = PPO.load(model_base_path, env=env)
    print(f"🧠 Loaded trained model from {model_path}")
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

    action_map = get_action_map()

    obs = env.reset()
    done = False
    trade_history = []
<<<<<<< HEAD
    step_info = None

    print(fnline(), "📈 Simulating live trading...")

    while not done:
        action, _states = model.predict(obs, deterministic=True)

        position_before = float(env.get_attr("position")[0])

        obs, rewards, done_array, info = env.step(action)
        done = done_array[0]
        step_info = info[0]

        actual_action = int(step_info["action"])
        position_after = float(env.get_attr("position")[0])

        # Record only real executed transactions where the position actually changed
        if (
            abs(position_after - position_before) > settings.MIN_POSITION_THRESHOLD
            and actual_action != 2
        ):
            step_idx = int(step_info["step"])
            step_idx = min(step_idx, len(test_df) - 1)

            trade_history.append({
                "Date": test_df.loc[step_idx, "Date"],
                "Action": action_map.get(actual_action, "UNKNOWN"),
                "Price": round(float(step_info["price"]), 2),
                "Portfolio_Value": round(float(step_info["portfolio_value"]), 2),
                "Position_Before": position_before,
                "Position_After": position_after,
            })

    if step_info is None:
        raise RuntimeError("❌ Backtest terminated before any environment step was completed.")

    initial_val = float(base_env.initial_balance)
    final_val = float(step_info["portfolio_value"])
    total_return = ((final_val - initial_val) / initial_val) * 100

    backtest_start = test_df["Date"].min()
    backtest_end = test_df["Date"].max()
    n_bars = len(test_df)

    # Save ledger first
    os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)
    ledger_path = os.path.join(settings.ARTIFACT_DIR, "backtest_ledger.csv")
=======
    equity_history = []
    last_step_info = None
    total_steps = 0

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        action_map = {0: "SELL_ALL", 1: "BUY_ALL", 2: "HOLD"}
    else:
        action_map = {
            0: "SELL_100",
            1: "SELL_50",
            2: "HOLD",
            3: "BUY_50",
            4: "BUY_100",
        }

    prev_position = float(env.get_attr("position")[0])
    print("📈 Simulating deterministic policy...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        position_before = float(env.get_attr("position")[0])
        obs, rewards, dones, infos = env.step(action)
        done = bool(dones[0])
        step_info = infos[0]
        last_step_info = step_info
        total_steps += 1

        step_idx = int(step_info["step"])
        if 0 <= step_idx < len(test_df):
            equity_history.append(
                {
                    "Date": test_df.iloc[step_idx]["Date"],
                    "Close": round(float(step_info["price"]), 6),
                    "Portfolio_Value": round(float(step_info["portfolio_value"]), 6),
                }
            )

        actual_action = int(step_info["action"])
        current_position = float(env.get_attr("position")[0])
        sl_tp_triggered = bool(step_info.get("sl_tp_triggered", False))
        position_changed = current_position != prev_position

        # Log explicit trade actions and forced SL/TP exits.
        if sl_tp_triggered or (position_changed and actual_action != 2):
            step_idx = int(step_info["step"])
            if step_idx < 0 or step_idx >= len(test_df):
                continue
            trade_history.append(
                {
                    "Date": test_df.iloc[step_idx]["Date"],
                    "Action": "FORCED_SL_TP" if sl_tp_triggered else action_map.get(actual_action, "UNKNOWN"),
                    "Price": round(float(step_info["price"]), 6),
                    "Portfolio_Value": round(float(step_info["portfolio_value"]), 6),
                    "Position_Before": round(position_before, 10),
                    "Position_After": round(current_position, 10),
                    "SL_TP_Triggered": sl_tp_triggered,
                }
            )

        prev_position = current_position

    if last_step_info is None:
        raise RuntimeError("❌ Backtest ended before any environment step was processed.")
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

    initial_val = float(base_env.initial_balance)
    final_val = float(last_step_info["portfolio_value"])
    total_return = ((final_val - initial_val) / (initial_val + 1e-8)) * 100.0

    print("=" * 60)
    print(f"🏆 BACKTEST RESULTS: {settings.EXPERIMENT_NAME}")
    print(f"Final Portfolio Value: ${final_val:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Logged Events: {len(trade_history)}")
    print("=" * 60)

    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    reports_paths = _ensure_reports_dirs(os.path.dirname(ledger_path))
    if trade_history:
        df_trades = pd.DataFrame(trade_history)
<<<<<<< HEAD
    else:
        df_trades = pd.DataFrame(columns=[
            "Date",
            "Action",
            "Price",
            "Portfolio_Value",
            "Position_Before",
            "Position_After",
        ])

    df_trades.to_csv(ledger_path, index=False)
    print(fnline(), f"💾 Backtest Ledger saved to Vault: {ledger_path}")

    # Optional plotting
    if getattr(settings, "PLOT_BACKTEST", False):
        plot_path = os.path.join(settings.ARTIFACT_DIR, "backtest_actions.png")
        plot_backtest_actions(test_df=test_df, ledger_path=ledger_path, save_path=plot_path)

    # Recompute metrics from saved ledger
    ledger_metrics = analyze_trade_ledger(ledger_path)

    n_transactions = ledger_metrics["n_transactions"]
    trades_per_100_bars = (n_transactions / n_bars) * 100 if n_bars > 0 else 0.0

        # Save compact backtest summary
    backtest_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": settings.EXPERIMENT_NAME,
=======
        df_trades.to_csv(ledger_path, index=False)
        print(f"💾 Ledger saved to {ledger_path}")
    else:
        pd.DataFrame(
            columns=[
                "Date",
                "Action",
                "Price",
                "Portfolio_Value",
                "Position_Before",
                "Position_After",
                "SL_TP_Triggered",
            ]
        ).to_csv(ledger_path, index=False)
        print("ℹ️ No trade events were logged for this backtest run.")

    # Legacy root-level plot, kept for backward compatibility if setting is enabled.
    _maybe_plot_backtest_actions(
        test_df=test_df,
        ledger_path=ledger_path,
        save_path=os.path.join(os.path.dirname(ledger_path), "backtest_actions.png"),
    )

    equity_df = pd.DataFrame(equity_history)
    report_artifacts = _write_backtest_reports(equity_df, ledger_path, reports_paths)
    print(f"📁 Backtest report bundle saved under {reports_paths['reports_dir']}")

    ledger_metrics = _analyze_trade_ledger(ledger_path)
    trades_per_100_bars = (ledger_metrics["n_transactions"] / total_steps) * 100 if total_steps > 0 else 0.0
    summary_path = os.path.join(os.path.dirname(ledger_path), "backtest_summary.json")
    report_index_path = os.path.join(reports_paths["summary_dir"], "report_index.json")

    backtest_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "action_space_type": settings.ACTION_SPACE_TYPE,
        "reward_strategy": settings.REWARD_STRATEGY,
<<<<<<< HEAD
        "backtest_period": {
            "start": backtest_start.isoformat(),
            "end": backtest_end.isoformat(),
            "bars_evaluated": n_bars,
        },
        "core_metrics": {
            "final_portfolio_value": round(final_val, 4),
            "total_return_pct": round(total_return, 4),
            "total_real_transactions": int(n_transactions),
            "trades_per_100_bars": round(trades_per_100_bars, 4),
        },
        "cycle_metrics": {
            "n_completed_cycles": int(ledger_metrics["n_completed_cycles"]),
            "avg_holding_duration_bars": round(ledger_metrics["avg_holding_duration_bars"], 4),
            "avg_pnl_per_cycle_dollars": round(ledger_metrics["avg_pnl_per_cycle_dollars"], 4),
            "avg_return_per_cycle_pct": round(ledger_metrics["avg_return_per_cycle_pct"], 4),
        },
        "metric_mode": (
            "trade_metrics"
            if settings.ACTION_SPACE_TYPE == "discrete_3"
            else "position_episode_metrics"
        ),
        "artifact_paths": {
            "artifact_dir": settings.ARTIFACT_DIR,
            "ledger_path": ledger_path,
            "summary_path": settings.BACKTEST_SUMMARY_PATH,
        },
    }

    with open(settings.BACKTEST_SUMMARY_PATH, "w") as f:
        json.dump(backtest_summary, f, indent=4)

    print(fnline(), f"🧾 Backtest summary saved to Vault: {settings.BACKTEST_SUMMARY_PATH}")

    # Print Final Experiments Metrices
    print(fnline(), "=" * 68)
    print(fnline(), f"🏆 BACKTEST RESULTS: {settings.EXPERIMENT_NAME} 🏆")
    print(
        fnline(),
        f"🗓️ Backtest Period: {backtest_start.strftime('%Y-%m-%d %H:%M')} "
        f"→ {backtest_end.strftime('%Y-%m-%d %H:%M')}"
    )
    print(fnline(), f"📏 Bars evaluated: {n_bars}")
    print(fnline(), f"Final Portfolio Value: ${final_val:.2f}")
    print(fnline(), f"Total Return: {total_return:.2f}%")
    print(fnline(), f"Total Real Transactions: {n_transactions}")
    print(fnline(), f"🔁 Trades per 100 bars: {trades_per_100_bars:.2f}")

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        print(fnline(), f"✅ Completed trades: {ledger_metrics['n_completed_cycles']}")
        print(
            fnline(),
            f"⏳ Average holding duration per trade: "
            f"{ledger_metrics['avg_holding_duration_bars']:.2f} bars"
        )
        print(
            fnline(),
            f"💵 Average realized PnL per trade: "
            f"${ledger_metrics['avg_pnl_per_cycle_dollars']:.2f}"
        )
        print(
            fnline(),
            f"📊 Average realized return per trade: "
            f"{ledger_metrics['avg_return_per_cycle_pct']:.2f}%"
        )

    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        print(
            fnline(),
            f"✅ Completed flat-to-flat position episodes: "
            f"{ledger_metrics['n_completed_cycles']}"
        )
        print(
            fnline(),
            f"⏳ Average holding duration per position episode: "
            f"{ledger_metrics['avg_holding_duration_bars']:.2f} bars"
        )
        print(
            fnline(),
            f"💵 Average realized PnL per position episode: "
            f"${ledger_metrics['avg_pnl_per_cycle_dollars']:.2f}"
        )
        print(
            fnline(),
            f"📊 Average realized return per position episode: "
            f"{ledger_metrics['avg_return_per_cycle_pct']:.2f}%"
        )
    print(fnline(), "=" * 68)
=======
        "test_dates": {
            "start": settings.TEST_START_DATE,
            "end": settings.TEST_END_DATE,
            "bars_evaluated": int(total_steps),
        },
        "core_metrics": {
            "final_portfolio_value": round(final_val, 6),
            "total_return_pct": round(total_return, 6),
            "logged_events": int(len(trade_history)),
            "trades_per_100_bars": round(trades_per_100_bars, 6),
        },
        "cycle_metrics": {
            "n_completed_cycles": int(ledger_metrics["n_completed_cycles"]),
            "avg_holding_duration_bars": round(ledger_metrics["avg_holding_duration_bars"], 6),
            "avg_pnl_per_cycle_dollars": round(ledger_metrics["avg_pnl_per_cycle_dollars"], 6),
            "avg_return_per_cycle_pct": round(ledger_metrics["avg_return_per_cycle_pct"], 6),
        },
        "artifact_paths": {
            "ledger_path": ledger_path,
            "summary_path": summary_path,
            "reports_dir": reports_paths["reports_dir"],
            "figures_dir": reports_paths["figures_dir"],
            "tables_dir": reports_paths["tables_dir"],
            "summary_dir": reports_paths["summary_dir"],
            "report_index_path": report_index_path,
            "report_artifacts": report_artifacts,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(backtest_summary, f, indent=2)
    with open(report_index_path, "w") as f:
        json.dump(backtest_summary, f, indent=2)
    print(f"🧾 Backtest summary saved to {summary_path}")
    print(f"🧾 Backtest report index saved to {report_index_path}")

    log_backtest_result(
        run_id=run_id,
        final_portfolio_value=final_val,
        total_return_pct=total_return,
        logged_events=len(trade_history),
        ledger_path=ledger_path,
    )
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485


if __name__ == "__main__":
    run_backtest()
