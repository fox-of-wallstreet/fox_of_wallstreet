"""
Run an out-of-sample backtest for a trained PPO trading agent, save a trade
ledger, compute summary metrics from that ledger, and optionally generate
a backtest action plot.
"""

import os
import os.path
import sys
import pandas as pd
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Ensure Python can find your folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators, prepare_features
from core.environment import TradingEnv
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

    csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Cannot find {csv_path}. Run data_engine.py first!")

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

    action_map = get_action_map()

    obs = env.reset()
    done = False
    trade_history = []
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

    if trade_history:
        df_trades = pd.DataFrame(trade_history)
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
        "symbol": settings.SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "action_space_type": settings.ACTION_SPACE_TYPE,
        "reward_strategy": settings.REWARD_STRATEGY,
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


if __name__ == "__main__":
    run_backtest()
