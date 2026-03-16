import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Ensure pathing is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from scripts.data_engine import build_and_save_dataset
from core.processor import add_technical_indicators, prepare_features
from core.tools import fnline, get_features_list, get_stack_size

# -----------------------------------------
# Helpers
# -----------------------------------------

def send_telegram_alert(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print(fnline(), "⚠️ Telegram credentials missing in .env. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(fnline(), f"❌ Failed to send Telegram alert: {e}")


def get_current_position_info(trading_client: TradingClient):
    """
    Return current shares, current cash, and approximate portfolio state features.
    """
    account = trading_client.get_account()
    cash = float(account.cash)

    try:
        position = trading_client.get_open_position(settings.SYMBOL)
        current_shares = float(position.qty)
        entry_price = float(position.avg_entry_price)
        unrealized_pnl_pct = float(position.unrealized_plpc) if position.unrealized_plpc is not None else 0.0
        in_position = 1.0
    except Exception:
        current_shares = 0.0
        entry_price = 0.0
        unrealized_pnl_pct = 0.0
        in_position = 0.0

    cash_ratio = cash / settings.INITIAL_BALANCE

    # bars_in_trade is not directly available from Alpaca position state.
    # We set it to 0.0 here; a more advanced live system would persist this locally.
    bars_in_trade_norm = 0.0

    portfolio_features = np.array([
        in_position,
        unrealized_pnl_pct,
        cash_ratio,
        bars_in_trade_norm
    ], dtype=np.float32)

    return current_shares, cash, entry_price, portfolio_features


def build_live_observation(scaled_features: pd.DataFrame, portfolio_features: np.ndarray):
    """
    Build a stacked observation compatible with PPO + VecFrameStack training.

    We stack the last N feature rows and append the current portfolio features
    to each frame. This approximates the training-time observation layout.
    """
    stack_size = get_stack_size()
    feature_array = scaled_features.values

    if len(feature_array) < stack_size:
        raise ValueError(
            f"Not enough processed rows to build a live observation. "
            f"Need at least {stack_size}, got {len(feature_array)}."
        )

    last_n = feature_array[-stack_size:]

    full_frames = []
    for row in last_n:
        full_frame = np.hstack([row, portfolio_features])
        full_frames.append(full_frame.astype(np.float32))

    # VecFrameStack for 1D observations effectively concatenates frames
    obs = np.hstack(full_frames).reshape(1, -1).astype(np.float32)
    return obs


def action_to_text(action: int) -> str:
    if settings.ACTION_SPACE_TYPE == "discrete_3":
        mapping = {
            0: "SELL ALL",
            1: "BUY ALL",
            2: "HOLD",
        }
    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        mapping = {
            0: "SELL 100%",
            1: "SELL 50%",
            2: "HOLD",
            3: "BUY 50%",
            4: "BUY 100%",
        }
    else:
        raise ValueError(f"Unsupported ACTION_SPACE_TYPE: {settings.ACTION_SPACE_TYPE}")

    return mapping.get(int(action), "UNKNOWN")


# -----------------------------------------
# Main Live Trader
# -----------------------------------------
def run_live_trader():
    print(fnline(), f"🟢 STARTING LIVE TRADER FOR {settings.SYMBOL} ({settings.TIMEFRAME}) 🟢")

    load_dotenv()

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

    trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True)

    # 1. Build / refresh the latest hybrid dataset
    print(fnline(), "📥 Refreshing latest hybrid dataset...")
    live_csv = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_live.csv"
    df = build_and_save_dataset(symbol=settings.SYMBOL, output_file=live_csv)

    # 2. Feature engineering
    df_proc = add_technical_indicators(df)
    if df_proc.empty:
        raise ValueError("Processed live dataframe is empty after feature engineering.")

    features_list = get_features_list()
    scaled_features = prepare_features(df_proc, features_list, is_training=False)

    # 3. Query current Alpaca position state
    current_shares, current_cash, entry_price, portfolio_features = get_current_position_info(trading_client)
    current_price = float(df_proc.iloc[-1]["Close"])

    print(fnline(), f"📊 Latest price: ${current_price:.2f}")
    print(fnline(), f"💼 Current shares: {current_shares:.6f}")
    print(fnline(), f"💵 Current cash: ${current_cash:.2f}")

    # 4. Build stacked live observation
    obs = build_live_observation(scaled_features, portfolio_features)

    # 5. Load model and predict discrete action
    print(fnline(), f"🧠 Loading model from {settings.MODEL_PATH}.zip")
    model = PPO.load(settings.MODEL_PATH)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)

    action_text = action_to_text(action)
    print(fnline(), f"🤖 PPO action: {action} -> {action_text}")

    # 6. Execute action
    executed = False
    message = None

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        if action == 1:  # BUY ALL
            if current_cash > 10.0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    notional=current_cash * settings.CASH_RISK_FRACTION,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🟢 *LIVE PPO BUY ALL*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Cash used: ${current_cash * settings.CASH_RISK_FRACTION:.2f}"
                )

        elif action == 0:  # SELL ALL
            if current_shares > 0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    qty=current_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🔴 *LIVE PPO SELL ALL*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Shares sold: {current_shares:.6f}"
                )

    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        if action == 4:  # BUY 100%
            buy_notional = current_cash * settings.CASH_RISK_FRACTION
            if buy_notional > 10.0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    notional=buy_notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🟢 *LIVE PPO BUY 100%*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Cash used: ${buy_notional:.2f}"
                )

        elif action == 3:  # BUY 50%
            buy_notional = current_cash * 0.5 * settings.CASH_RISK_FRACTION
            if buy_notional > 10.0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    notional=buy_notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🟢 *LIVE PPO BUY 50%*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Cash used: ${buy_notional:.2f}"
                )

        elif action == 0:  # SELL 100%
            if current_shares > 0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    qty=current_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🔴 *LIVE PPO SELL 100%*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Shares sold: {current_shares:.6f}"
                )

        elif action == 1:  # SELL 50%
            shares_to_sell = current_shares * 0.5
            if shares_to_sell > 0:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    qty=shares_to_sell,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                executed = True
                message = (
                    f"🔴 *LIVE PPO SELL 50%*\n"
                    f"Symbol: {settings.SYMBOL}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Shares sold: {shares_to_sell:.6f}"
                )

    if not executed:
        message = (
            f"⚪ *LIVE PPO HOLD / NO ACTION*\n"
            f"Symbol: {settings.SYMBOL}\n"
            f"Price: ${current_price:.2f}\n"
            f"Model action: {action_text}"
        )

    print(fnline(), message.replace("*", ""))
    send_telegram_alert(message)

if __name__ == "__main__":
    run_live_trader()
