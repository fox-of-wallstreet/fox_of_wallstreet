<<<<<<< HEAD
import os
import sys
import datetime
import pandas as pd
import numpy as np
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
from config.settings import SYMBOL, MODEL_PATH
from scripts.data_engine import build_and_save_dataset
from core.processor import add_technical_indicators, prepare_features

# --- CONFIGURATION ---
BUDGET = 10000.00  # The AI's fenced maximum budget
BUY_THRESHOLD = 0.1
SELL_THRESHOLD = -0.1

def fnline():
    '''
    For logging and tracing.
    Returns current filename and line number.
    E.g.: backtest.py(144))
    '''
    return os.path.basename(__file__) + '(' + str(sys._getframe(1).f_lineno) + '):'

def send_telegram_alert(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(fnline(), "⚠️ Telegram credentials missing in .env. Skipping alert.")
=======
"""Live trader aligned with the modular training/backtest architecture.

This script:
1) Resolves the latest compatible trained artifact (model + scaler + metadata),
2) Builds a live feature snapshot from recent market data + news + macro context,
3) Scales with the resolved training scaler,
4) Builds a stacked observation compatible with VecFrameStack training,
5) Predicts one action and optionally submits a paper/live Alpaca market order.
"""

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from stable_baselines3 import PPO

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import build_news_sentiment, load_raw_news, merge_prices_news_macro, add_technical_indicators


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_merge_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Normalize merge key datetime dtype to datetime64[ns] for merge_asof compatibility."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out.dropna(subset=[col]).sort_values(col).drop_duplicates(subset=[col]).reset_index(drop=True)
    out[col] = out[col].values.astype("datetime64[ns]")
    return out


def _send_telegram_alert(message: str) -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ Telegram credentials missing; skipping alert.")
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
<<<<<<< HEAD

    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(fnline(), f"❌ Failed to send Telegram alert: {e}")

def run_live_trader():
    print(fnline(), f"🟢 STARTING LIVE TRADER FOR {SYMBOL} 🟢")

    # 1. Authenticate with Alpaca
    load_dotenv()
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    # 2. Fetch the latest market data (Last 60 days to ensure SMA50 works)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=100)

    print(fnline(), "📥 Fetching latest market data and reading today's news...")
    df = build_and_save_dataset(SYMBOL)
    df_with_indicators = add_technical_indicators(df)

    # DEBUG: Check if we have data after indicators
    print(fnline(), f"📊 Rows available for AI after indicators: {len(df_with_indicators)}")

    if len(df_with_indicators) < 5:
        print(fnline(), "❌ ERROR: Not enough data points to build a 5-day observation. Try increasing timedelta.")
        return

    # 3. Process and Scale features
    features = [
        'Close', 'Volume', 'Volatility', 'RSI', 'MACD',
        'BB_Pct', 'Dist_SMA_50', 'AI_Score', 'AI_Confidence',
        'Sin_Time', 'Cos_Time'
    ]
    features = [
        'Close', 'RSI',
        'BB_Pct',
        'Sin_Time', 'Cos_Time'
    ]
    if settings.TIMEFRAME == "1h":
        features = features + ['Sin_Time', 'Cos_Time', 'Mins_to_Close']
    scaled_df = prepare_features(df_with_indicators, features, is_training=False)

    # 4. Build the Observation (The last 5 days = 90 features)
    last_5_days = scaled_df.iloc[-5:].values
    obs = last_5_days.flatten().reshape(1, -1)

    # 5. Ask the AI for a decision
    print(fnline(), f"🧠 Asking Sentinel V7 for a decision on {SYMBOL}...")
    load_path = MODEL_PATH if MODEL_PATH.endswith(".zip") else MODEL_PATH
    model = PPO.load(load_path)

    action, _ = model.predict(obs, deterministic=True)
    signal = float(action[0])
    current_price = df_with_indicators.iloc[-1]['Close']

    print(fnline(), f"\n📊 Current Price: ${current_price:.2f}")
    print(fnline(), f"🤖 AI Raw Signal: {signal:.3f}")

    # 6. Check current Alpaca position
    try:
        position = trading_client.get_open_position(SYMBOL)
        current_shares = float(position.qty)
    except:
        current_shares = 0.0

    print(fnline(), f"💼 Current Position: {current_shares} shares")

    # 7. Execute the Strategy
    print(fnline(), "\n" + "="*30)
    if signal > BUY_THRESHOLD:
        target_dollar_amount = BUDGET * signal
        current_value = current_shares * current_price
        needed_dollars = target_dollar_amount - current_value

        if needed_dollars > 10.00:
            print(fnline(), f"🟢 ACTION: BUYING ${needed_dollars:.2f} of {SYMBOL}")
            order = MarketOrderRequest(
                symbol=SYMBOL,
                notional=needed_dollars,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)

            # Send Telegram Notification
            msg = f"🟢 *SENTINEL AI: BUY ALERT*\nSymbol: {SYMBOL}\nAction: Bought ${needed_dollars:.2f}\nPrice: ${current_price:.2f}\nAI Signal: {signal:.2f}"
            send_telegram_alert(msg)

            print(fnline(), "✅ Buy order submitted to exchange.")
        else:
            print(fnline(), "⚪ ACTION: AI wants to buy, but we already own the target amount. Holding.")

    elif signal < SELL_THRESHOLD:
        sell_pct = abs(signal)
        shares_to_sell = current_shares * sell_pct

        if shares_to_sell > 0.001:
            print(fnline(), f"🔴 ACTION: SELLING {shares_to_sell:.4f} shares of {SYMBOL}")
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=shares_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)

            # Send Telegram Notification
            msg = f"🔴 *SENTINEL AI: SELL ALERT*\nSymbol: {SYMBOL}\nAction: Sold {shares_to_sell:.4f} shares\nPrice: ${current_price:.2f}\nAI Signal: {signal:.2f}"
            send_telegram_alert(msg)

            print(fnline(), "✅ Sell order submitted to exchange.")
        else:
            print(fnline(), "⚪ ACTION: AI wants to sell, but we don't own any shares. Ignoring.")

    else:
        print(fnline(), "⚪ ACTION: Signal in Deadzone (-0.1 to 0.1). No trade executed.")
    print(fnline(), "="*30)
=======
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        print(f"⚠️ Failed to send Telegram alert: {exc}")


def _resolve_trained_artifact_paths():
    """Resolve model/scaler/metadata similarly to backtest's compatibility pattern."""
    current_model_zip = f"{settings.MODEL_PATH}.zip"
    current_scaler = settings.SCALER_PATH
    current_metadata = settings.METADATA_PATH

    if os.path.exists(current_model_zip) and os.path.exists(current_scaler):
        return settings.MODEL_PATH, settings.SCALER_PATH, current_metadata

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
            "❌ No compatible trained artifacts found. Run scripts/train.py first."
        )

    latest_run_dir = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    print(f"ℹ️ Using latest compatible artifact run: {latest_run_dir}")
    return (
        os.path.join(latest_run_dir, "model"),
        os.path.join(latest_run_dir, "scaler.pkl"),
        os.path.join(latest_run_dir, "metadata.json"),
    )


def _validate_live_compatibility(metadata_path: str) -> None:
    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata not found at {metadata_path}; skipping compatibility validation.")
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
            "❌ Live compatibility check failed.",
            "Current settings differ from resolved training metadata:",
        ]
        for key, trained, current in mismatches:
            lines.append(f"  - {key}: trained={trained} | current={current}")
        lines.append("Align config/settings.py with the trained run, or retrain.")
        raise ValueError("\n".join(lines))

    print("✅ Live compatibility check passed against training metadata.")


def _download_recent_prices(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == "1h":
        period = "120d"
    elif timeframe == "1d":
        period = "5y"
    else:
        raise ValueError(f"Unsupported TIMEFRAME: {timeframe}")

    df = yf.download(symbol, period=period, interval=timeframe, progress=False)
    if df is None or df.empty:
        raise ValueError(f"❌ No recent market data for {symbol} ({timeframe}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    elif "Date" not in df.columns:
        raise ValueError("❌ No Date/Datetime column in market download.")

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required market columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    df["Date"] = df["Date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return df[required]


def _download_recent_macro(timeframe: str) -> pd.DataFrame:
    if not settings.USE_MACRO_FEATURES:
        return pd.DataFrame(columns=["Date"])

    period = "120d" if timeframe == "1h" else "5y"
    merged = None

    for symbol, out_col in settings.MACRO_SYMBOL_MAP.items():
        df = yf.download(symbol, period=period, interval=timeframe, progress=False)
        if df is None or df.empty:
            raise ValueError(f"❌ No recent macro data for {symbol}.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if "Close" not in df.columns:
            raise ValueError(f"❌ Missing Close in macro download for {symbol}.")

        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "Date" not in df.columns:
            raise ValueError(f"❌ No Date column in macro download for {symbol}.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        df["Date"] = df["Date"].dt.tz_convert("America/New_York").dt.tz_localize(None)

        one = df[["Date", "Close"]].rename(columns={"Close": out_col})
        merged = one if merged is None else pd.merge(merged, one, on="Date", how="outer")

    merged = merged.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    cols = list(settings.MACRO_SYMBOL_MAP.values())
    merged[cols] = merged[cols].ffill().bfill()
    return merged


def _build_live_feature_dataframe() -> pd.DataFrame:
    prices_df = _download_recent_prices(settings.SYMBOL, settings.TIMEFRAME)

    if settings.USE_NEWS_FEATURES:
        news_df = load_raw_news()
        sentiment_df = build_news_sentiment(news_df, timeframe=settings.TIMEFRAME)
    else:
        sentiment_df = pd.DataFrame(columns=["Date", "Sentiment_Mean", "News_Intensity"])

    macro_df = _download_recent_macro(settings.TIMEFRAME)

    # Ensure exact datetime dtype match for merge_asof across all live inputs.
    prices_df = _normalize_merge_datetime(prices_df, "Date")
    if not sentiment_df.empty:
        sentiment_df = _normalize_merge_datetime(sentiment_df, "Date")
    if not macro_df.empty:
        macro_df = _normalize_merge_datetime(macro_df, "Date")

    merged = merge_prices_news_macro(prices_df, sentiment_df, macro_df)
    features_df = add_technical_indicators(merged)
    if features_df.empty:
        raise ValueError("❌ Live features are empty after processing.")

    return features_df


def _get_current_position_features(trading_client: TradingClient):
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

    cash_ratio = cash / (settings.INITIAL_BALANCE + 1e-8)
    bars_in_trade_norm = 0.0

    portfolio_features = np.array(
        [in_position, unrealized_pnl_pct, cash_ratio, bars_in_trade_norm],
        dtype=np.float32,
    )
    return current_shares, cash, entry_price, portfolio_features


def _build_live_observation(scaled_market_features: pd.DataFrame, portfolio_features: np.ndarray) -> np.ndarray:
    n_stack = settings.N_STACK
    market = scaled_market_features.values
    if len(market) < n_stack:
        raise ValueError(
            f"❌ Not enough rows for stacked live observation. Need >= {n_stack}, got {len(market)}."
        )

    frames = []
    for row in market[-n_stack:]:
        frames.append(np.hstack([row.astype(np.float32), portfolio_features]))

    obs = np.hstack(frames).reshape(1, -1).astype(np.float32)
    return obs


def _action_to_text(action: int) -> str:
    if settings.ACTION_SPACE_TYPE == "discrete_3":
        mapping = {0: "SELL_ALL", 1: "BUY_ALL", 2: "HOLD"}
    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        mapping = {0: "SELL_100", 1: "SELL_50", 2: "HOLD", 3: "BUY_50", 4: "BUY_100"}
    else:
        raise ValueError(f"Unsupported ACTION_SPACE_TYPE: {settings.ACTION_SPACE_TYPE}")
    return mapping.get(int(action), "UNKNOWN")


def _should_execute_orders() -> bool:
    """Return True only when explicitly enabled via env flag."""
    return os.getenv("LIVE_TRADER_EXECUTE", "false").strip().lower() in {"1", "true", "yes", "y"}


def _submit_action(trading_client: TradingClient, action: int, current_cash: float, current_shares: float) -> str:
    min_notional = 10.0

    execute_orders = _should_execute_orders()

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        if action == 1:
            notional = current_cash * settings.CASH_RISK_FRACTION
            if notional > min_notional:
                if execute_orders:
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        notional=notional,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    trading_client.submit_order(order)
                    return f"BUY_ALL executed | notional=${notional:.2f}"
                return f"BUY_ALL simulated | notional=${notional:.2f}"
            return "BUY_ALL skipped (insufficient cash)"

        if action == 0:
            if current_shares > 0:
                if execute_orders:
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        qty=current_shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    trading_client.submit_order(order)
                    return f"SELL_ALL executed | qty={current_shares:.6f}"
                return f"SELL_ALL simulated | qty={current_shares:.6f}"
            return "SELL_ALL skipped (no shares)"

        return "HOLD"

    # discrete_5
    if action == 4:
        notional = current_cash * settings.CASH_RISK_FRACTION
        if notional > min_notional:
            if execute_orders:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                trading_client.submit_order(order)
                return f"BUY_100 executed | notional=${notional:.2f}"
            return f"BUY_100 simulated | notional=${notional:.2f}"
        return "BUY_100 skipped (insufficient cash)"

    if action == 3:
        notional = current_cash * 0.5 * settings.CASH_RISK_FRACTION
        if notional > min_notional:
            if execute_orders:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                trading_client.submit_order(order)
                return f"BUY_50 executed | notional=${notional:.2f}"
            return f"BUY_50 simulated | notional=${notional:.2f}"
        return "BUY_50 skipped (insufficient cash)"

    if action == 0:
        if current_shares > 0:
            if execute_orders:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    qty=current_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                trading_client.submit_order(order)
                return f"SELL_100 executed | qty={current_shares:.6f}"
            return f"SELL_100 simulated | qty={current_shares:.6f}"
        return "SELL_100 skipped (no shares)"

    if action == 1:
        qty = current_shares * 0.5
        if qty > 0:
            if execute_orders:
                order = MarketOrderRequest(
                    symbol=settings.SYMBOL,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                trading_client.submit_order(order)
                return f"SELL_50 executed | qty={qty:.6f}"
            return f"SELL_50 simulated | qty={qty:.6f}"
        return "SELL_50 skipped (no shares)"

    return "HOLD"


def run_live_trader() -> None:
    print(f"🟢 STARTING LIVE TRADER | {settings.SYMBOL} ({settings.TIMEFRAME})")
    print(f"🧪 Order mode: {'EXECUTE' if _should_execute_orders() else 'SIMULATE'}")
    load_dotenv()

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("❌ Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")

    paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
    trading_client = TradingClient(alpaca_key, alpaca_secret, paper=paper_flag)

    model_base_path, scaler_path, metadata_path = _resolve_trained_artifact_paths()
    _validate_live_compatibility(metadata_path)

    feature_df = _build_live_feature_dataframe()
    missing = [c for c in settings.FEATURES_LIST if c not in feature_df.columns]
    if missing:
        raise ValueError(f"❌ Missing live feature columns: {missing}")

    scaler = joblib.load(scaler_path)
    market_scaled = scaler.transform(feature_df[settings.FEATURES_LIST])
    scaled_df = pd.DataFrame(market_scaled, columns=settings.FEATURES_LIST, index=feature_df.index)

    current_shares, current_cash, entry_price, portfolio_features = _get_current_position_features(trading_client)
    latest_price = float(feature_df.iloc[-1]["Close"])

    obs = _build_live_observation(scaled_df, portfolio_features)

    print(f"🧠 Loading model from {model_base_path}.zip")
    model = PPO.load(model_base_path)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
    action_text = _action_to_text(action)

    print(f"📊 Latest price: ${latest_price:.2f}")
    print(f"💼 Current shares: {current_shares:.6f}")
    print(f"💵 Current cash: ${current_cash:.2f}")
    print(f"🤖 PPO action: {action} -> {action_text}")

    execution_result = _submit_action(trading_client, action, current_cash, current_shares)
    print(f"✅ Execution result: {execution_result}")

    alert = (
        f"*LIVE PPO SIGNAL*\n"
        f"Time: {_now_utc_iso()}\n"
        f"Symbol: {settings.SYMBOL}\n"
        f"Timeframe: {settings.TIMEFRAME}\n"
        f"Action: {action_text}\n"
        f"Price: ${latest_price:.2f}\n"
        f"Cash: ${current_cash:.2f}\n"
        f"Shares: {current_shares:.6f}\n"
        f"Entry: ${entry_price:.2f}\n"
        f"Result: {execution_result}"
    )
    _send_telegram_alert(alert)

>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

if __name__ == "__main__":
    run_live_trader()
