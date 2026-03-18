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
from pathlib import Path
import sys
import time
from datetime import datetime, timezone, timedelta
import argparse

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
sys.path.append('.')

from core.tools import fnline
from config import settings
from core.processor import build_news_sentiment, load_raw_news, merge_prices_news_macro, add_technical_indicators

def setup_artifact_symlinks():
    """
    Synchronizes the 'artifacts' directory with directories found in 'preloaded'.
    Expects both to be at the same level in the project root.
    """
    source_dir = Path("preloaded")
    target_dir = Path("artifacts")

    # 1. Ensure directories exist
    if not source_dir.exists():
        print(fnline(), f"[!] Warning: Source directory {source_dir} not found.")
        return

    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        print(fnline(), f"[*] Created target directory: {target_dir}")

    # 2. Iterate through 'preloaded'
    for item in source_dir.iterdir():
        if item.is_dir():
            link_name = target_dir / item.name
            
            # Use relative pathing for the symlink (more portable)
            # This points from 'artifacts/dir' back to '../preloaded/dir'
            relative_source = os.path.join("..", "preloaded", item.name)

            if not link_name.exists():
                try:
                    os.symlink(relative_source, link_name)
                    print(fnline(), f"[✓] Linked: artifacts/{item.name} -> {relative_source}")
                except OSError as e:
                    print(fnline(), f"[X] Failed to link {item.name}: {e}")
            else:
                print(fnline(), f"[-] {item.name} already exists in artifacts, skipping.")

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
        print(fnline(), "⚠️ Telegram credentials missing; skipping alert.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        print(fnline(), f"⚠️ Failed to send Telegram alert: {exc}")


def _send_telegram_confirmation_request(message: str, state: "_BotState") -> bool:
    """Send an inline-keyboard confirmation to Telegram and block until the owner
    taps ✅ Confirm or ❌ Reject, or the timeout elapses.

    While waiting, /mode and other commands are also processed via state.
    """
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    timeout_seconds = int(os.getenv("CONFIRMATION_TIMEOUT_SECONDS", "300"))

    if not token or not chat_id:
        print(fnline(), "⚠️ Telegram credentials missing; order will NOT execute in secure mode.")
        return False

    base_url = f"https://api.telegram.org/bot{token}"

    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Confirm", "callback_data": "confirm"},
            {"text": "❌ Reject",  "callback_data": "reject"},
        ]]
    }
    try:
        resp = requests.post(
            f"{base_url}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown", "reply_markup": keyboard},
            timeout=10,
        )
        if not resp.ok:
            print(fnline(), f"⚠️ Telegram confirmation send failed: {resp.text}")
            return False
    except Exception as exc:
        print(fnline(), f"⚠️ Telegram confirmation send error: {exc}")
        return False

    print(fnline(), f"⏳ Awaiting Telegram confirmation (timeout: {timeout_seconds}s)...")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        remaining = int(deadline - time.time())
        poll_timeout = min(20, remaining)
        if poll_timeout <= 0:
            break
        try:
            poll = requests.get(
                f"{base_url}/getUpdates",
                params={"offset": state.update_offset, "timeout": poll_timeout},
                timeout=poll_timeout + 5,
            )
            for update in poll.json().get("result", []):
                state.update_offset = update["update_id"] + 1
                result = _process_telegram_update(update, state, base_url, chat_id)
                if result == "confirm":
                    print(fnline(), "✅ Owner confirmed the order.")
                    return True
                if result == "reject":
                    print(fnline(), "❌ Owner rejected the order.")
                    return False
        except Exception as exc:
            print(fnline(), f"⚠️ Telegram polling error: {exc}")
            time.sleep(2)

    print(fnline(), "⏰ Confirmation timed out — order will NOT execute.")
    _tg_send(base_url, chat_id, "⏰ Confirmation timed out. Order *NOT* executed.")
    return False


def _resolve_trained_artifact_paths():
    """Resolve model/scaler/metadata.

    Priority:
    1. ARTIFACT_RUN env var — pin a specific run folder by name.
    2. settings.MODEL_PATH / SCALER_PATH if the files already exist.
    3. Auto-resolve: latest compatible run matching current settings prefix.
    """
    artifact_run = os.getenv("ARTIFACT_RUN", "").strip()
    if artifact_run:
        run_dir = os.path.join(settings.ARTIFACTS_BASE_DIR, artifact_run)
        model_zip = os.path.join(run_dir, "model.zip")
        scaler_pkl = os.path.join(run_dir, "scaler.pkl")
        if not os.path.exists(model_zip):
            raise FileNotFoundError(f"❌ ARTIFACT_RUN='{artifact_run}' — model.zip not found at {model_zip}")
        if not os.path.exists(scaler_pkl):
            raise FileNotFoundError(f"❌ ARTIFACT_RUN='{artifact_run}' — scaler.pkl not found at {scaler_pkl}")
        print(fnline(), f"📌 Using pinned artifact run: {run_dir}")
        return (
            os.path.join(run_dir, "model"),
            scaler_pkl,
            os.path.join(run_dir, "metadata.json"),
        )

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
    print(fnline(), f"ℹ️ Using latest compatible artifact run: {latest_run_dir}")
    return (
        os.path.join(latest_run_dir, "model"),
        os.path.join(latest_run_dir, "scaler.pkl"),
        os.path.join(latest_run_dir, "metadata.json"),
    )


def _validate_live_compatibility(metadata_path: str) -> None:
    if not os.path.exists(metadata_path):
        print(fnline(), f"⚠️ Metadata not found at {metadata_path}; skipping compatibility validation.")
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

    print(fnline(), "✅ Live compatibility check passed against training metadata.")

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


def _get_current_position_features(trading_client: TradingClient, latest_price: float):
    """Return portfolio features matching TradingEnv.NUM_PORTFOLIO_FEATURES (5):
    [cash_ratio, position_size, inventory_fraction, unrealized_pnl, last_action_norm]

    Cash is capped to settings.LIVE_TRADING_BUDGET so the agent only allocates
    from its designated envelope, regardless of total Alpaca account size.
    """
    account = trading_client.get_account()
    raw_cash = float(account.cash)
    cash = min(raw_cash, settings.LIVE_TRADING_BUDGET)

    try:
        position = trading_client.get_open_position(settings.SYMBOL)
        current_shares = float(position.qty)
        entry_price = float(position.avg_entry_price)
        in_position = True
    except Exception:
        current_shares = 0.0
        entry_price = 0.0
        in_position = False

    position_value = current_shares * latest_price
    portfolio_value = cash + position_value

    cash_ratio = cash / (settings.INITIAL_BALANCE + 1e-8)
    position_size = position_value / (settings.INITIAL_BALANCE + 1e-8)
    inventory_fraction = position_value / (portfolio_value + 1e-8)
    unrealized_pnl = (
        (latest_price - entry_price) / (entry_price + 1e-8)
        if in_position else 0.0
    )
    max_action = 4 if settings.ACTION_SPACE_TYPE == "discrete_5" else 2
    last_action_norm = 0.0 / max_action  # no prior action known in live context

    portfolio_features = np.array(
        [cash_ratio, position_size, inventory_fraction, unrealized_pnl, last_action_norm],
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


# ---------------------------------------------------------------------------
# Bot state — persists current trading mode across restarts
# ---------------------------------------------------------------------------

class _BotState:
    """Mutable state shared across bot helpers."""

    _STATE_FILE = os.path.join(settings.ARTIFACTS_BASE_DIR, ".trader_state.json")
    VALID_MODES = ("autopilot", "secure", "simulate")

    def __init__(self):
        self.mode: str = os.getenv("TRADER_MODE", "simulate").strip().lower()
        self.update_offset: int = 0
        self.stop_requested: bool = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._STATE_FILE):
            try:
                with open(self._STATE_FILE) as f:
                    data = json.load(f)
                saved_mode = data.get("mode", self.mode)
                if saved_mode in self.VALID_MODES:
                    self.mode = saved_mode
            except Exception:
                pass

    def save(self) -> None:
        try:
            with open(self._STATE_FILE, "w") as f:
                json.dump({"mode": self.mode}, f)
        except Exception as exc:
            print(fnline(), f"⚠️ Could not save trader state: {exc}")


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def _tg_send(base_url: str, chat_id: str, text: str, reply_markup=None) -> None:
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        requests.post(f"{base_url}/sendMessage", json=payload, timeout=10)
    except Exception as exc:
        print(fnline(), f"⚠️ Telegram send error: {exc}")


def _tg_send_status(state: _BotState, base_url: str, chat_id: str, trading_client=None) -> None:
    lines = [
        "📊 *BOT STATUS*",
        f"Mode: *{state.mode.upper()}*",
        f"Symbol: `{settings.SYMBOL}` | `{settings.TIMEFRAME}`",
        f"Budget: ${settings.LIVE_TRADING_BUDGET:,.0f}",
    ]
    if trading_client is not None:
        try:
            account = trading_client.get_account()
            lines.append(f"Account cash: ${float(account.cash):,.2f}")
            try:
                pos = trading_client.get_open_position(settings.SYMBOL)
                lines.append(f"Position: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
                if pos.unrealized_plpc is not None:
                    lines.append(f"Unrealized P&L: {float(pos.unrealized_plpc)*100:.2f}%")
            except Exception:
                lines.append("Position: none")
        except Exception:
            lines.append("_(could not fetch account info)_")
    _tg_send(base_url, chat_id, "\n".join(lines))


def _process_telegram_update(
    update: dict,
    state: _BotState,
    base_url: str,
    chat_id: str,
    trading_client=None,
) -> str | None:
    """Process one Telegram update.

    Returns 'confirm', 'reject', 'stop', or None.
    Commands (/mode, /status, /stop, /help) are handled as side-effects.
    """
    cb = update.get("callback_query")
    if cb and cb.get("data") in ("confirm", "reject"):
        try:
            requests.post(
                f"{base_url}/answerCallbackQuery",
                json={"callback_query_id": cb["id"], "text": "Received."},
                timeout=5,
            )
        except Exception:
            pass
        return cb["data"]

    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    if str(msg.get("chat", {}).get("id", "")) != str(chat_id):
        return None

    text = (msg.get("text") or "").strip()

    if text.startswith("/mode"):
        parts = text.split()
        new_mode = parts[1].lower() if len(parts) >= 2 else ""
        if new_mode in _BotState.VALID_MODES:
            state.mode = new_mode
            state.save()
            _tg_send(base_url, chat_id, f"✅ Mode switched to *{new_mode.upper()}*")
        else:
            _tg_send(base_url, chat_id, "Usage: `/mode autopilot|secure|simulate`")

    elif text == "/status":
        _tg_send_status(state, base_url, chat_id, trading_client)

    elif text == "/stop":
        state.stop_requested = True
        _tg_send(base_url, chat_id, "🛑 Bot stopping after the current cycle...")
        return "stop"

    elif text in ("/help", "/start"):
        _tg_send(
            base_url, chat_id,
            "*Available commands:*\n"
            "`/mode autopilot` — execute orders automatically\n"
            "`/mode secure` — wait for your confirmation before each trade\n"
            "`/mode simulate` — observe signals only, no orders submitted\n"
            "`/status` — current mode, position & account info\n"
            "`/stop` — gracefully stop the bot",
        )

    return None


def _poll_telegram_commands_once(state: _BotState, trading_client=None) -> None:
    """Do one 10-second long-poll for Telegram commands between agent cycles."""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        time.sleep(10)
        return
    base_url = f"https://api.telegram.org/bot{token}"
    try:
        resp = requests.get(
            f"{base_url}/getUpdates",
            params={"offset": state.update_offset, "timeout": 10},
            timeout=15,
        )
        for update in resp.json().get("result", []):
            state.update_offset = update["update_id"] + 1
            _process_telegram_update(update, state, base_url, chat_id, trading_client)
    except Exception as exc:
        print(fnline(), f"⚠️ Telegram poll error: {exc}")
        time.sleep(5)


def _next_candle_time(timeframe: str) -> datetime:
    """Return the UTC datetime of the next candle close + a small propagation buffer."""
    now = datetime.now(timezone.utc)
    if timeframe == "1h":
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=2)
    elif timeframe == "1d":
        candidate = now.replace(hour=21, minute=2, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    else:
        raise ValueError(f"No candle schedule defined for TIMEFRAME='{timeframe}'")


def _submit_action(trading_client: TradingClient, action: int, current_cash: float, current_shares: float, execute: bool = False) -> str:
    min_notional = 10.0

    execute_orders = execute

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


def _run_one_agent_cycle(
    trading_client: TradingClient,
    model: PPO,
    scaler,
    state: _BotState,
) -> None:
    """Build features, predict, confirm if needed, execute, notify."""
    feature_df = _build_live_feature_dataframe()
    missing = [c for c in settings.FEATURES_LIST if c not in feature_df.columns]
    if missing:
        raise ValueError(f"❌ Missing live feature columns: {missing}")

    market_scaled = scaler.transform(feature_df[settings.FEATURES_LIST])
    scaled_df = pd.DataFrame(market_scaled, columns=settings.FEATURES_LIST, index=feature_df.index)

    latest_price = float(feature_df.iloc[-1]["Close"])
    current_shares, current_cash, entry_price, portfolio_features = _get_current_position_features(trading_client, latest_price)
    obs = _build_live_observation(scaled_df, portfolio_features)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
    action_text = _action_to_text(action)

    print(fnline(), f"📊 Price: ${latest_price:.2f} | Action: {action} -> {action_text}")
    print(fnline(), f"💵 Budget cash: ${current_cash:.2f} | Shares: {current_shares:.6f}")

    action_is_trade = action_text != "HOLD"

    if state.mode == "secure" and action_is_trade:
        confirm_msg = (
            f"*⚠️ TRADE CONFIRMATION REQUIRED*\n"
            f"Time: {_now_utc_iso()}\n"
            f"Symbol: {settings.SYMBOL} | {settings.TIMEFRAME}\n"
            f"Action: *{action_text}*\n"
            f"Price: ${latest_price:.2f}\n"
            f"Budget cash: ${current_cash:.2f}\n"
            f"Shares held: {current_shares:.6f}\n"
            f"Entry price: ${entry_price:.2f}\n\n"
            f"Approve this order?"
        )
        execute = _send_telegram_confirmation_request(confirm_msg, state)
    else:
        execute = state.mode == "autopilot"

    execution_result = _submit_action(trading_client, action, current_cash, current_shares, execute=execute)
    print(fnline(), f"✅ Execution result: {execution_result}")

    mode_label = {"autopilot": "AUTOPILOT", "secure": "SECURE"}.get(state.mode, "SIMULATE")
    _send_telegram_alert(
        f"*{mode_label} PPO SIGNAL*\n"
        f"Time: {_now_utc_iso()}\n"
        f"Symbol: {settings.SYMBOL} | {settings.TIMEFRAME}\n"
        f"Action: {action_text}\n"
        f"Price: ${latest_price:.2f}\n"
        f"Budget cash: ${current_cash:.2f}\n"
        f"Shares: {current_shares:.6f}\n"
        f"Entry: ${entry_price:.2f}\n"
        f"Result: {execution_result}"
    )


def run_live_trader() -> None:
    """One-shot run: build features, predict one action, optionally execute, notify.
    Suitable for cron-based scheduling (e.g. every 1h).
    """
    load_dotenv()
    state = _BotState()
    print(fnline(), f"🟢 STARTING LIVE TRADER | {settings.SYMBOL} ({settings.TIMEFRAME})")
    print(fnline(), f"🔧 Trader mode: {state.mode.upper()}")
    print(fnline(), f"💰 Trading budget: ${settings.LIVE_TRADING_BUDGET:,.0f}")

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("❌ Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")

    paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
    trading_client = TradingClient(alpaca_key, alpaca_secret, paper=paper_flag)

    # Drain stale Telegram updates so stale callbacks don't trigger things.
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        base_url = f"https://api.telegram.org/bot{token}"
        try:
            drain = requests.get(f"{base_url}/getUpdates", params={"timeout": 0}, timeout=10)
            prev = drain.json().get("result", [])
            state.update_offset = (prev[-1]["update_id"] + 1) if prev else 0
        except Exception:
            pass

    model_base_path, scaler_path, metadata_path = _resolve_trained_artifact_paths()
    _validate_live_compatibility(metadata_path)

    scaler = joblib.load(scaler_path)
    print(fnline(), f"🧠 Loading model from {model_base_path}.zip")
    model = PPO.load(model_base_path)

    _run_one_agent_cycle(trading_client, model, scaler, state)


def run_live_trader_bot() -> None:
    """Persistent bot: schedules agent cycles at each candle close and handles
    Telegram commands (/mode, /status, /stop, /help) between cycles.

    Model and scaler are loaded once at startup for efficiency.
    Mode is persisted to artifacts/.trader_state.json between restarts.

    Env vars:
      ARTIFACT_RUN=<folder>         — pin a specific artifact run
      TRADER_MODE=autopilot|secure|simulate — initial mode (overridden by /mode)
      CONFIRMATION_TIMEOUT_SECONDS  — seconds to wait for secure-mode confirmation
    """
    load_dotenv()
    state = _BotState()

    print(fnline(), f"🤖 STARTING LIVE TRADER BOT | {settings.SYMBOL} ({settings.TIMEFRAME})")
    print(fnline(), f"🔧 Initial mode: {state.mode.upper()}")
    print(fnline(), f"💰 Trading budget: ${settings.LIVE_TRADING_BUDGET:,.0f}")

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    if not alpaca_key or not alpaca_secret:
        raise ValueError("❌ Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")
    paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
    trading_client = TradingClient(alpaca_key, alpaca_secret, paper=paper_flag)

    model_base_path, scaler_path, metadata_path = _resolve_trained_artifact_paths()
    _validate_live_compatibility(metadata_path)
    scaler = joblib.load(scaler_path)
    print(fnline(), f"🧠 Loading model from {model_base_path}.zip")
    model = PPO.load(model_base_path)

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        base_url = f"https://api.telegram.org/bot{token}"
        try:
            drain = requests.get(f"{base_url}/getUpdates", params={"timeout": 0}, timeout=10)
            prev = drain.json().get("result", [])
            state.update_offset = (prev[-1]["update_id"] + 1) if prev else 0
        except Exception as exc:
            print(fnline(), f"⚠️ Telegram drain failed: {exc}")
        _tg_send(
            base_url, chat_id,
            f"🤖 *LIVE TRADER BOT STARTED*\n"
            f"Symbol: `{settings.SYMBOL}` | `{settings.TIMEFRAME}`\n"
            f"Mode: *{state.mode.upper()}*\n"
            f"Budget: ${settings.LIVE_TRADING_BUDGET:,.0f}\n"
            f"Send /help for available commands.",
        )
    else:
        print(fnline(), "⚠️ TELEGRAM_TOKEN/TELEGRAM_CHAT_ID not set — Telegram disabled.")

    while not state.stop_requested:
        next_run = _next_candle_time(settings.TIMEFRAME)
        print(fnline(), f"⏰ Next agent cycle at: {next_run.isoformat()}")

        while datetime.now(timezone.utc) < next_run and not state.stop_requested:
            _poll_telegram_commands_once(state, trading_client)

        if state.stop_requested:
            break

        print(fnline(), f"🔔 Running agent cycle | {_now_utc_iso()} | mode={state.mode.upper()}")
        try:
            _run_one_agent_cycle(trading_client, model, scaler, state)
        except Exception as exc:
            errmsg = f"❌ Agent cycle error: {exc}"
            print(errmsg)
            _send_telegram_alert(errmsg)

    _send_telegram_alert(f"🛑 Live trader bot stopped at {_now_utc_iso()}.")
    print(fnline(), "🛑 Bot stopped.")


if __name__ == "__main__":
    setup_artifact_symlinks()
    parser = argparse.ArgumentParser(description="Fox of Wallstreet — live PPO trader")
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Run as a persistent bot with Telegram control and automatic candle scheduling.",
    )
    args = parser.parse_args()
    if args.bot:
        run_live_trader_bot()
    else:
        run_live_trader()
