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

from config.settings import SYMBOL, MODEL_PATH
from core.data_engine import get_hybrid_dataset
from core.processor import add_technical_indicators, prepare_features

# --- CONFIGURATION ---
BUDGET = 10000.00  # The AI's fenced maximum budget
BUY_THRESHOLD = 0.1
SELL_THRESHOLD = -0.1

def send_telegram_alert(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram credentials missing in .env. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")

def run_live_trader():
    print(f"🟢 STARTING LIVE TRADER FOR {SYMBOL} 🟢")

    # 1. Authenticate with Alpaca
    load_dotenv()
    trading_client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)

    # 2. Fetch the latest market data (Last 60 days to ensure SMA50 works)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=100)

    print("📥 Fetching latest market data and reading today's news...")
    df = get_hybrid_dataset(SYMBOL, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df_with_indicators = add_technical_indicators(df)

    # DEBUG: Check if we have data after indicators
    print(f"📊 Rows available for AI after indicators: {len(df_with_indicators)}")

    if len(df_with_indicators) < 5:
        print("❌ ERROR: Not enough data points to build a 5-day observation. Try increasing timedelta.")
        return

    # 3. Process and Scale features
    features = [
        'Close', 'Volume', 'Volatility', 'RSI', 'MACD',
        'BB_Pct', 'Dist_SMA_50', 'AI_Score', 'AI_Confidence',
        'Sin_Time', 'Cos_Time'
    ]
    scaled_df = prepare_features(df_with_indicators, features, is_training=False)

    # 4. Build the Observation (The last 5 days = 90 features)
    last_5_days = scaled_df.iloc[-5:].values
    obs = last_5_days.flatten().reshape(1, -1)

    # 5. Ask the AI for a decision
    print(f"🧠 Asking Sentinel V7 for a decision on {SYMBOL}...")
    load_path = MODEL_PATH if MODEL_PATH.endswith(".zip") else MODEL_PATH
    model = PPO.load(load_path)

    action, _ = model.predict(obs, deterministic=True)
    signal = float(action[0])
    current_price = df_with_indicators.iloc[-1]['Close']

    print(f"\n📊 Current Price: ${current_price:.2f}")
    print(f"🤖 AI Raw Signal: {signal:.3f}")

    # 6. Check current Alpaca position
    try:
        position = trading_client.get_open_position(SYMBOL)
        current_shares = float(position.qty)
    except:
        current_shares = 0.0

    print(f"💼 Current Position: {current_shares} shares")

    # 7. Execute the Strategy
    print("\n" + "="*30)
    if signal > BUY_THRESHOLD:
        target_dollar_amount = BUDGET * signal
        current_value = current_shares * current_price
        needed_dollars = target_dollar_amount - current_value

        if needed_dollars > 10.00:
            print(f"🟢 ACTION: BUYING ${needed_dollars:.2f} of {SYMBOL}")
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

            print("✅ Buy order submitted to exchange.")
        else:
            print("⚪ ACTION: AI wants to buy, but we already own the target amount. Holding.")

    elif signal < SELL_THRESHOLD:
        sell_pct = abs(signal)
        shares_to_sell = current_shares * sell_pct

        if shares_to_sell > 0.001:
            print(f"🔴 ACTION: SELLING {shares_to_sell:.4f} shares of {SYMBOL}")
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

            print("✅ Sell order submitted to exchange.")
        else:
            print("⚪ ACTION: AI wants to sell, but we don't own any shares. Ignoring.")

    else:
        print("⚪ ACTION: Signal in Deadzone (-0.1 to 0.1). No trade executed.")
    print("="*30)

if __name__ == "__main__":
    run_live_trader()
