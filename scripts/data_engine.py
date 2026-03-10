import time
import yfinance as yf
import pandas as pd
import os

# 🎛️ Connect to the Control Room
from config import settings

def safe_download(symbol, period, interval, retries=3, sleep_sec=5):
    """Retry wrapper for Yahoo Finance downloads."""
    for attempt in range(retries):
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)

            if data is not None and not data.empty:
                return data

            print(f"   -> Empty download for {symbol}, retry {attempt + 1}/{retries}")

        except Exception as e:
            print(f"   -> Download error for {symbol}, retry {attempt + 1}/{retries}: {e}")

        time.sleep(sleep_sec)

    return pd.DataFrame()


def download_data():
    print(f"📥 Downloading {settings.TIMEFRAME} data for {settings.SYMBOL}...")

    # yfinance limits hourly data to the last 730 days. Daily can be "max".
    period = "730d" if settings.TIMEFRAME == "1h" else "max"

    data = safe_download(settings.SYMBOL, period=period, interval=settings.TIMEFRAME)

    if data.empty:
        raise ValueError(
            f"❌ No data found for {settings.SYMBOL} at interval {settings.TIMEFRAME}. "
            "Check symbol, internet, or Yahoo throttling."
        )

    # yfinance sometimes returns MultiIndex columns, we flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
    data['Date'] = pd.to_datetime(data.index)

    # Save to dynamic path
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    data.to_csv(file_path, index=False)

    print(f"✅ Raw data saved to {file_path}")
    print(f"📊 Total Rows: {len(data)}")

    return file_path

if __name__ == "__main__":
    download_data()
