import yfinance as yf
import pandas as pd
import os

# 🎛️ Connect to the Control Room
from config import settings

def download_data():
    print(f"📥 Downloading {settings.TIMEFRAME} data for {settings.SYMBOL}...")

    # yfinance limits hourly data to the last 730 days. Daily can be "max".
    period = "730d" if settings.TIMEFRAME == "1h" else "max"

    data = yf.download(settings.SYMBOL, period=period, interval=settings.TIMEFRAME)

    if data.empty:
        print(f"❌ Error: No data found for {settings.SYMBOL}. Check symbol or internet.")
        return

    # yfinance sometimes returns MultiIndex columns, we flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
    data['Date'] = data.index

    # Save to dynamic path
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    data.to_csv(file_path, index=False)

    print(f"✅ Raw data saved to {file_path}")
    print(f"📊 Total Rows: {len(data)}")
    return file_path

if __name__ == "__main__":
    download_data()
