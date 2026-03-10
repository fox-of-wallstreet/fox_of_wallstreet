'''
Missing module docstring.
'''

import os
import pandas as pd
import yfinance as yf

# 🎛️ Connect to the Control Room
from config import settings

def download_data(symbol: str, timeframe: str) -> str:
    '''
    Missing function or method docstring.
    '''
    print(f"📥 Downloading {timeframe} data for {symbol}...")

    # yfinance limits hourly data to the last 730 days. Daily can be "max".
    period = "730d" if timeframe == "1h" else "max"

    data = yf.download(symbol, period=period, interval=timeframe)

    if data.empty:
        print(f"❌ Error: No data found for {symbol}. Check symbol or internet.")
        return None

    # yfinance sometimes returns MultiIndex columns, we flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
    data['Date'] = data.index

    # Save to dynamic path
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{symbol.lower()}_{timeframe}_hybrid.csv"
    data.to_csv(file_path, index=False)

    print(f"✅ Raw data saved to {file_path}")
    print(f"📊 Total Rows: {len(data)}")
    return file_path

if __name__ == "__main__":
    download_data(symbol=settings.SYMBOL, timeframe=settings.TIMEFRAME)

