"""
Downloads, validates, and saves raw historical market data from yfinance.
Outputs a raw CSV checkpoint for downstream processing.
"""

import os
import pandas as pd
import yfinance as yf

from config import settings


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_data(symbol: str, timeframe: str) -> str:
    """
    Download OHLCV data, validate it, filter it to the experiment window,
    and save it as the raw prices CSV checkpoint.
    """
    if timeframe not in settings.VALID_TIMEFRAMES:
        raise ValueError(
            f"❌ Invalid timeframe '{timeframe}'. Expected one of {settings.VALID_TIMEFRAMES}."
        )

    print(f"📥 Downloading {timeframe} data for {symbol}...")

    period = "730d" if timeframe == "1h" else "max"
    data = yf.download(symbol, period=period, interval=timeframe, progress=False)

    if data.empty:
        raise ValueError(f"❌ No data found for {symbol}. Check symbol or internet connection.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns from yfinance: {missing}")

    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    data = data.reset_index()
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    elif "Date" not in data.columns:
        raise ValueError("❌ Could not find a datetime column after download.")

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    start_date = pd.to_datetime(settings.TRAIN_START_DATE)
    end_date = pd.to_datetime(settings.TEST_END_DATE)

    data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)].copy()

    if len(data) < settings.MIN_TRAIN_ROWS:
        raise ValueError(
            f"❌ Insufficient data: only {len(data)} rows available between "
            f"{start_date.date()} and {end_date.date()}."
        )

    null_counts = data[REQUIRED_PRICE_COLUMNS + ["Date"]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"❌ Null values found in raw price data:\n{null_counts[null_counts > 0]}")

    os.makedirs(os.path.dirname(settings.RAW_PRICES_CSV), exist_ok=True)
    data.to_csv(settings.RAW_PRICES_CSV, index=False)

    print(f"✅ Raw prices saved to {settings.RAW_PRICES_CSV}")
    print(f"📊 Total Rows: {len(data)}")
    return settings.RAW_PRICES_CSV


if __name__ == "__main__":
    download_data(symbol=settings.SYMBOL, timeframe=settings.TIMEFRAME)
