"""Download raw macro series used by macro features and save a single checkpoint CSV."""

import os
import pandas as pd
import yfinance as yf

from config import settings


def _download_single_symbol(symbol: str, out_col: str, timeframe: str) -> pd.DataFrame:
    period = "730d" if timeframe == "1h" else "max"
    data = yf.download(symbol, period=period, interval=timeframe, progress=False)

    if data.empty:
        raise ValueError(f"❌ No macro data returned for {symbol}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if "Close" not in data.columns:
        raise ValueError(f"❌ Missing Close column for {symbol}.")

    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    df = data.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    elif "Date" not in df.columns:
        raise ValueError(f"❌ Could not locate datetime column for {symbol}.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df[["Date", "Close"]].rename(columns={"Close": out_col})


def download_macro_data(timeframe: str) -> str:
    if timeframe not in settings.VALID_TIMEFRAMES:
        raise ValueError(
            f"❌ Invalid timeframe '{timeframe}'. Expected one of {settings.VALID_TIMEFRAMES}."
        )

    print(f"📥 Downloading macro data for timeframe {timeframe}...")

    merged = None
    for symbol, out_col in settings.MACRO_SYMBOL_MAP.items():
        print(f"   • {symbol} -> {out_col}")
        symbol_df = _download_single_symbol(symbol, out_col, timeframe)
        merged = symbol_df if merged is None else pd.merge(merged, symbol_df, on="Date", how="outer")

    start_date = pd.to_datetime(settings.TRAIN_START_DATE)
    end_date = pd.to_datetime(settings.TEST_END_DATE)

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    merged = merged[(merged["Date"] >= start_date) & (merged["Date"] <= end_date)].copy()

    macro_cols = list(settings.MACRO_SYMBOL_MAP.values())
    merged[macro_cols] = merged[macro_cols].ffill().bfill()

    os.makedirs(os.path.dirname(settings.RAW_MACRO_CSV), exist_ok=True)
    merged.to_csv(settings.RAW_MACRO_CSV, index=False)

    print(f"✅ Raw macro data saved to {settings.RAW_MACRO_CSV}")
    print(f"📊 Rows: {len(merged)}")
    return settings.RAW_MACRO_CSV


if __name__ == "__main__":
    download_macro_data(settings.TIMEFRAME)
