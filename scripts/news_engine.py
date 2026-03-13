"""
Fetch raw Alpaca news for the configured symbol, apply light cleaning,
and save a raw CSV checkpoint for downstream processing.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from config import settings


RAW_NEWS_COLUMNS = [
    "id",
    "headline",
    "summary",
    "author",
    "source",
    "url",
    "symbols",
    "created_at",
    "created_at_ny",
]


def _get_alpaca_client() -> NewsClient:
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "❌ Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    return NewsClient(api_key, secret_key)


def _to_utc_timestamp(value, is_end=False) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")

    if is_end and ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return ts.tz_convert("UTC")


def _normalize_news_batch(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame(columns=RAW_NEWS_COLUMNS)

    df = batch_df.copy()

    if "id" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "id"})

    if "id" not in df.columns:
        raise ValueError("❌ Alpaca news batch does not contain an 'id' column.")

    for col in ["headline", "summary", "author", "source", "url", "symbols", "created_at"]:
        if col not in df.columns:
            df[col] = None

    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    df["author"] = df["author"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["id", "created_at"])
    df = df[df["headline"] != ""]

    def normalize_symbols(value):
        if isinstance(value, (list, tuple, set)):
            return ",".join(map(str, value))
        if pd.isna(value):
            return ""
        return str(value)

    df["symbols"] = df["symbols"].apply(normalize_symbols)

    if "symbols" in df.columns:
        df = df[df["symbols"].str.contains(symbol, case=False, na=False) | (df["symbols"] == "")]

    df["created_at_ny"] = (
        df["created_at"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    df = df.drop_duplicates(subset=["id"]).sort_values("created_at").reset_index(drop=True)
    return df[RAW_NEWS_COLUMNS]


def download_news(symbol: str, start_date, end_date, limit: int = 500) -> str:
    client = _get_alpaca_client()

    start_ts = _to_utc_timestamp(start_date, is_end=False)
    end_ts = _to_utc_timestamp(end_date, is_end=True)

    print(f"📰 Downloading raw Alpaca news for {symbol}...")
    print(f"📅 Window: {start_ts.isoformat()} -> {end_ts.isoformat()}")

    all_batches = []
    current_end = end_ts

    while True:
        request = NewsRequest(
            symbols=symbol,
            start=start_ts.isoformat(),
            end=current_end.isoformat(),
            limit=limit,
        )

        response = client.get_news(request)
        batch_df = response.df.reset_index()

        cleaned_batch = _normalize_news_batch(batch_df, symbol)
        if cleaned_batch.empty:
            break

        all_batches.append(cleaned_batch)

        oldest_timestamp = pd.to_datetime(cleaned_batch["created_at"], utc=True).min()
        if oldest_timestamp <= start_ts:
            break

        current_end = oldest_timestamp - pd.Timedelta(seconds=1)

    if all_batches:
        news_df = pd.concat(all_batches, ignore_index=True)
        news_df["created_at"] = pd.to_datetime(news_df["created_at"], utc=True, errors="coerce")
        news_df = news_df.dropna(subset=["created_at"])
        news_df = news_df[
            (news_df["created_at"] >= start_ts) &
            (news_df["created_at"] <= end_ts)
        ].copy()
        news_df = news_df.drop_duplicates(subset=["id"]).sort_values("created_at").reset_index(drop=True)
    else:
        news_df = pd.DataFrame(columns=RAW_NEWS_COLUMNS)


    os.makedirs(os.path.dirname(settings.RAW_NEWS_CSV), exist_ok=True)
    news_df.to_csv(settings.RAW_NEWS_CSV, index=False)

    print(f"✅ Raw news saved to {settings.RAW_NEWS_CSV}")
    print(f"🗞️ Articles saved: {len(news_df)}")
    return settings.RAW_NEWS_CSV


if __name__ == "__main__":
    download_news(
        symbol=settings.SYMBOL,
        start_date=settings.TRAIN_START_DATE,
        end_date=settings.TEST_END_DATE,
    )
