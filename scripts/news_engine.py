"""
Download raw news from Alpaca and store it as a local CSV checkpoint.

Purpose
-------
This script fetches raw news articles for the configured symbol and date window,
normalizes them into a stable flat schema, and saves them to settings.RAW_NEWS_CSV.

Notes
-----
- Keeps raw news separate from sentiment processing.
- Sentiment scoring (FinBERT) is handled later in processor.py.
- Designed to be robust against Alpaca SDK response-shape differences.
"""

import os
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
except ImportError as exc:
    raise ImportError(
        "alpaca-py is required for news_engine.py.\n"
        "Install it with: pip install alpaca-py"
    ) from exc


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
    """
    Create Alpaca NewsClient from environment variables.
    """
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "❌ Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    return NewsClient(api_key=api_key, secret_key=secret_key)


def _to_utc_timestamp(value, is_end: bool) -> pd.Timestamp:
    """
    Convert a date-like input into a UTC timestamp aligned to New York market date boundaries.

    Parameters
    ----------
    value : str or datetime-like
        Date input, e.g. '2024-01-01'
    is_end : bool
        If False -> start of day in New York
        If True  -> end of day in New York
    """
    if isinstance(value, str):
        dt = pd.Timestamp(value)
    else:
        dt = pd.Timestamp(value)

    if dt.tzinfo is not None:
        return dt.tz_convert("UTC")

    ny_tz = ZoneInfo("America/New_York")
    if is_end:
        local_dt = datetime.combine(dt.date(), time(23, 59, 59), tzinfo=ny_tz)
    else:
        local_dt = datetime.combine(dt.date(), time(0, 0, 0), tzinfo=ny_tz)

    return pd.Timestamp(local_dt).tz_convert("UTC")


def _extract_dict_from_article(article):
    """
    Convert one Alpaca article object into a plain dict as safely as possible.
    """
    if article is None:
        return None

    if isinstance(article, dict):
        data = article
    elif hasattr(article, "model_dump"):
        # pydantic-style models
        data = article.model_dump()
    elif hasattr(article, "__dict__"):
        data = dict(article.__dict__)
    else:
        return None

    # Unwrap common nested payload layouts if present
    if "raw_data" in data and isinstance(data["raw_data"], dict):
        data = data["raw_data"]
    elif "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    return data


def _article_to_record(article) -> dict | None:
    """
    Convert one Alpaca article object/dict into a flat record matching RAW_NEWS_COLUMNS.
    Returns None if the record is unusable.
    """
    data = _extract_dict_from_article(article)
    if not isinstance(data, dict):
        return None

    record = {
        "id": data.get("id"),
        "headline": data.get("headline"),
        "summary": data.get("summary"),
        "author": data.get("author"),
        "source": data.get("source"),
        "url": data.get("url"),
        "symbols": data.get("symbols"),
        "created_at": data.get("created_at"),
    }

    # Must have at least id + created_at to be useful downstream
    if record["id"] is None or record["created_at"] is None:
        return None

    return record


def _response_to_records(response) -> list[dict]:
    """
    Safely extract a flat list of article dicts from an Alpaca news response.

    Handles:
    - dict responses with a 'news' key
    - SDK responses with .data['news']
    - SDK responses with .news
    """
    if response is None:
        return []

    items = None

    if isinstance(response, dict):
        items = response.get("news", [])
    elif hasattr(response, "data") and isinstance(response.data, dict):
        items = response.data.get("news", [])
    elif hasattr(response, "news"):
        items = response.news

    if items is None:
        return []

    records = []
    for article in items:
        rec = _article_to_record(article)
        if rec is not None:
            records.append(rec)

    return records


def _normalize_symbols(value) -> str:
    """
    Convert symbols field into a comma-separated string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(map(str, value))
    return str(value)


def _normalize_news_batch(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalize one raw article batch into the stable schema used by the project.
    """
    if batch_df is None or batch_df.empty:
        return pd.DataFrame(columns=RAW_NEWS_COLUMNS)

    df = batch_df.copy()

    # Defensive fallback if an alternate id field slips through
    if "id" not in df.columns:
        for candidate in ["index", "_id", "article_id"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "id"})
                break

    if "id" not in df.columns:
        print(f"⚠️ Skipping batch without usable id column. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=RAW_NEWS_COLUMNS)

    for col in ["headline", "summary", "author", "source", "url", "symbols", "created_at"]:
        if col not in df.columns:
            df[col] = None

    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    df["author"] = df["author"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["symbols"] = df["symbols"].apply(_normalize_symbols)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # Keep only valid, non-empty records
    df = df.dropna(subset=["id", "created_at"])
    df = df[df["headline"] != ""]

    # Keep articles tagged with the symbol when symbols metadata exists.
    # If symbols is empty, keep the row rather than discarding potentially useful news.
    df = df[
        df["symbols"].str.contains(symbol, case=False, na=False) |
        (df["symbols"] == "")
    ].copy()

    df["created_at_ny"] = (
        df["created_at"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    df = df.drop_duplicates(subset=["id"]).sort_values("created_at").reset_index(drop=True)

    return df[RAW_NEWS_COLUMNS]


def download_news(symbol: str, start_date, end_date, limit: int = 50) -> str:
    """
    Download raw news for one symbol and save it to settings.RAW_NEWS_CSV.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'NVDA'
    start_date : str or datetime-like
        Inclusive start date.
    end_date : str or datetime-like
        Inclusive end date.
    limit : int
        Alpaca batch size per request. 50 is a safe default.
    """
    client = _get_alpaca_client()

    start_ts = _to_utc_timestamp(start_date, is_end=False)
    end_ts = _to_utc_timestamp(end_date, is_end=True)

    print(f"📰 Downloading raw Alpaca news for {symbol}...")
    print(f"📅 Window: {start_ts.isoformat()} -> {end_ts.isoformat()}")

    all_batches = []
    current_end = end_ts
    n_requests = 0

    while True:
        n_requests += 1

        request = NewsRequest(
            symbols=symbol,
            start=start_ts.isoformat(),
            end=current_end.isoformat(),
            limit=limit,
        )

        response = client.get_news(request)
        batch_records = _response_to_records(response)

        if not batch_records:
            break

        batch_df = pd.DataFrame(batch_records)
        cleaned_batch = _normalize_news_batch(batch_df, symbol)

        if cleaned_batch.empty:
            break

        all_batches.append(cleaned_batch)

        oldest_timestamp = pd.to_datetime(cleaned_batch["created_at"], utc=True).min()
        if pd.isna(oldest_timestamp) or oldest_timestamp <= start_ts:
            break

        # move window backwards by one second to paginate earlier news
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
    print(f"🔁 Requests made: {n_requests}")

    return settings.RAW_NEWS_CSV


if __name__ == "__main__":
    download_news(
        symbol=settings.SYMBOL,
        start_date=settings.TRAIN_START_DATE,
        end_date=settings.TEST_END_DATE,
    )
