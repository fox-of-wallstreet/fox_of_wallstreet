"""
Downloads, validates, and saves raw historical market data from yfinance.
Outputs a raw CSV checkpoint for downstream processing.
"""

import os
import pandas as pd
import yfinance as yf

from config import settings
from core.tools import fnline

<<<<<<< HEAD
# =========================
# 1. ENV / MODEL SETUP
# =========================
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError(f"{fnline()} Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")

TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Safer explicit label mapping
LABEL_TO_IDX = {str(v).lower(): int(k) for k, v in FINBERT_MODEL.config.id2label.items()}
if "positive" not in LABEL_TO_IDX or "negative" not in LABEL_TO_IDX:
    raise ValueError(f"{fnline()} Unexpected FinBERT label mapping: {FINBERT_MODEL.config.id2label}")

POS_IDX = LABEL_TO_IDX["positive"]
NEG_IDX = LABEL_TO_IDX["negative"]


# =========================
# 2. HELPERS
# =========================
def get_download_period():
    """Map timeframe to a sensible yfinance period."""
    return "730d" if settings.TIMEFRAME == "1h" else "max"

def normalize_timestamp(series):
    """Normalize timestamps to New York timezone and correct resolution."""
    dt = pd.to_datetime(series)

    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")

    dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)

    if settings.TIMEFRAME == "1h":
        return dt.dt.floor("h")
    elif settings.TIMEFRAME == "1d":
        return dt.dt.floor("d")
    else:
        raise ValueError(f"{fnline()} Unsupported TIMEFRAME: {settings.TIMEFRAME}")


def safe_download(symbol, period, interval, retries=3, sleep_sec=5):
    """Retry wrapper around yfinance downloads to handle throttling / intermittent failure."""
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)

            if df is not None and not df.empty:
                return df

            print(fnline(), f"   -> Empty download for {symbol}, retry {attempt + 1}/{retries}")

        except Exception as e:
            print(fnline(), f"   -> Download error for {symbol}, retry {attempt + 1}/{retries}: {e}")

        time.sleep(sleep_sec)

    print(fnline(), f"   -> FAILED to download {symbol} after {retries} retries")
    return pd.DataFrame()


def score_headline_finbert(headline):
    """Return positive-minus-negative sentiment score."""
    inputs = TOKENIZER(headline, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = FINBERT_MODEL(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return probs[POS_IDX].item() - probs[NEG_IDX].item()


# =========================
# 3. MARKET DATA
# =========================
def get_macro_market_data(symbol):
    """Fetch traded asset + macro market series with strict date alignment."""
    print(fnline(), f"📥 Fetching master market dataset for {symbol} ({settings.TIMEFRAME})...")

    tickers = {
        symbol: "Close",
        "^VIX": "VIX_Close",
        "QQQ": "QQQ_Close",
        "ARKK": "ARKK_Close",
        "^TNX": "TNX_Close",
    }

    master_df = None
    period = get_download_period()

    for ticker, col_name in tickers.items():
        print(fnline(), f"   -> Pulling {ticker}...")
        df = safe_download(ticker, period=period, interval=settings.TIMEFRAME)

        if df.empty:
            print(fnline(), f"   -> WARNING: No data returned for {ticker}")
            if ticker == symbol:
                raise ValueError(f"{fnline()} {symbol} download failed, cannot build dataset.")
            continue

        # Flatten yfinance MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()
        df.rename(columns={"Datetime": "Date", "index": "Date"}, inplace=True)

        if "Date" not in df.columns:
            raise ValueError(f"{fnline()} No Date column found after reset_index() for {ticker}.")

        # Normalize timestamps (timeframe-aware)
        df["Date"] = normalize_timestamp(df["Date"])

        if ticker == symbol:
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{fnline()} Missing required columns for {ticker}: {missing}")

            master_df = df[required_cols].copy()
        else:
            temp_df = df[["Date", "Close"]].rename(columns={"Close": col_name})
            temp_df = temp_df.drop_duplicates(subset=["Date"])
            master_df = pd.merge(master_df, temp_df, on="Date", how="left")

        time.sleep(2)

    if master_df is None or master_df.empty:
        raise ValueError(f"{fnline()} No market data could be assembled.")

    # Forward-fill macro series and drop remaining missing rows
    master_df = master_df.sort_values("Date").reset_index(drop=True)
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)

    if master_df.empty:
        raise ValueError(f"{fnline()} Market dataframe became empty after ffill/dropna cleanup.")

    print(fnline(), f"✅ Market data merged: {len(master_df)} rows.")
    return master_df


# =========================
# 4. NEWS + SENTIMENT
# =========================
def get_paginated_news_sentiment(symbol, start_date, end_date):
    """Fetch all Alpaca news between start_date and end_date and score with FinBERT."""
    print(fnline(), f"📰 Fetching and scoring paginated news for {symbol}...")
    client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_news_dfs = []

    current_end = end_date.isoformat()
    final_start = start_date.isoformat()

    print(fnline(), "Collecting headlines from Alpaca (backward time-stepping)...")

    while True:
        request_params = NewsRequest(
            symbols=symbol,
            start=final_start,
            end=current_end,
            limit=500
=======
REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_data(symbol: str, timeframe: str) -> str:
    """
    Download OHLCV data, validate it, filter it to the experiment window,
    and save it as the raw prices CSV checkpoint.
    """
    if timeframe not in settings.VALID_TIMEFRAMES:
        raise ValueError(
            f"❌ Invalid timeframe '{timeframe}'. Expected one of {settings.VALID_TIMEFRAMES}."
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        )

    print(f"📥 Downloading {timeframe} data for {symbol}...")

    period = "730d" if timeframe == "1h" else "max"
    data = yf.download(symbol, period=period, interval=timeframe, progress=False)

    if data.empty:
        raise ValueError(f"❌ No data found for {symbol}. Check symbol or internet connection.")

<<<<<<< HEAD
            current_total = sum(len(d) for d in all_news_dfs)
            print(fnline(), f"   -> Progress: {current_total} articles fetched...", end="\r")
=======
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns from yfinance: {missing}")

<<<<<<< HEAD
            if oldest_time <= start_date:
                print(fnline(), f"\n✅ Reached start date: {start_date}")
                break
=======
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

    data = data.reset_index()
    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    elif "Date" not in data.columns:
        raise ValueError("❌ Could not find a datetime column after download.")

<<<<<<< HEAD
        except Exception as e:
            print(fnline(), f"\nAPI error during time-stepping: {e}")
            break

    if not all_news_dfs:
        print(fnline(), "⚠️ No news returned from Alpaca.")
        return pd.DataFrame(columns=["Date", "Sentiment_EMA", "News_Intensity"])

    news_df = pd.concat(all_news_dfs, ignore_index=True)
    news_df = news_df.drop_duplicates(subset=["id"])
    print(fnline(), f"✅ Total unique articles collected: {len(news_df)}")
=======
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    start_date = pd.to_datetime(settings.TRAIN_START_DATE)
    end_date = pd.to_datetime(settings.TEST_END_DATE)

    data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)].copy()
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485

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

<<<<<<< HEAD
    return hourly_sentiment


# =========================
# 5. BUILD FINAL DATASET
# =========================
def build_and_save_dataset(symbol=None, output_file=None):
    if symbol is None:
        symbol = settings.SYMBOL

    if output_file is None:
        output_file = f"data/{symbol.lower()}_{settings.TIMEFRAME}_hybrid.csv"

    os.makedirs("data", exist_ok=True)

    # 1. Market data
    price_df = get_macro_market_data(symbol)

    # 2. News data
    start_date = pd.Timestamp(price_df["Date"].min()).tz_localize("America/New_York")
    end_date = pd.Timestamp(price_df["Date"].max()).tz_localize("America/New_York")
    sentiment_df = get_paginated_news_sentiment(symbol, start_date, end_date)

    # 3. Merge
    print(fnline(), "🔄 Merging market and news datasets...")
    hybrid_df = pd.merge(price_df, sentiment_df, on="Date", how="left")

    # 4. Fill missing news-derived values
    hybrid_df["Sentiment_EMA"] = hybrid_df["Sentiment_EMA"].fillna(0.0)
    hybrid_df["News_Intensity"] = hybrid_df["News_Intensity"].fillna(0)

    # 5. Save
    hybrid_df.to_csv(output_file, index=False)
    print(fnline(), f"🎉 Dataset built and saved to: {output_file}")
    print(fnline(), f"📊 Final rows: {len(hybrid_df)}")

    return hybrid_df
=======
    print(f"✅ Raw prices saved to {settings.RAW_PRICES_CSV}")
    print(f"📊 Total Rows: {len(data)}")
    return settings.RAW_PRICES_CSV
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485


if __name__ == "__main__":
    download_data(symbol=settings.SYMBOL, timeframe=settings.TIMEFRAME)
