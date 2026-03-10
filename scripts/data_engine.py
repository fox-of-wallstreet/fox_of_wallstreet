import os
import time
import yfinance as yf
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from tqdm import tqdm
from dotenv import load_dotenv

# 🎛️ Connect to the Control Room
from config import settings


# =========================
# 1. ENV / MODEL SETUP
# =========================
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment.")

TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Safer explicit label mapping
LABEL_TO_IDX = {str(v).lower(): int(k) for k, v in FINBERT_MODEL.config.id2label.items()}
if "positive" not in LABEL_TO_IDX or "negative" not in LABEL_TO_IDX:
    raise ValueError(f"Unexpected FinBERT label mapping: {FINBERT_MODEL.config.id2label}")

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
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")


def safe_download(symbol, period, interval, retries=3, sleep_sec=5):
    """Retry wrapper around yfinance downloads to handle throttling / intermittent failure."""
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)

            if df is not None and not df.empty:
                return df

            print(f"   -> Empty download for {symbol}, retry {attempt + 1}/{retries}")

        except Exception as e:
            print(f"   -> Download error for {symbol}, retry {attempt + 1}/{retries}: {e}")

        time.sleep(sleep_sec)

    print(f"   -> FAILED to download {symbol} after {retries} retries")
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
    print(f"📥 Fetching master market dataset for {symbol} ({settings.TIMEFRAME})...")

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
        print(f"   -> Pulling {ticker}...")
        df = safe_download(ticker, period=period, interval=settings.TIMEFRAME)

        if df.empty:
            print(f"   -> WARNING: No data returned for {ticker}")
            if ticker == symbol:
                raise ValueError(f"{symbol} download failed, cannot build dataset.")
            continue

        # Flatten yfinance MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()
        df.rename(columns={"Datetime": "Date", "index": "Date"}, inplace=True)

        if "Date" not in df.columns:
            raise ValueError(f"No Date column found after reset_index() for {ticker}.")

        # Normalize timestamps (timeframe-aware)
        df["Date"] = normalize_timestamp(df["Date"])

        if ticker == symbol:
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for {ticker}: {missing}")

            master_df = df[required_cols].copy()
        else:
            temp_df = df[["Date", "Close"]].rename(columns={"Close": col_name})
            temp_df = temp_df.drop_duplicates(subset=["Date"])
            master_df = pd.merge(master_df, temp_df, on="Date", how="left")

        time.sleep(2)

    if master_df is None or master_df.empty:
        raise ValueError("No market data could be assembled.")

    # Forward-fill macro series and drop remaining missing rows
    master_df = master_df.sort_values("Date").reset_index(drop=True)
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)

    if master_df.empty:
        raise ValueError("Market dataframe became empty after ffill/dropna cleanup.")

    print(f"✅ Market data merged: {len(master_df)} rows.")
    return master_df


# =========================
# 4. NEWS + SENTIMENT
# =========================
def get_paginated_news_sentiment(symbol, start_date, end_date):
    """Fetch all Alpaca news between start_date and end_date and score with FinBERT."""
    print(f"📰 Fetching and scoring paginated news for {symbol}...")
    client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_news_dfs = []

    current_end = end_date.isoformat()
    final_start = start_date.isoformat()

    print("Collecting headlines from Alpaca (backward time-stepping)...")

    while True:
        request_params = NewsRequest(
            symbols=symbol,
            start=final_start,
            end=current_end,
            limit=500
        )

        try:
            response = client.get_news(request_params)
            batch_df = response.df.reset_index()

            if batch_df.empty:
                break

            all_news_dfs.append(batch_df)

            current_total = sum(len(d) for d in all_news_dfs)
            print(f"   -> Progress: {current_total} articles fetched...", end="\r")

            oldest_time = batch_df["created_at"].min()

            if oldest_time <= start_date:
                print(f"\n✅ Reached start date: {start_date}")
                break

            current_end = (oldest_time - pd.Timedelta(seconds=1)).isoformat()

        except Exception as e:
            print(f"\nAPI error during time-stepping: {e}")
            break

    if not all_news_dfs:
        print("⚠️ No news returned from Alpaca.")
        return pd.DataFrame(columns=["Date", "Sentiment_EMA", "News_Intensity"])

    news_df = pd.concat(all_news_dfs, ignore_index=True)
    news_df = news_df.drop_duplicates(subset=["id"])
    print(f"✅ Total unique articles collected: {len(news_df)}")

    sentiment_scores = []
    for headline in tqdm(news_df["headline"], desc="🧠 FinBERT scoring"):
        sentiment_scores.append(score_headline_finbert(headline))

    news_df["Raw_Sentiment"] = sentiment_scores
    news_df["Date"] = normalize_timestamp(news_df["created_at"])

    hourly_sentiment = news_df.groupby("Date").agg(
        Sentiment_EMA=("Raw_Sentiment", "mean"),
        News_Intensity=("headline", "count")
    ).reset_index()

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
    print("🔄 Merging market and news datasets...")
    hybrid_df = pd.merge(price_df, sentiment_df, on="Date", how="left")

    # 4. Fill missing news-derived values
    hybrid_df["Sentiment_EMA"] = hybrid_df["Sentiment_EMA"].fillna(0.0)
    hybrid_df["News_Intensity"] = hybrid_df["News_Intensity"].fillna(0)

    # 5. Save
    hybrid_df.to_csv(output_file, index=False)
    print(f"🎉 Dataset built and saved to: {output_file}")
    print(f"📊 Final rows: {len(hybrid_df)}")

    return hybrid_df


if __name__ == "__main__":
    build_and_save_dataset()
