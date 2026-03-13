import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv

from config import settings


# ==========================================
# PRIVATE COMPUTE FUNCTIONS
# Each takes a full df, adds exactly one column, returns df.
# To add a new feature: write a function here, register it below.
# ==========================================

def _ensure_columns(df, required_cols, source_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns in {source_name}: {missing}")


def _compute_log_return(df):
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def _compute_volume_z_score(df):
    vol_mean = df["Volume"].rolling(window=settings.VOLATILITY_WINDOW).mean()
    vol_std  = df["Volume"].rolling(window=settings.VOLATILITY_WINDOW).std()
    df["Volume_Z_Score"] = (df["Volume"] - vol_mean) / (vol_std + 1e-8)
    return df


def _compute_rsi(df):
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=settings.RSI_WINDOW).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=settings.RSI_WINDOW).mean()
    rs    = gain / (loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def _compute_macd_hist(df):
    ema_fast    = df["Close"].ewm(span=settings.MACD_FAST,   adjust=False).mean()
    ema_slow    = df["Close"].ewm(span=settings.MACD_SLOW,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=settings.MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"] = macd_line - signal_line
    return df


def _compute_bb_pct(df):
    rolling_mean = df["Close"].rolling(window=settings.VOLATILITY_WINDOW).mean()
    rolling_std  = df["Close"].rolling(window=settings.VOLATILITY_WINDOW).std()
    upper = rolling_mean + (2 * rolling_std)
    lower = rolling_mean - (2 * rolling_std)
    df["BB_Pct"] = (df["Close"] - lower) / ((upper - lower) + 1e-8)
    return df


def _compute_atr_pct(df):
    high_low    = df["High"] - df["Low"]
    high_close  = (df["High"] - df["Close"].shift(1)).abs()
    low_close   = (df["Low"]  - df["Close"].shift(1)).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_Pct"] = (true_range.rolling(window=settings.RSI_WINDOW).mean() / (df["Close"] + 1e-8)) * 100
    return df


def _compute_qqq_ret(df):
    if "QQQ_Close" in df.columns:
        df["QQQ_Ret"] = np.log(df["QQQ_Close"] / df["QQQ_Close"].shift(1))
    else:
        df["QQQ_Ret"] = 0.0
    return df


def _compute_arkk_ret(df):
    if "ARKK_Close" in df.columns:
        df["ARKK_Ret"] = np.log(df["ARKK_Close"] / df["ARKK_Close"].shift(1))
    else:
        df["ARKK_Ret"] = 0.0
    return df


def _compute_rel_strength_qqq(df):
    if "QQQ_Close" in df.columns:
        df["Rel_Strength_QQQ"] = df["Close"] / (df["QQQ_Close"] + 1e-8)
    else:
        df["Rel_Strength_QQQ"] = 0.0
    return df


def _compute_vix_level(df):
    df["VIX_Level"] = df["VIX_Close"] if "VIX_Close" in df.columns else 0.0
    return df


def _compute_tnx_level(df):
    df["TNX_Level"] = df["TNX_Close"] if "TNX_Close" in df.columns else 0.0
    return df


def _compute_sentiment_ema(df):
    if "Sentiment_EMA" not in df.columns:
        df["Sentiment_EMA"] = 0.0
    return df


def _compute_news_intensity(df):
    if "News_Intensity" not in df.columns:
        df["News_Intensity"] = 0.0
    return df


def _compute_sin_time(df):
    if settings.TIMEFRAME == "1h":
        mins = df["Date"].dt.hour * 60 + df["Date"].dt.minute
        df["Sin_Time"] = np.sin(2 * np.pi * mins / 1440.0)
    else:
        df["Sin_Time"] = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7.0)
    return df


def _compute_cos_time(df):
    if settings.TIMEFRAME == "1h":
        mins = df["Date"].dt.hour * 60 + df["Date"].dt.minute
        df["Cos_Time"] = np.cos(2 * np.pi * mins / 1440.0)
    else:
        df["Cos_Time"] = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7.0)
    return df


def _compute_mins_to_close(df):
    if settings.TIMEFRAME == "1h":
        mins = df["Date"].dt.hour * 60 + df["Date"].dt.minute
        df["Mins_to_Close"] = (16 * 60 - mins).clip(lower=0)
    else:
        df["Mins_to_Close"] = 0.0
    return df


# ==========================================
# FEATURE REGISTRY
# The only place that connects a feature name to its computation.
# ==========================================
FEATURE_REGISTRY = {
    "Log_Return":        _compute_log_return,
    "Volume_Z_Score":    _compute_volume_z_score,
    "RSI":               _compute_rsi,
    "MACD_Hist":         _compute_macd_hist,
    "BB_Pct":            _compute_bb_pct,
    "ATR_Pct":           _compute_atr_pct,
    "QQQ_Ret":           _compute_qqq_ret,
    "ARKK_Ret":          _compute_arkk_ret,
    "Rel_Strength_QQQ":  _compute_rel_strength_qqq,
    "VIX_Level":         _compute_vix_level,
    "TNX_Level":         _compute_tnx_level,
    "Sentiment_EMA":     _compute_sentiment_ema,
    "News_Intensity":    _compute_news_intensity,
    "Sin_Time":          _compute_sin_time,
    "Cos_Time":          _compute_cos_time,
    "Mins_to_Close":     _compute_mins_to_close,
}


# ==========================================
# INGESTION HELPERS
# ==========================================

def load_raw_prices(csv_path=None):
    csv_path = csv_path or settings.RAW_PRICES_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Raw prices CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _ensure_columns(df, ["Date", "Open", "High", "Low", "Close", "Volume"], "raw prices")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def load_raw_news(csv_path=None):
    csv_path = csv_path or settings.RAW_NEWS_CSV
    if not os.path.exists(csv_path):
        return pd.DataFrame(
            columns=["id", "headline", "summary", "author", "source", "url", "symbols", "created_at", "created_at_ny"]
        )

    df = pd.read_csv(csv_path)

    if "created_at_ny" in df.columns:
        df["created_at_ny"] = pd.to_datetime(df["created_at_ny"], errors="coerce")
    elif "created_at" in df.columns:
        df["created_at_ny"] = (
            pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    else:
        df["created_at_ny"] = pd.NaT

    if "headline" not in df.columns:
        df["headline"] = ""
    return df


def load_raw_macro(csv_path=None):
    csv_path = csv_path or settings.RAW_MACRO_CSV
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Date"])

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"❌ Raw macro CSV missing 'Date' column: {csv_path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


# ==========================================
# SENTIMENT
# ==========================================

def _load_finbert():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", **kwargs)
    return tokenizer, model, torch


def score_headlines_finbert(headlines):
    tokenizer, model, torch = _load_finbert()
    scores = []
    for headline in headlines:
        text = str(headline).strip()
        if not text:
            scores.append(0.0)
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        scores.append(probs[0].item() - probs[1].item())
    return scores


def build_news_sentiment(news_df, timeframe=None, scorer=None):
    timeframe = timeframe or settings.TIMEFRAME

    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["Date", "Sentiment_EMA", "News_Intensity"])

    df = news_df.copy()
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df = df[df["headline"] != ""].copy()

    if df.empty:
        return pd.DataFrame(columns=["Date", "Sentiment_EMA", "News_Intensity"])

    if "Raw_Sentiment" not in df.columns:
        scorer = scorer or score_headlines_finbert
        df["Raw_Sentiment"] = scorer(df["headline"].tolist())

    df["Date"] = pd.to_datetime(df["created_at_ny"], errors="coerce")
    df = df.dropna(subset=["Date"])

    floor_alias = "h" if timeframe == "1h" else "D"
    df["Date"] = df["Date"].dt.floor(floor_alias)

    grouped = (
        df.groupby("Date", as_index=False)
        .agg(
            Raw_Sentiment=("Raw_Sentiment", "mean"),
            News_Intensity=("headline", "count"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    grouped["Sentiment_EMA"] = grouped["Raw_Sentiment"].ewm(
        span=settings.NEWS_EMA_SPAN, adjust=False
    ).mean()

    return grouped[["Date", "Sentiment_EMA", "News_Intensity"]]


def merge_prices_and_news(price_df, news_sentiment_df):
    price_df         = price_df.copy()
    news_sentiment_df = news_sentiment_df.copy()

    price_df["Date"]          = pd.to_datetime(price_df["Date"], errors="coerce")
    news_sentiment_df["Date"] = pd.to_datetime(news_sentiment_df["Date"], errors="coerce")

    price_df          = price_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    news_sentiment_df = news_sentiment_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(price_df, news_sentiment_df, on="Date", direction="backward")
    merged["Sentiment_EMA"]  = merged["Sentiment_EMA"].fillna(0.0)
    merged["News_Intensity"] = merged["News_Intensity"].fillna(0.0)
    return merged


def merge_prices_news_macro(price_df, news_sentiment_df, macro_df=None):
    merged = merge_prices_and_news(price_df, news_sentiment_df)
    macro_df = macro_df if macro_df is not None else pd.DataFrame(columns=["Date"])

    if macro_df.empty or "Date" not in macro_df.columns:
        return merged

    macro_df = macro_df.copy()
    macro_df["Date"] = pd.to_datetime(macro_df["Date"], errors="coerce")
    macro_df = macro_df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    merged = pd.merge_asof(merged.sort_values("Date"), macro_df, on="Date", direction="backward")

    macro_cols = list(settings.MACRO_SYMBOL_MAP.values())
    existing_macro_cols = [col for col in macro_cols if col in merged.columns]
    if existing_macro_cols:
        merged[existing_macro_cols] = merged[existing_macro_cols].ffill().bfill().fillna(0.0)

    return merged


# ==========================================
# FEATURE ENGINEERING (Registry-driven)
# ==========================================

def add_technical_indicators(df, features_list=None):
    features_list = features_list or settings.FEATURES_LIST

    _ensure_columns(df, ["Date", "Close", "High", "Low", "Volume"], "merged dataset")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Validate all requested features are registered BEFORE computing anything
    unregistered = [f for f in features_list if f not in FEATURE_REGISTRY]
    if unregistered:
        raise ValueError(
            f"❌ Features requested but not in FEATURE_REGISTRY: {unregistered}\n"
            f"   Add a _compute_xxx() function and register it."
        )

    for feature_name in features_list:
        df = FEATURE_REGISTRY[feature_name](df)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ==========================================
# SCALING
# ==========================================

def prepare_features(df, features_list=None, is_training=True):
    features_list = features_list or settings.FEATURES_LIST

    missing = [col for col in features_list if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing requested feature columns: {missing}")

    data_to_scale = df[features_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if is_training:
        scaler      = RobustScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        os.makedirs(os.path.dirname(settings.SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, settings.SCALER_PATH)
        print(f"✅ Scaler saved to {settings.SCALER_PATH}")
    else:
        if not os.path.exists(settings.SCALER_PATH):
            raise FileNotFoundError(f"❌ No scaler found at {settings.SCALER_PATH}. Train first!")
        scaler      = joblib.load(settings.SCALER_PATH)
        scaled_data = scaler.transform(data_to_scale)

    return pd.DataFrame(scaled_data, columns=features_list, index=df.index)


# ==========================================
# ORCHESTRATOR
# ==========================================

def build_training_dataset(news_scorer=None):
    price_df = load_raw_prices()
    news_df  = load_raw_news()
    macro_df = load_raw_macro()

    news_sentiment_df = build_news_sentiment(news_df, timeframe=settings.TIMEFRAME, scorer=news_scorer)
    os.makedirs(os.path.dirname(settings.NEWS_SENTIMENT_CSV), exist_ok=True)
    news_sentiment_df.to_csv(settings.NEWS_SENTIMENT_CSV, index=False)

    merged_df = merge_prices_news_macro(price_df, news_sentiment_df, macro_df)
    os.makedirs(os.path.dirname(settings.MERGED_DATA_CSV), exist_ok=True)
    merged_df.to_csv(settings.MERGED_DATA_CSV, index=False)

    feature_df = add_technical_indicators(merged_df)

    train_start = pd.to_datetime(settings.TRAIN_START_DATE)
    train_end   = pd.to_datetime(settings.TRAIN_END_DATE)
    train_df    = feature_df[
        (feature_df["Date"] >= train_start) & (feature_df["Date"] <= train_end)
    ].copy()

    if train_df.empty:
        raise ValueError("❌ Training dataset is empty after feature engineering and date filtering.")

    os.makedirs(os.path.dirname(settings.TRAIN_FEATURES_CSV), exist_ok=True)
    train_df.to_csv(settings.TRAIN_FEATURES_CSV, index=False)

    print(f"✅ News sentiment  → {settings.NEWS_SENTIMENT_CSV}")
    print(f"✅ Merged dataset  → {settings.MERGED_DATA_CSV}")
    print(f"✅ Train features  → {settings.TRAIN_FEATURES_CSV}")
    return train_df
