import os
import pandas as pd
import numpy as np

from config import settings
from core.processor import (
    build_news_sentiment,
    build_training_dataset,
    prepare_features,
)


def _make_price_df():
    dates = pd.date_range(start="2024-01-01", periods=200, freq="h")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(100, 120, 200),
            "High": np.linspace(101, 121, 200),
            "Low": np.linspace(99, 119, 200),
            "Close": np.linspace(100, 120, 200),
            "Volume": np.linspace(1000, 2000, 200),
        }
    )


def _make_news_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "headline": ["Good TSLA delivery news", "TSLA update", "TSLA faces challenge"],
            "created_at_ny": ["2024-01-01 10:15:00", "2024-01-01 10:45:00", "2024-01-01 11:10:00"],
            "Raw_Sentiment": [0.8, 0.4, -0.2],
        }
    )


def test_build_news_sentiment_hourly_aggregation():
    news_df = _make_news_df()
    out = build_news_sentiment(news_df, timeframe="1h")

    assert {"Date", "Sentiment_EMA", "News_Intensity"}.issubset(out.columns)
    assert len(out) == 2
    assert out["News_Intensity"].sum() == 3


def test_build_training_dataset_writes_checkpoint_csvs(tmp_path, monkeypatch):
    raw_prices = tmp_path / "prices.csv"
    raw_news = tmp_path / "news.csv"
    news_sentiment = tmp_path / "news_sentiment.csv"
    merged_csv = tmp_path / "merged.csv"
    train_features_csv = tmp_path / "train_features.csv"
    scaler_path = tmp_path / "scaler.pkl"

    _make_price_df().to_csv(raw_prices, index=False)
    _make_news_df().to_csv(raw_news, index=False)

    monkeypatch.setattr(settings, "RAW_PRICES_CSV", str(raw_prices))
    monkeypatch.setattr(settings, "RAW_NEWS_CSV", str(raw_news))
    monkeypatch.setattr(settings, "NEWS_SENTIMENT_CSV", str(news_sentiment))
    monkeypatch.setattr(settings, "MERGED_DATA_CSV", str(merged_csv))
    monkeypatch.setattr(settings, "TRAIN_FEATURES_CSV", str(train_features_csv))
    monkeypatch.setattr(settings, "SCALER_PATH", str(scaler_path))
    monkeypatch.setattr(settings, "TRAIN_START_DATE", "2024-01-01")
    monkeypatch.setattr(settings, "TRAIN_END_DATE", "2024-01-31")
    monkeypatch.setattr(settings, "TIMEFRAME", "1h")
    monkeypatch.setattr(
        settings,
        "FEATURES_LIST",
        [
            "Log_Return",
            "Volume_Z_Score",
            "RSI",
            "MACD_Hist",
            "BB_Pct",
            "ATR_Pct",
            "QQQ_Ret",
            "ARKK_Ret",
            "Rel_Strength_QQQ",
            "VIX_Level",
            "TNX_Level",
            "Sentiment_EMA",
            "News_Intensity",
            "Sin_Time",
            "Cos_Time",
            "Mins_to_Close",
        ],
    )

    train_df = build_training_dataset()

    assert os.path.exists(news_sentiment)
    assert os.path.exists(merged_csv)
    assert os.path.exists(train_features_csv)
    assert not train_df.empty
    assert set(settings.FEATURES_LIST).issubset(train_df.columns)


def test_prepare_features_saves_and_loads_scaler(tmp_path, monkeypatch):
    scaler_path = tmp_path / "scaler.pkl"
    monkeypatch.setattr(settings, "SCALER_PATH", str(scaler_path))

    df = pd.DataFrame(
        {
            "Log_Return": [0.1, 0.2, 0.3],
            "Volume_Z_Score": [1.0, 0.0, -1.0],
        }
    )
    features = ["Log_Return", "Volume_Z_Score"]

    scaled_train = prepare_features(df, features_list=features, is_training=True)
    scaled_test = prepare_features(df, features_list=features, is_training=False)

    assert os.path.exists(scaler_path)
    assert scaled_train.shape == (3, 2)
    assert scaled_test.shape == (3, 2)
