import os
import pandas as pd
from unittest.mock import MagicMock, patch

from config import settings
from scripts.news_engine import _normalize_news_batch, download_news


def _mock_alpaca_batch():
    df = pd.DataFrame(
        {
            "headline": [" TSLA beats estimates ", "New TSLA factory update"],
            "summary": ["good quarter", "factory progress"],
            "author": ["Author A", "Author B"],
            "source": ["Reuters", "Bloomberg"],
            "url": ["http://a.com", "http://b.com"],
            "symbols": [["TSLA"], ["TSLA", "QQQ"]],
            "created_at": ["2025-01-01T15:00:00Z", "2025-01-01T16:00:00Z"],
        },
        index=pd.Index([101, 102], name="id"),
    )
    return df


def test_normalize_news_batch_cleans_and_shapes_data():
    batch_df = _mock_alpaca_batch().reset_index()
    cleaned = _normalize_news_batch(batch_df, "TSLA")

    assert list(cleaned.columns) == [
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
    assert len(cleaned) == 2
    assert cleaned["headline"].iloc[0] == "TSLA beats estimates"
    assert cleaned["symbols"].iloc[0] == "TSLA"


@patch("scripts.news_engine._get_alpaca_client")
def test_download_news_saves_csv(mock_get_client):
    first_response = MagicMock()
    first_response.df = _mock_alpaca_batch()

    second_response = MagicMock()
    second_response.df = pd.DataFrame()

    mock_client = MagicMock()
    mock_client.get_news.side_effect = [first_response, second_response]
    mock_get_client.return_value = mock_client

    path = download_news("TSLA", settings.TRAIN_START_DATE, settings.TEST_END_DATE)

    assert path == settings.RAW_NEWS_CSV
    assert path.endswith(".csv")
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert {"id", "headline", "created_at", "created_at_ny"}.issubset(df.columns)
    assert len(df) == 2


def test_normalize_news_batch_drops_empty_headlines():
    batch_df = pd.DataFrame(
        {
            "id": [1, 2],
            "headline": ["", "TSLA launches update"],
            "summary": ["", "something happened"],
            "author": ["", "Author"],
            "source": ["", "Reuters"],
            "url": ["", "http://example.com"],
            "symbols": [["TSLA"], ["TSLA"]],
            "created_at": ["2025-01-01T15:00:00Z", "2025-01-01T16:00:00Z"],
        }
    )

    cleaned = _normalize_news_batch(batch_df, "TSLA")
    assert len(cleaned) == 1
    assert cleaned["headline"].iloc[0] == "TSLA launches update"
