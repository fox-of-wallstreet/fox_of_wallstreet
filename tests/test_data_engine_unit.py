import pandas as pd
import pytest
from unittest.mock import patch

from config import settings
from scripts.data_engine import download_data


def _mock_price_df():
    dates = pd.date_range(start=settings.TRAIN_START_DATE, periods=150, freq="h")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 150,
            "High": [101.0] * 150,
            "Low": [99.0] * 150,
            "Close": [100.5] * 150,
            "Volume": [1000] * 150,
        },
        index=dates,
    )
    df.index.name = "Datetime"
    return df


@patch("scripts.data_engine.yf.download")
def test_download_data_returns_expected_path(mock_download):
    mock_download.return_value = _mock_price_df()

    path = download_data(settings.SYMBOL, settings.TIMEFRAME)

    assert path == settings.RAW_PRICES_CSV
    assert path.endswith(".csv")


@patch("scripts.data_engine.yf.download")
def test_download_data_filters_and_saves_required_columns(mock_download):
    mock_download.return_value = _mock_price_df()

    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    df = pd.read_csv(path)

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    assert required_cols.issubset(df.columns)


@patch("scripts.data_engine.yf.download")
def test_download_data_raises_on_empty_dataframe(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No data found"):
        download_data(settings.SYMBOL, settings.TIMEFRAME)


@patch("scripts.data_engine.yf.download")
def test_download_data_raises_on_missing_columns(mock_download):
    dates = pd.date_range(start=settings.TRAIN_START_DATE, periods=150, freq="h")
    bad_df = pd.DataFrame(
        {
            "Open": [100.0] * 150,
            "High": [101.0] * 150,
            "Low": [99.0] * 150,
            "Close": [100.5] * 150,
        },
        index=dates,
    )
    bad_df.index.name = "Datetime"
    mock_download.return_value = bad_df

    with pytest.raises(ValueError, match="Missing required columns"):
        download_data(settings.SYMBOL, settings.TIMEFRAME)


def test_invalid_timeframe_raises_error():
    with pytest.raises(ValueError, match="Invalid timeframe"):
        download_data(settings.SYMBOL, "5m")
