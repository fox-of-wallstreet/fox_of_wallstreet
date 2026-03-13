import os
import pandas as pd
import pytest

from config import settings
from scripts.data_engine import download_data


def test_download_data_returns_csv_path():
    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    assert path.endswith(".csv")
    assert path == settings.RAW_PRICES_CSV
    assert os.path.exists(path)


def test_download_data_has_required_columns():
    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    df = pd.read_csv(path)

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    assert required_cols.issubset(df.columns)


def test_download_data_is_sorted_and_unique():
    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    assert df["Date"].is_monotonic_increasing
    assert not df["Date"].duplicated().any()


def test_download_data_respects_experiment_window():
    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    start_date = pd.to_datetime(settings.TRAIN_START_DATE)
    end_date = pd.to_datetime(settings.TEST_END_DATE)

    assert df["Date"].min() >= start_date
    assert df["Date"].max() <= end_date


def test_download_data_has_minimum_rows():
    path = download_data(settings.SYMBOL, settings.TIMEFRAME)
    df = pd.read_csv(path)
    assert len(df) >= settings.MIN_TRAIN_ROWS


def test_invalid_timeframe_raises_error():
    with pytest.raises(ValueError, match="Invalid timeframe"):
        download_data(settings.SYMBOL, "5m")
