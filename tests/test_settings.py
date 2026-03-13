import os
import pandas as pd

from config import settings


def test_timeframe_is_valid():
    assert settings.TIMEFRAME in settings.VALID_TIMEFRAMES


def test_action_space_is_valid():
    assert settings.ACTION_SPACE_TYPE in settings.VALID_ACTION_SPACES


def test_reward_strategy_is_valid():
    assert settings.REWARD_STRATEGY in settings.VALID_REWARD_STRATEGIES


def test_date_order_is_valid():
    train_start = pd.to_datetime(settings.TRAIN_START_DATE)
    train_end = pd.to_datetime(settings.TRAIN_END_DATE)
    test_start = pd.to_datetime(settings.TEST_START_DATE)
    test_end = pd.to_datetime(settings.TEST_END_DATE)

    assert train_start < train_end
    assert test_start < test_end
    assert train_end < test_start


def test_features_list_is_not_empty():
    assert isinstance(settings.FEATURES_LIST, list)
    assert len(settings.FEATURES_LIST) > 0


def test_features_list_has_no_duplicates():
    assert len(settings.FEATURES_LIST) == len(set(settings.FEATURES_LIST))


def test_core_directories_exist():
    assert os.path.isdir(settings.DATA_DIR)
    assert os.path.isdir(settings.RAW_DATA_DIR)
    assert os.path.isdir(settings.INTERMEDIATE_DATA_DIR)
    assert os.path.isdir(settings.ARTIFACTS_BASE_DIR)


def test_output_paths_have_expected_suffixes():
    assert settings.RAW_PRICES_CSV.endswith(".csv")
    assert settings.RAW_NEWS_CSV.endswith(".csv")
    assert settings.NEWS_SENTIMENT_CSV.endswith(".csv")
    assert settings.MERGED_DATA_CSV.endswith(".csv")
    assert settings.TRAIN_FEATURES_CSV.endswith(".csv")
    assert settings.TEST_FEATURES_CSV.endswith(".csv")
    assert settings.SCALER_PATH.endswith(".pkl")
    assert settings.METADATA_PATH.endswith(".json")
    assert settings.BACKTEST_LEDGER_PATH.endswith(".csv")


def test_numeric_config_values_are_positive():
    assert settings.INITIAL_BALANCE > 0
    assert settings.TOTAL_TIMESTEPS > 0
    assert settings.LEARNING_RATE > 0
    assert settings.N_STACK > 0
    assert settings.MIN_TRAIN_ROWS > 0


def test_risk_values_are_in_valid_ranges():
    assert 0 < settings.CASH_RISK_FRACTION <= 1
    assert 0 < settings.BANKRUPTCY_THRESHOLD_PCT <= 1
    assert settings.BANKRUPTCY_PENALTY >= 0
    assert settings.INVALID_ACTION_PENALTY >= 0
    assert settings.SLIPPAGE_PCT >= 0
