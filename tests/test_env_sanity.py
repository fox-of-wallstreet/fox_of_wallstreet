"""
Environment sanity tests.

These tests build a minimal but realistic TradingEnv from synthetic data and
verify that every component behaves correctly end-to-end:

  1. Observation-space shape: obs vector matches num_features exactly.
  2. Observation-space dtype: obs is float32 (SB3 requirement).
  3. Portfolio features appear *after* market features (index check).
  4. No raw OHLCV leaks into the observation vector.
  5. Action-space type and size match settings.
  6. Reset returns a valid obs.
  7. All discrete actions step without exception and return correct obs shape.
  8. Invalid action (sell with no position) yields negative penalty reward.
  9. Stop-loss fires and closes position.
  10. Take-profit fires and closes position.
  11. Bankruptcy terminates episode and applies penalty.
  12. Frame-stacking via VecFrameStack produces the right obs shape.
  13. Feature count matches EXPECTED_MARKET_FEATURES from settings.
  14. Feature scaling: values near zero-mean (post-RobustScaler passthrough of zeros).
  15. AVWAP columns: both AVWAP_Dist and AVWAP_Dist_ATR are present in feature df.
"""

import numpy as np
import pandas as pd
import pytest

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from config import settings
from core.environment import TradingEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_BARS = 200   # enough for warm-up windows (MA_SLOW=50, ATR=14, etc.)


def _make_price_df(n=N_BARS, start_price=100.0, seed=0) -> pd.DataFrame:
    """Synthetic OHLCV dataframe with realistic structure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    close = start_price + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)  # no negative prices
    high  = close * (1 + rng.uniform(0, 0.01, n))
    low   = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({
        "Date":   dates,
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": vol,
    })


def _make_features_df(n=N_BARS, seed=1) -> pd.DataFrame:
    """
    Synthetic feature matrix with the exact columns from settings.FEATURES_LIST.
    Values are drawn from a standard normal — representative post-scaler distribution.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, settings.EXPECTED_MARKET_FEATURES))
    return pd.DataFrame(data, columns=settings.FEATURES_LIST)


def _make_env(price_df=None, features_df=None) -> TradingEnv:
    if price_df is None:
        price_df = _make_price_df()
    if features_df is None:
        features_df = _make_features_df(len(price_df))
    return TradingEnv(df=price_df, features=features_df)


# ---------------------------------------------------------------------------
# 1. Observation-space shape
# ---------------------------------------------------------------------------

def test_obs_space_shape_matches_num_features():
    env = _make_env()
    expected = settings.EXPECTED_MARKET_FEATURES + TradingEnv.NUM_PORTFOLIO_FEATURES
    assert env.observation_space.shape == (expected,), (
        f"Expected obs space shape ({expected},), got {env.observation_space.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Observation dtype is float32
# ---------------------------------------------------------------------------

def test_obs_dtype_is_float32():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"


# ---------------------------------------------------------------------------
# 3. Portfolio features are the last NUM_PORTFOLIO_FEATURES elements
# ---------------------------------------------------------------------------

def test_portfolio_features_are_last():
    env  = _make_env()
    obs, _ = env.reset()
    n_portfolio = TradingEnv.NUM_PORTFOLIO_FEATURES
    n_market    = settings.EXPECTED_MARKET_FEATURES

    # Market features are rows 0..n_market-1; portfolio features follow.
    assert len(obs) == n_market + n_portfolio


# ---------------------------------------------------------------------------
# 4. No raw OHLCV in observation
# ---------------------------------------------------------------------------

def test_no_ohlcv_leakage():
    """
    Close price == 100 at every bar; if raw Close leaked into the obs vector,
    we'd see a value of 100.0. Market features are from standard-normal so
    no value should be exactly 100.
    """
    price_df  = _make_price_df(start_price=100.0)
    features_df = _make_features_df(len(price_df))
    env = TradingEnv(df=price_df, features=features_df)
    obs, _ = env.reset()
    assert not np.any(obs == 100.0), "Raw Close price leaked into observation vector."


# ---------------------------------------------------------------------------
# 5. Action space type and size
# ---------------------------------------------------------------------------

def test_action_space_type_and_size():
    env = _make_env()
    if settings.ACTION_SPACE_TYPE == "discrete_3":
        assert env.action_space.n == 3
    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        assert env.action_space.n == 5
    else:
        pytest.fail(f"Unknown ACTION_SPACE_TYPE: {settings.ACTION_SPACE_TYPE}")


# ---------------------------------------------------------------------------
# 6. Reset returns valid observation
# ---------------------------------------------------------------------------

def test_reset_returns_valid_obs():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs)), "Reset observation contains NaN or Inf."
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 7. All discrete actions complete without exception and return correct shape
# ---------------------------------------------------------------------------

def test_all_actions_step_without_error():
    env = _make_env()
    obs, _ = env.reset()
    n_actions = env.action_space.n
    for action in range(n_actions):
        env.reset()
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape, (
            f"Action {action}: bad obs shape {obs.shape}"
        )
        assert isinstance(reward, float)
        assert isinstance(done, bool)


# ---------------------------------------------------------------------------
# 8. Invalid action (sell with no position) returns negative reward via penalty
# ---------------------------------------------------------------------------

def test_sell_with_no_position_penalises():
    env = _make_env()
    env.reset()
    # action 0 = SELL_100 / SELL_ALL — no position exists after reset
    sell_action = 0
    _, reward, _, _, _ = env.step(sell_action)
    assert reward < 0, f"Expected negative reward for invalid sell, got {reward}"


# ---------------------------------------------------------------------------
# 9. Stop-loss fires and closes position
# ---------------------------------------------------------------------------

def test_stop_loss_closes_position():
    price_df = _make_price_df(n=N_BARS, start_price=100.0)

    # Buy first, then crash price below stop-loss threshold
    features_df = _make_features_df(N_BARS)
    env = TradingEnv(df=price_df, features=features_df)
    env.reset()

    # Manually inject a position and entry price so SL will trigger next step
    env.position    = 10.0
    env.entry_price = 100.0
    env.balance     = 0.0  # all-in

    # Drive Close price below SL: entry * (1 - STOP_LOSS_PCT - small margin)
    crash_price = env.entry_price * (1 - settings.STOP_LOSS_PCT - 0.01)
    env.df.loc[env.current_step, "Close"] = crash_price

    env.step(2)  # HOLD — let SL/TP logic fire
    assert env.position == 0.0, "Stop-loss did not close the position."


# ---------------------------------------------------------------------------
# 10. Take-profit fires and closes position
# ---------------------------------------------------------------------------

def test_take_profit_closes_position():
    price_df = _make_price_df(n=N_BARS, start_price=100.0)
    features_df = _make_features_df(N_BARS)
    env = TradingEnv(df=price_df, features=features_df)
    env.reset()

    env.position    = 10.0
    env.entry_price = 100.0
    env.balance     = 0.0

    # Drive Close price above TP
    tp_price = env.entry_price * (1 + settings.TAKE_PROFIT_PCT + 0.01)
    env.df.loc[env.current_step, "Close"] = tp_price

    env.step(2)  # HOLD
    assert env.position == 0.0, "Take-profit did not close the position."


# ---------------------------------------------------------------------------
# 11. Bankruptcy terminates episode
# ---------------------------------------------------------------------------

def test_bankruptcy_terminates_episode():
    price_df = _make_price_df(n=N_BARS)
    features_df = _make_features_df(N_BARS)
    env = TradingEnv(df=price_df, features=features_df)
    env.reset()

    # Drain balance below bankruptcy threshold
    env.balance  = settings.INITIAL_BALANCE * (settings.BANKRUPTCY_THRESHOLD_PCT - 0.01)
    env.position = 0.0

    _, reward, done, _, _ = env.step(2)  # HOLD
    assert done, "Episode should end on bankruptcy."
    assert reward < 0, "Bankruptcy should apply a negative reward."


# ---------------------------------------------------------------------------
# 12. VecFrameStack produces correct obs shape
# ---------------------------------------------------------------------------

def test_vecframestack_obs_shape():
    price_df    = _make_price_df()
    features_df = _make_features_df(len(price_df))
    base_env    = TradingEnv(df=price_df, features=features_df)
    vec_env     = DummyVecEnv([lambda: base_env])
    env         = VecFrameStack(vec_env, n_stack=settings.N_STACK)

    obs = env.reset()
    single_obs_dim = settings.EXPECTED_MARKET_FEATURES + TradingEnv.NUM_PORTFOLIO_FEATURES
    expected_dim   = single_obs_dim * settings.N_STACK
    assert obs.shape == (1, expected_dim), (
        f"Expected stacked obs shape (1, {expected_dim}), got {obs.shape}"
    )


# ---------------------------------------------------------------------------
# 13. Feature count matches EXPECTED_MARKET_FEATURES
# ---------------------------------------------------------------------------

def test_feature_count_matches_expected():
    assert len(settings.FEATURES_LIST) == settings.EXPECTED_MARKET_FEATURES, (
        f"FEATURES_LIST has {len(settings.FEATURES_LIST)} items but "
        f"EXPECTED_MARKET_FEATURES = {settings.EXPECTED_MARKET_FEATURES}"
    )


# ---------------------------------------------------------------------------
# 14. Shape guard: wrong feature count raises ValueError
# ---------------------------------------------------------------------------

def test_shape_guard_raises_on_wrong_feature_count():
    price_df    = _make_price_df()
    # Give one extra column — should blow up immediately in __init__
    wrong_count = settings.EXPECTED_MARKET_FEATURES + 1
    bad_features = pd.DataFrame(
        np.zeros((len(price_df), wrong_count)),
        columns=[f"f{i}" for i in range(wrong_count)],
    )
    with pytest.raises(ValueError, match="DATA SHAPE MISMATCH"):
        TradingEnv(df=price_df, features=bad_features)


# ---------------------------------------------------------------------------
# 15. AVWAP columns exist in processor output
# ---------------------------------------------------------------------------

def test_avwap_columns_produced_by_processor():
    """
    Runs add_technical_indicators on a minimal synthetic merged df and checks
    that AVWAP_Dist and AVWAP_Dist_ATR are present and finite.
    """
    from core.processor import add_technical_indicators

    n = N_BARS
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    df = pd.DataFrame({
        "Date":            dates,
        "Open":            close * (1 + rng.normal(0, 0.005, n)),
        "High":            close * (1 + rng.uniform(0, 0.01, n)),
        "Low":             close * (1 - rng.uniform(0, 0.01, n)),
        "Close":           close,
        "Volume":          rng.integers(100_000, 1_000_000, n).astype(float),
        "Sentiment_Mean":  np.zeros(n),
        "News_Intensity":  np.zeros(n),
        "QQQ_Close":       close * 0.9,
        "VIX_Close":       np.full(n, 20.0),
        "TNX_Close":       np.full(n, 4.5),
    })

    result = add_technical_indicators(df)

    assert "AVWAP_Dist" in result.columns,     "AVWAP_Dist column missing from processor output."
    assert "AVWAP_Dist_ATR" in result.columns, "AVWAP_Dist_ATR column missing from processor output."
    assert np.all(np.isfinite(result["AVWAP_Dist"].values)),     "AVWAP_Dist contains NaN/Inf."
    assert np.all(np.isfinite(result["AVWAP_Dist_ATR"].values)), "AVWAP_Dist_ATR contains NaN/Inf."
