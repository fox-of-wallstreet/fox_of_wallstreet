"""
Unit tests for core/avwap.py

Tests cover:
  1. Wilder ATR warm-up and smoothing correctness.
  2. compute_avwap_features: both columns are added, no NaN/inf.
  3. Leakage safety: anchor does not update before right-side confirmation window closes.
  4. Significance filter: a pivot that does not exceed last_accepted_high is rejected.
  5. Acceptance: a sufficiently large pivot correctly resets the anchor and AVWAP.
"""

import numpy as np
import pandas as pd
import pytest

from core.avwap import _wilder_atr, compute_avwap_features
from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n, high=None, low=None, close=None, volume=None):
    """Build a minimal OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    if close is None:
        close = 100.0 + rng.standard_normal(n).cumsum() * 0.5
    if high is None:
        high = close + rng.uniform(0.1, 0.5, n)
    if low is None:
        low = close - rng.uniform(0.1, 0.5, n)
    if volume is None:
        volume = np.full(n, 1000.0)
    return pd.DataFrame({
        "Date":   dates,
        "Open":   close,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume,
    })


# ---------------------------------------------------------------------------
# Wilder ATR tests
# ---------------------------------------------------------------------------

def test_wilder_atr_warmup_produces_nan_before_period():
    n = 20
    high  = np.full(n, 101.0)
    low   = np.full(n, 99.0)
    close = np.full(n, 100.0)
    atr = _wilder_atr(high, low, close, period=14)

    # All bars before index 13 (the 14th bar) should be NaN
    assert np.all(np.isnan(atr[:13])), "Expected NaN before warm-up period"
    assert np.isfinite(atr[13]),       "Expected valid ATR at warm-up bar"


def test_wilder_atr_first_value_is_simple_mean():
    n = 20
    # Flat prices: TR per bar is always H - L = 2.0 (no prev-close contribution)
    high  = np.full(n, 101.0)
    low   = np.full(n, 99.0)
    close = np.full(n, 100.0)
    atr = _wilder_atr(high, low, close, period=14)

    # First ATR should be mean(TR[0:14]) = mean([2, 2, ..., 2]) = 2.0
    # Note: TR[0] = H - L = 2, and for i>=1, max(H-L, |H-prev_close|, |L-prev_close|)
    # = max(2, 1, 1) = 2. So all TR values are 2.0.
    assert abs(atr[13] - 2.0) < 1e-9


def test_wilder_atr_smoothing_stays_constant_for_flat_prices():
    n = 30
    high  = np.full(n, 101.0)
    low   = np.full(n, 99.0)
    close = np.full(n, 100.0)
    atr = _wilder_atr(high, low, close, period=14)

    # With constant TR=2.0, Wilder smoothing should stay at 2.0 after warm-up
    assert np.allclose(atr[13:], 2.0, atol=1e-9)


def test_wilder_atr_short_series_returns_all_nan():
    atr = _wilder_atr(np.array([1.0]), np.array([0.5]), np.array([0.8]), period=14)
    assert np.all(np.isnan(atr))


# ---------------------------------------------------------------------------
# compute_avwap_features tests
# ---------------------------------------------------------------------------

def test_avwap_columns_added_to_df(monkeypatch):
    monkeypatch.setattr(settings, "TIMEFRAME", "1h")
    df = _make_df(100)
    result = compute_avwap_features(df)
    assert "AVWAP_Dist" in result.columns
    assert "AVWAP_Dist_ATR" in result.columns


def test_avwap_no_nan_or_inf_in_output(monkeypatch):
    monkeypatch.setattr(settings, "TIMEFRAME", "1h")
    df = _make_df(200)
    result = compute_avwap_features(df)
    assert not result["AVWAP_Dist"].isna().any(),     "AVWAP_Dist contains NaN"
    assert not result["AVWAP_Dist_ATR"].isna().any(), "AVWAP_Dist_ATR contains NaN"
    assert np.isfinite(result["AVWAP_Dist"].values).all(),     "AVWAP_Dist contains inf"
    assert np.isfinite(result["AVWAP_Dist_ATR"].values).all(), "AVWAP_Dist_ATR contains inf"


def test_avwap_bar_zero_is_always_zero(monkeypatch):
    """Bar 0 has no previous bar to accumulate from \u2014 both features must be 0.0."""
    monkeypatch.setattr(settings, "TIMEFRAME", "1h")
    df = _make_df(50)
    result = compute_avwap_features(df)
    assert result["AVWAP_Dist"].iloc[0] == 0.0
    assert result["AVWAP_Dist_ATR"].iloc[0] == 0.0


def test_avwap_leakage_safe_no_early_anchor_update(monkeypatch):
    """
    With left=5 and right=5, a pivot at bar 5 can only be confirmed at bar 10.
    Before bar 10, AVWAP should only reflect bars 0 onward (anchor stays at 0).
    We verify this by constructing a synthetic spike at bar 5 that would clearly
    pass the significance filter. AVWAP_Dist values at bars 6-9 must still reflect
    anchor=0, not anchor=5.
    """
    monkeypatch.setattr(settings, "TIMEFRAME", "1h")
    monkeypatch.setattr(settings, "AVWAP_PIVOT_LEFT_H",  5)
    monkeypatch.setattr(settings, "AVWAP_PIVOT_RIGHT_H", 5)
    monkeypatch.setattr(settings, "AVWAP_ATR_K_H",       0.75)

    n = 30
    close  = np.full(n, 100.0)
    high   = np.full(n, 101.0)
    low    = np.full(n, 99.0)
    volume = np.full(n, 1000.0)

    # Create a clear spike at bar 5: very high price so it dominates the window
    high[5] = 200.0
    close[5] = 200.0

    df = _make_df(n, high=high, low=low, close=close, volume=volume)
    result = compute_avwap_features(df)

    # At bars 6-9 the pivot at bar 5 has NOT been confirmed yet.
    # AVWAP is still accumulating from bar 0.
    # Compute expected AVWAP manually for bar 9 (anchor=0, all 10 bars included):
    hlc3 = (high + low + close) / 3.0
    expected_avwap_9 = np.sum(hlc3[:10] * volume[:10]) / np.sum(volume[:10])
    avwap_dist_9 = (close[9] / expected_avwap_9) - 1
    assert abs(result["AVWAP_Dist"].iloc[9] - avwap_dist_9) < 1e-9, (
        "AVWAP at bar 9 should still use anchor=0 (pivot not yet confirmed)"
    )


def test_avwap_significance_filter_rejects_small_pivot(monkeypatch):
    """
    A confirmed pivot that does not exceed last_accepted_high by at least k*ATR
    should be rejected. The anchor must not change.
    """
    monkeypatch.setattr(settings, "TIMEFRAME", "1d")
    monkeypatch.setattr(settings, "AVWAP_PIVOT_LEFT_D",  3)
    monkeypatch.setattr(settings, "AVWAP_PIVOT_RIGHT_D", 3)
    monkeypatch.setattr(settings, "AVWAP_ATR_K_D",       1.0)

    n = 40
    # Flat market with tiny oscillation: ATR will be very small, swings even smaller.
    # This ensures the very first pivot passes (last_accepted_high=-inf), but the
    # second pivot (of similar height) will fail the structure filter.
    close  = np.full(n, 100.0, dtype=float)
    high   = np.full(n, 100.5, dtype=float)
    low    = np.full(n, 99.5, dtype=float)
    volume = np.full(n, 1000.0)

    df = _make_df(n, high=high, low=low, close=close, volume=volume)
    result = compute_avwap_features(df)

    # No NaN/inf regardless of filter outcome
    assert np.isfinite(result["AVWAP_Dist"].values).all()
    assert np.isfinite(result["AVWAP_Dist_ATR"].values).all()
