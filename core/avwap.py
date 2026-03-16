"""
Leakage-safe Anchored VWAP feature computation.

Produces two columns:
  AVWAP_Dist     = (Close / Active_AVWAP) - 1
  AVWAP_Dist_ATR = (Close - Active_AVWAP) / ATR_14

Design decisions
----------------
- Anchor policy: confirmed, significant swing pivot (high or low), leakage-safe.
- Pivot confirmation: a pivot at bar i is only usable on bar i+R, after all right-side
  bars have closed. No lookahead ever touches a bar that has not yet occurred.
- Significance filter: structure filter (higher-high / lower-low) AND ATR threshold.
- ATR: Wilder's smoothing (period=14), computed independently of processor ATR_Pct
  which uses a simple rolling mean.
- Initial anchor: bar 0. Before the first pivot is accepted the AVWAP accumulates
  from the very first bar, producing a full-history VWAP until structure reasserts.
- last_accepted_high starts at -inf and last_accepted_low starts at +inf so that
  the very first confirmed pivot always passes the significance filter naturally.
- When a new anchor is accepted, running sums are rebuilt from the new anchor bar
  to the current bar. This is O(right_window) work and is safe for batch processing.
"""

import numpy as np

from config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range array. First bar uses H-L only (no previous close)."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR using Wilder's smoothing.

    First ATR value = simple mean of the first `period` True Range values.
    Subsequent values = (ATR_prev * (period - 1) + TR_current) / period.
    Bars before the warm-up period are NaN.
    """
    tr = _compute_tr(high, low, close)
    n = len(tr)
    atr = np.full(n, np.nan)

    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_avwap_features(df):
    """
    Add AVWAP_Dist and AVWAP_Dist_ATR columns to df in-place and return df.

    Called by processor.py registry functions. The df passed here has already
    been copied by add_technical_indicators, so in-place assignment is safe.

    Processing order per bar t
    --------------------------
    1. Accumulate running HLC3*Volume and Volume sums (for AVWAP from anchor).
    2. Compute TruRange; update Wilder ATR once warm-up is complete.
    3. Identify candidate pivot bar = t - right. Check it now, because all right-side
       bars through bar t have closed — the confirmation window is complete.
    4. Test both pivot high and pivot low for the candidate bar.
       For each that passes pivot detection:
         a. Structure filter: must be a higher high / lower low than last accepted.
         b. ATR threshold: move from last accepted must be >= k * ATR_14[t].
       Only runs when ATR is valid (past warm-up).
    5. On acceptance: update anchor, last_accepted reference, rebuild running sums
       from new anchor to t.
    6. Compute AVWAP and write both feature values.
    """
    if settings.TIMEFRAME == "1h":
        left  = settings.AVWAP_PIVOT_LEFT_H
        right = settings.AVWAP_PIVOT_RIGHT_H
        k     = settings.AVWAP_ATR_K_H
    else:
        left  = settings.AVWAP_PIVOT_LEFT_D
        right = settings.AVWAP_PIVOT_RIGHT_D
        k     = settings.AVWAP_ATR_K_D

    n      = len(df)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    close  = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    hlc3   = (high + low + close) / 3.0

    atr = _wilder_atr(high, low, close, period=14)

    avwap_dist     = np.zeros(n)
    avwap_dist_atr = np.zeros(n)

    # Anchor state
    anchor_bar         = 0
    cum_pv             = hlc3[0] * volume[0]   # cumulative (HLC3 * Volume) from anchor
    cum_v              = volume[0]              # cumulative Volume from anchor

    # Significance state — independent per side
    last_accepted_high = -np.inf
    last_accepted_low  =  np.inf

    for t in range(1, n):

        # ------------------------------------------------------------------
        # Step 1: accumulate running sums for the new bar
        # ------------------------------------------------------------------
        cum_pv += hlc3[t] * volume[t]
        cum_v  += volume[t]

        # ------------------------------------------------------------------
        # Step 2: pivot confirmation check
        # candidate bar = t - right  (all right-side bars have now closed)
        # requires candidate to have at least `left` bars to its left
        # ------------------------------------------------------------------
        candidate = t - right
        if candidate >= left:
            w_high = high[candidate - left: t + 1]   # window length = left+right+1
            w_low  = low[candidate - left: t + 1]

            # --- Pivot HIGH ---
            if high[candidate] == np.max(w_high) and np.isfinite(atr[t]):
                passes_structure = high[candidate] > last_accepted_high
                passes_atr       = (high[candidate] - last_accepted_high) >= k * atr[t]
                if passes_structure and passes_atr:
                    anchor_bar         = candidate
                    last_accepted_high = high[candidate]
                    # Rebuild sums from new anchor to current bar
                    cum_pv = np.dot(hlc3[anchor_bar: t + 1], volume[anchor_bar: t + 1])
                    cum_v  = np.sum(volume[anchor_bar: t + 1])

            # --- Pivot LOW ---
            if low[candidate] == np.min(w_low) and np.isfinite(atr[t]):
                passes_structure = low[candidate] < last_accepted_low
                passes_atr       = (last_accepted_low - low[candidate]) >= k * atr[t]
                if passes_structure and passes_atr:
                    anchor_bar        = candidate
                    last_accepted_low = low[candidate]
                    # Rebuild sums from new anchor to current bar
                    cum_pv = np.dot(hlc3[anchor_bar: t + 1], volume[anchor_bar: t + 1])
                    cum_v  = np.sum(volume[anchor_bar: t + 1])

        # ------------------------------------------------------------------
        # Step 3: compute features for bar t
        # ------------------------------------------------------------------
        if cum_v > 0:
            active_avwap   = cum_pv / cum_v
            avwap_dist[t]  = (close[t] / (active_avwap + 1e-8)) - 1
            if np.isfinite(atr[t]) and atr[t] > 0:
                avwap_dist_atr[t] = (close[t] - active_avwap) / atr[t]

    df["AVWAP_Dist"]     = avwap_dist
    df["AVWAP_Dist_ATR"] = avwap_dist_atr
    return df
