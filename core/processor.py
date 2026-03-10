import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler

from config import settings
from config.settings import (
    RSI_WINDOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    VOLATILITY_WINDOW, SHORT_VOL_WINDOW, LONG_VOL_WINDOW,
    MA_FAST_WINDOW, MA_SLOW_WINDOW, SCALER_PATH
)

def add_technical_indicators(df):
    """Calculates a richer RL state space using real market, macro, and news features."""
    print("📈 Calculating advanced Micro + Macro + Context indicators...")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 1. MICRO: PRICE & VOLUME
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    vol_mean = df['Volume'].rolling(window=VOLATILITY_WINDOW).mean()
    vol_std = df['Volume'].rolling(window=VOLATILITY_WINDOW).std()
    df['Volume_Z_Score'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

    # 2. MICRO: TREND & VOLATILITY
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_WINDOW).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    sma_vol = df['Close'].rolling(window=VOLATILITY_WINDOW).mean()
    std_vol = df['Close'].rolling(window=VOLATILITY_WINDOW).std()
    upper_band = sma_vol + (std_vol * 2)
    lower_band = sma_vol - (std_vol * 2)
    df['BB_Pct'] = (df['Close'] - lower_band) / (upper_band - lower_band + 1e-8)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=RSI_WINDOW).mean()
    df['ATR_Pct'] = df['ATR'] / (df['Close'] + 1e-8)

    # 3. VOLATILITY REGIME
    df['Realized_Vol_Short'] = df['Log_Return'].rolling(window=SHORT_VOL_WINDOW).std()
    df['Realized_Vol_Long'] = df['Log_Return'].rolling(window=LONG_VOL_WINDOW).std()
    df['Vol_Regime'] = df['Realized_Vol_Short'] / (df['Realized_Vol_Long'] + 1e-8)

    # 4. TREND DISTANCE
    ma_fast = df['Close'].rolling(window=MA_FAST_WINDOW).mean()
    ma_slow = df['Close'].rolling(window=MA_SLOW_WINDOW).mean()

    df['Dist_MA_Fast'] = (df['Close'] - ma_fast) / (ma_fast + 1e-8)
    df['Dist_MA_Slow'] = (df['Close'] - ma_slow) / (ma_slow + 1e-8)

    # 5. MACRO: MARKET WEATHER & RELATIVE STRENGTH
    df['QQQ_Ret'] = np.log(df['QQQ_Close'] / df['QQQ_Close'].shift(1))
    df['ARKK_Ret'] = np.log(df['ARKK_Close'] / df['ARKK_Close'].shift(1))
    df['Rel_Strength_QQQ'] = df['Log_Return'] - df['QQQ_Ret']

    vix_mean = df['VIX_Close'].rolling(window=LONG_VOL_WINDOW).mean()
    vix_std = df['VIX_Close'].rolling(window=LONG_VOL_WINDOW).std()
    df['VIX_Z'] = (df['VIX_Close'] - vix_mean) / (vix_std + 1e-8)

    tnx_mean = df['TNX_Close'].rolling(window=LONG_VOL_WINDOW).mean()
    tnx_std = df['TNX_Close'].rolling(window=LONG_VOL_WINDOW).std()
    df['TNX_Z'] = (df['TNX_Close'] - tnx_mean) / (tnx_std + 1e-8)

    # 6. CONTEXT: NEWS
    if 'Sentiment_EMA' in df.columns:
        df['Sentiment_EMA'] = df['Sentiment_EMA'].ewm(span=5, adjust=False).mean()
    else:
        df['Sentiment_EMA'] = 0.0

    if 'News_Intensity' in df.columns:
        df['News_Intensity'] = np.log1p(df['News_Intensity'])
    else:
        df['News_Intensity'] = 0.0

    # 7. CONTEXT: TIME
    if settings.TIMEFRAME == "1h":
        hour_of_day = df['Date'].dt.hour + (df['Date'].dt.minute / 60.0)
        df['Sin_Time'] = np.sin(2 * np.pi * hour_of_day / 24.0)
        df['Cos_Time'] = np.cos(2 * np.pi * hour_of_day / 24.0)

        current_minutes = df['Date'].dt.hour * 60 + df['Date'].dt.minute
        df['Mins_to_Close'] = ((16 * 60) - current_minutes).clip(lower=0)

    elif settings.TIMEFRAME == "1d":
        # Intraday features do not make sense for daily bars
        df['Sin_Time'] = 0.0
        df['Cos_Time'] = 0.0
        df['Mins_to_Close'] = 0.0

    else:
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

    # Clean up intermediate/raw columns
    cols_to_drop = [
        'ATR', 'Open', 'High', 'Low', 'Volume',
        'VIX_Close', 'QQQ_Close', 'ARKK_Close', 'TNX_Close'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df.dropna().reset_index(drop=True)


def prepare_features(df, features_list, is_training=True):
    """Scales the data."""
    if is_training:
        print("⚖️ Fitting new RobustScaler...")
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df[features_list])

        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler saved to {SCALER_PATH}")
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"❌ No scaler found at {SCALER_PATH}. Train the model first!")

        print("⚖️ Loading existing RobustScaler for inference...")
        scaler = joblib.load(SCALER_PATH)
        scaled_data = scaler.transform(df[features_list])

    return pd.DataFrame(scaled_data, columns=features_list)
