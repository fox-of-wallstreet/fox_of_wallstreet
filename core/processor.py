import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler

# 🎛️ Connect to the Control Room
from config import settings

def add_technical_indicators(df):
    """Calculates all 16 features. Shared between Train, Backtest, and Live."""
    df = df.copy()

    # 1. Micro (Price Action)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volume_Z_Score'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal

    # Bollinger Bands %B
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    df['BB_Pct'] = (df['Close'] - lower_band) / (upper_band - lower_band)

    # ATR %
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_Pct'] = (true_range.rolling(14).mean() / df['Close']) * 100

    # 2. Mocking Macro & News (Simulating the external APIs for standalone training)
    # In a live environment, you would merge actual QQQ, VIX, and FinBERT data here.
    df['QQQ_Ret'] = df['Log_Return'] * 0.8 + np.random.normal(0, 0.001, len(df))
    df['ARKK_Ret'] = df['Log_Return'] * 1.2 + np.random.normal(0, 0.002, len(df))
    df['Rel_Strength_QQQ'] = df['Log_Return'] - df['QQQ_Ret']
    df['VIX_Level'] = 20.0 + np.random.normal(0, 1, len(df))
    df['TNX_Level'] = 4.0 + np.random.normal(0, 0.1, len(df))
    df['Sentiment_EMA'] = np.random.uniform(-1, 1, len(df))
    df['News_Intensity'] = np.random.uniform(0, 1, len(df))

    # 3. Time Features
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute
    total_minutes = df['Hour'] * 60 + df['Minute']

    df['Sin_Time'] = np.sin(2 * np.pi * total_minutes / 1440)
    df['Cos_Time'] = np.cos(2 * np.pi * total_minutes / 1440)
    df['Mins_to_Close'] = 960 - total_minutes # 16:00 is 960 mins

    # Clean up NaNs from rolling windows
    df = df.fillna(0)
    return df

def prepare_features(df, features_list, is_training=True):
    """Scales the features and manages the Scaler Vault"""
    data_to_scale = df[features_list]

    if is_training:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)
        joblib.dump(scaler, settings.SCALER_PATH)
        print(f"⚖️ RobustScaler FITTED and SAVED to: {settings.SCALER_PATH}")
    else:
        if not os.path.exists(settings.SCALER_PATH):
            raise FileNotFoundError(f"❌ Scaler not found at {settings.SCALER_PATH}. You must train first!")
        scaler = joblib.load(settings.SCALER_PATH)
        scaled_data = scaler.transform(data_to_scale)
        print(f"⚖️ RobustScaler LOADED from: {settings.SCALER_PATH}")

    return scaled_data
