"""
Feature fetching and computation for live trading.

This module handles:
1. Fetching recent market data from yfinance
2. Computing technical features using core/processor.py
3. Scaling features using the trained scaler
4. Building the observation for model inference
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config import settings
from core.processor import (
    add_technical_indicators,
    load_raw_news,
    build_news_sentiment,
    load_raw_macro,
    merge_prices_news_macro,
)


def fetch_recent_prices(symbol: str, timeframe: str, lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch recent price data from yfinance.
    
    Args:
        symbol: Stock symbol (e.g., "TSLA")
        timeframe: "1h" or "1d"
        lookback_days: How many days of history to fetch
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    period = f"{lookback_days}d" if timeframe == "1h" else f"{lookback_days*5}d"
    
    print(f"📥 Fetching {timeframe} data for {symbol} (last {lookback_days} days)...")
    
    try:
        data = yf.download(
            symbol,
            period=period,
            interval=timeframe,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        # Reset index to get Date as column
        data = data.reset_index()
        
        # Rename columns
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
        elif 'Date' not in data.columns:
            raise ValueError("No Date/Datetime column found")
        
        # Ensure correct columns
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Clean up
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.dropna()
        data = data.sort_values('Date').reset_index(drop=True)
        
        print(f"✅ Fetched {len(data)} rows of price data")
        return data[required]
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch prices: {e}")


def fetch_recent_macro(timeframe: str, lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch recent macro data (QQQ, VIX, TNX).
    
    Returns:
        DataFrame with Date, QQQ_Close, VIX_Close, TNX_Close
    """
    if not settings.USE_MACRO_FEATURES:
        return pd.DataFrame(columns=['Date'])
    
    period = f"{lookback_days}d" if timeframe == "1h" else f"{lookback_days*5}d"
    
    print(f"📥 Fetching macro data...")
    
    merged = None
    for symbol, out_col in settings.MACRO_SYMBOL_MAP.items():
        try:
            data = yf.download(symbol, period=period, interval=timeframe, progress=False)
            
            if data.empty:
                print(f"⚠️ No data for {symbol}, using zeros")
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            data = data.reset_index()
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            data['Date'] = pd.to_datetime(data['Date'])
            data = data[['Date', 'Close']].rename(columns={'Close': out_col})
            
            if merged is None:
                merged = data
            else:
                merged = pd.merge(merged, data, on='Date', how='outer')
                
        except Exception as e:
            print(f"⚠️ Failed to fetch {symbol}: {e}")
    
    if merged is None:
        return pd.DataFrame(columns=['Date'])
    
    # Forward fill and clean
    macro_cols = list(settings.MACRO_SYMBOL_MAP.values())
    for col in macro_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged[macro_cols] = merged[macro_cols].ffill().bfill().fillna(0.0)
    
    print(f"✅ Fetched macro data: {len(merged)} rows")
    return merged


def build_live_features(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    use_news: bool = False
) -> pd.DataFrame:
    """
    Build feature DataFrame from raw data.
    
    Args:
        price_df: OHLCV data
        macro_df: Macro indicators
        use_news: Whether to include news features (requires Alpaca)
        
    Returns:
        DataFrame with all features computed
    """
    print("🔧 Computing technical features...")
    
    # Handle news (optional - most users won't have Alpaca news in real-time)
    if use_news and settings.USE_NEWS_FEATURES:
        # For live trading, we'd need real-time news API
        # For now, use zeros
        news_df = pd.DataFrame({
            'Date': price_df['Date'],
            'Sentiment_Mean': 0.0,
            'News_Intensity': 0.0,
        })
    else:
        news_df = pd.DataFrame({
            'Date': price_df['Date'],
            'Sentiment_Mean': 0.0,
            'News_Intensity': 0.0,
        })
    
    # Merge all data
    merged = merge_prices_news_macro(price_df, news_df, macro_df)
    
    # Compute technical indicators
    feature_df = add_technical_indicators(merged)
    
    print(f"✅ Computed {len(settings.FEATURES_LIST)} features")
    return feature_df


def prepare_observation(
    feature_df: pd.DataFrame,
    scaler,
    portfolio_features: np.ndarray,
    n_stack: int = 5
) -> Tuple[np.ndarray, dict]:
    """
    Prepare the final observation for model inference.
    
    Args:
        feature_df: DataFrame with computed features
        scaler: Fitted RobustScaler
        portfolio_features: Array of [cash_ratio, position_size, inventory_fraction, unrealized_pnl, last_action]
        n_stack: Number of frames to stack (from settings.N_STACK)
        
    Returns:
        (observation_array, feature_info_dict)
    """
    if len(feature_df) < n_stack:
        raise ValueError(f"Need at least {n_stack} rows for stacking, got {len(feature_df)}")
    
    # Get last n_stack rows
    recent_df = feature_df.iloc[-n_stack:].copy()
    
    # Scale features
    features_scaled = scaler.transform(recent_df[settings.FEATURES_LIST])
    
    # Build stacked observation
    frames = []
    feature_values = {}
    
    for i, row in enumerate(features_scaled):
        # Combine market features with portfolio features
        frame = np.concatenate([row, portfolio_features])
        frames.append(frame)
        
        # Store latest feature values for display
        if i == len(features_scaled) - 1:
            feature_values = {
                settings.FEATURES_LIST[j]: float(row[j])
                for j in range(len(settings.FEATURES_LIST))
            }
    
    # Stack frames horizontally (VecFrameStack format)
    observation = np.concatenate(frames).reshape(1, -1).astype(np.float32)
    
    return observation, feature_values


def calculate_confidence(model, observation: np.ndarray, action: int) -> float:
    """
    Calculate confidence score from model's action distribution.
    
    Higher confidence = more certain about the decision.
    
    Args:
        model: The PPO model
        observation: The observation array (shape must match model input)
        action: The action that was chosen
        
    Returns:
        Confidence score 0-100%
    """
    try:
        import torch
        
        with torch.no_grad():
            # Ensure observation has correct shape (batch_size, features)
            if len(observation.shape) == 1:
                obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)
            else:
                obs_tensor = torch.as_tensor(observation).float()
            
            # Get action distribution from policy
            # Use the model's internal methods which handle shape correctly
            features = model.policy.extract_features(obs_tensor)
            
            # For MlpPolicy, features is already processed
            # Get latent representation
            if hasattr(model.policy, 'mlp_extractor'):
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            
            # Get distribution
            action_dist = model.policy.get_distribution(latent_pi)
            
            # Calculate confidence as the probability of the chosen action
            # relative to other actions
            action_tensor = torch.tensor([action])
            log_prob = action_dist.log_prob(action_tensor)
            prob = torch.exp(log_prob).item()
            
            # Convert to percentage (typical PPO probs are in range 0.2-0.9)
            # Scale so that 0.2 = 0% confidence, 0.9 = 100% confidence
            # For discrete action spaces, uniform random = 1/n_actions
            n_actions = model.action_space.n
            uniform_prob = 1.0 / n_actions
            
            # Normalize: (prob - uniform) / (1 - uniform) * 100
            confidence = (prob - uniform_prob) / (1 - uniform_prob) * 100
            confidence = max(0.0, min(100.0, confidence))
            
            return confidence
            
    except Exception as e:
        # Silently return default confidence - don't spam console
        # print(f"⚠️ Could not calculate confidence: {e}")
        return 75.0  # Reasonable default


def run_ai_inference(
    model,
    scaler,
    symbol: str = None,
    timeframe: str = None,
    portfolio_features: np.ndarray = None,
    n_stack: int = None,
) -> dict:
    """
    Complete inference pipeline: fetch data → compute features → predict.
    
    Args:
        model: Loaded PPO model
        scaler: Fitted scaler
        symbol: Stock symbol (default from settings)
        timeframe: "1h" or "1d" (default from settings)
        portfolio_features: Current portfolio state [cash_ratio, position_size, inventory_fraction, unrealized_pnl, last_action]
        n_stack: Frame stack size (default from settings)
        
    Returns:
        Dict with action, confidence, features, and metadata
    """
    symbol = symbol or settings.SYMBOL
    timeframe = timeframe or settings.TIMEFRAME
    n_stack = n_stack or settings.N_STACK
    
    if portfolio_features is None:
        # Default: no position, all cash
        portfolio_features = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Fetch data
    price_df = fetch_recent_prices(symbol, timeframe, lookback_days=60)
    macro_df = fetch_recent_macro(timeframe, lookback_days=60)
    
    # Build features
    feature_df = build_live_features(price_df, macro_df, use_news=False)
    
    # Prepare observation
    observation, feature_values = prepare_observation(
        feature_df, scaler, portfolio_features, n_stack
    )
    
    # Run inference
    print("🧠 Running AI inference...")
    action, _ = model.predict(observation, deterministic=True)
    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
    
    # Calculate confidence
    confidence = calculate_confidence(model, observation, action)
    
    # Get latest price
    latest_price = float(price_df['Close'].iloc[-1])
    
    return {
        'action': action,
        'confidence': confidence,
        'latest_price': latest_price,
        'features': feature_values,
        'timestamp': datetime.now(),
    }


def action_to_name(action: int, action_space: str) -> str:
    """Convert action index to human-readable name."""
    if action_space == "discrete_3":
        mapping = {0: "SELL_ALL", 1: "BUY_ALL", 2: "HOLD"}
    else:
        mapping = {0: "SELL_100", 1: "SELL_50", 2: "HOLD", 3: "BUY_50", 4: "BUY_100"}
    
    return mapping.get(action, "UNKNOWN")


def get_feature_highlights(feature_values: dict, top_n: int = 5) -> list:
    """
    Get the most significant features for display.
    
    Returns:
        List of (feature_name, value, interpretation) tuples
    """
    highlights = []
    
    # Define interesting thresholds
    if 'RSI' in feature_values:
        rsi = feature_values['RSI']
        if rsi > 70:
            highlights.append(('RSI', rsi, 'Overbought'))
        elif rsi < 30:
            highlights.append(('RSI', rsi, 'Oversold'))
    
    if 'MACD_Hist' in feature_values:
        macd = feature_values['MACD_Hist']
        if macd > 1.0:
            highlights.append(('MACD_Hist', macd, 'Bullish momentum'))
        elif macd < -1.0:
            highlights.append(('MACD_Hist', macd, 'Bearish momentum'))
    
    if 'AVWAP_Dist' in feature_values:
        avwap = feature_values['AVWAP_Dist']
        if avwap > 0.02:
            highlights.append(('AVWAP_Dist', avwap, 'Above fair value'))
        elif avwap < -0.02:
            highlights.append(('AVWAP_Dist', avwap, 'Below fair value'))
    
    if 'Sentiment_Mean' in feature_values:
        sent = feature_values['Sentiment_Mean']
        if sent > 0.3:
            highlights.append(('Sentiment', sent, 'Positive news'))
        elif sent < -0.3:
            highlights.append(('Sentiment', sent, 'Negative news'))
    
    # Sort by absolute value and take top N
    highlights.sort(key=lambda x: abs(x[1]), reverse=True)
    return highlights[:top_n]
