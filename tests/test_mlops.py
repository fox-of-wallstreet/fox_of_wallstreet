import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.processor import add_technical_indicators, prepare_features
from config import settings

def test_processor_logic():
    print("🧪 RUNNING PROCESSOR & SCALER UNIT TESTS...")

    # 1. Create Fake OHLCV Data
    dates = pd.date_range(start="2025-01-01", periods=50, freq="h")
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, 50),
        'High': np.random.uniform(105, 115, 50),
        'Low': np.random.uniform(95, 105, 50),
        'Close': np.random.uniform(100, 110, 50),
        'Volume': np.random.randint(1000, 5000, 50)
    })

    # ==========================================
    # 🧪 TEST 1: Indicator Generation
    # ==========================================
    processed_df = add_technical_indicators(df)

    expected_columns = ['Log_Return', 'Volume_Z_Score', 'RSI', 'MACD_Hist', 'BB_Pct', 'ATR_Pct']
    for col in expected_columns:
        assert col in processed_df.columns, f"❌ Missing indicator column: {col}"

    assert processed_df.isna().sum().sum() == 0, "❌ NaN values found in processed DataFrame!"
    print("✅ TEST 1 PASSED: Technical indicators generated perfectly with zero NaNs.")

    # ==========================================
    # 🧪 TEST 2: RobustScaler Vault Integration
    # ==========================================
    features_list = ['Log_Return', 'Volume_Z_Score', 'RSI'] # Testing a subset

    # Force the Vault path to exist
    os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)

    # Simulate Training (Should FIT and SAVE)
    scaled_train = prepare_features(processed_df, features_list, is_training=True)

    assert os.path.exists(settings.SCALER_PATH), "❌ Scaler was not saved to the Artifact Vault!"
    assert scaled_train.shape[1] == len(features_list), "❌ Scaled output shape mismatch."
    print(f"✅ TEST 2 PASSED: RobustScaler successfully fitted and saved to {settings.SCALER_PATH}")

    # ==========================================
    # 🧪 TEST 3: RobustScaler Loading
    # ==========================================
    # Simulate Backtesting (Should LOAD and TRANSFORM)
    scaled_test = prepare_features(processed_df, features_list, is_training=False)

    # The output should be mathematically identical since we passed the same data
    np.testing.assert_array_almost_equal(scaled_train, scaled_test, err_msg="❌ Loaded scaler produced different results!")
    print("✅ TEST 3 PASSED: RobustScaler successfully loaded and transformed data.")

    print("\n🛡️ ALL PROCESSOR TESTS PASSED. Data pipeline is secure.")

if __name__ == "__main__":
    test_processor_logic()
