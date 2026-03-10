import os
import sys
import numpy as np
import pandas as pd

# Ensure Python can find your 'core' and 'config' folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import TradingEnv
from config import settings

def test_environment_math():
    print("🧪 RUNNING DETERMINISTIC UNIT TESTS...")

    # 1. SETUP: Override settings for a predictable test
    settings.ACTION_SPACE_TYPE = "discrete_3"
    settings.REWARD_STRATEGY = "absolute_asymmetric"
    settings.CASH_RISK_FRACTION = 1.0 # Use 100% of cash to make math easy

    # 2. FAKE DATA: We control the future.
    df = pd.DataFrame({'Close': [100.0, 110.0, 99.0]})
    features = np.zeros((3, 16))

    env = TradingEnv(df, features)
    env.reset()

    # ==========================================
    # 🧪 TEST 1: The "Ghost Trade" Penalty
    # ==========================================
    _, reward, _, _, _ = env.step(0)
    assert round(reward, 2) == -0.05, f"Expected -0.05, got {reward}"
    print("✅ TEST 1 PASSED: Invalid Action correctly penalized (-0.05).")

    # ==========================================
    # 🧪 TEST 2: Slippage & Execution Math
    # ==========================================
    env.reset()
    _, reward, _, _, info = env.step(1)

    expected_shares = 10000.0 / 100.05
    assert round(env.position, 4) == round(expected_shares, 4), f"Execution math failed! Got {env.position}"
    print(f"✅ TEST 2 PASSED: Slippage executed perfectly. Shares owned: {env.position:.2f}")

    # ==========================================
    # 🧪 TEST 3: Asymmetric Reward Logic (Gains)
    # ==========================================
    _, reward, _, _, info = env.step(2)

    expected_return = (info['portfolio_value'] - 10000.0) / 10000.0
    expected_reward = expected_return * 100
    assert round(reward, 4) == round(expected_reward, 4), "Reward calculation failed!"
    print(f"✅ TEST 3 PASSED: Positive reward calculated perfectly: +{reward:.2f}")

    # ==========================================
    # 🧪 TEST 4: Asymmetric Reward Logic (Losses)
    # ==========================================
    prev_value = info['portfolio_value']
    _, reward, _, _, info = env.step(2)

    expected_return = (info['portfolio_value'] - prev_value) / prev_value
    expected_reward = expected_return * 200  # <--- The 2x Multiplier
    assert round(reward, 4) == round(expected_reward, 4), "Asymmetric loss calculation failed!"
    print(f"✅ TEST 4 PASSED: 2x Asymmetric loss calculated perfectly: {reward:.2f}")

    # ==========================================
    # 🧪 TEST 5: 5-Action Space (Partial Buys)
    # ==========================================
    settings.ACTION_SPACE_TYPE = "discrete_5"
    env = TradingEnv(df, features) # Re-init to grab the new action space
    env.reset()

    # Action 3 is "Light Buy" (50% of cash). Cash = $10,000. Price = $100.00.
    _, reward, _, _, info = env.step(3)
    expected_light_shares = 5000.0 / 100.05
    assert round(env.position, 4) == round(expected_light_shares, 4), "Partial Buy Math Failed!"
    assert round(env.balance, 2) == 5000.0, "Balance not properly deducted!"
    print(f"✅ TEST 5 PASSED: Extended Discrete 50% buy executed perfectly.")

    print("\n🛡️ ALL CORE ENVIRONMENT TESTS PASSED. The math is bulletproof.")

if __name__ == "__main__":
    test_environment_math()
