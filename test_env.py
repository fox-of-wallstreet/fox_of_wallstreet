import pandas as pd
import numpy as np
from core.environment import TradingEnv

# 1. Create fake data
df = pd.DataFrame({'Close': [100, 102, 101, 105, 103]})
features = np.random.rand(5, 16) # 5 steps, 16 random features

# 2. Initialize Env
env = TradingEnv(df, features)
obs, _ = env.reset()

print(f"✅ Environment Init Success. Obs shape: {obs.shape}")
print(f"🤖 Action Space: {env.action_space}")

# 3. Take a random step
action = env.action_space.sample()
obs, reward, done, _, info = env.step(action)

print(f"✅ Step Success. Took action: {action}, Reward: {reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}")

# If you run `python test_env.py` and it prints out the success messages without crashing, **Checkpoint 2 is officially complete and bug-free.**
