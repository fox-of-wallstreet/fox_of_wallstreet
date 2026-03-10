from config import settings
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, features):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.features = features.values if isinstance(features, pd.DataFrame) else features

        # 16 Market Features + 4 Portfolio Features = 20
        self.num_features = self.features.shape[1] + 4

        # 🟢 SUB-CHECKPOINT 2.1: DYNAMIC ACTION SPACE
        # The environment reads settings.py and adapts instantly!
        if settings.ACTION_SPACE_TYPE == "discrete_3":
            self.action_space = spaces.Discrete(3) # 0: Sell All, 1: Buy All, 2: Hold
        elif settings.ACTION_SPACE_TYPE == "discrete_5":
            self.action_space = spaces.Discrete(5) # 0: Sell 100%, 1: Sell 50%, 2: Hold, 3: Buy 50%, 4: Buy 100%
        else:
            raise ValueError("Invalid ACTION_SPACE_TYPE in settings.py")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0.0 # Number of shares
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.prev_portfolio_value = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self):
        """🟢 SUB-CHECKPOINT 2.3: STATE REPRESENTATION"""
        obs = self.features[self.current_step].copy()

        current_price = self.df.loc[self.current_step, 'Close']
        unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price if self.position > 0 else 0.0
        cash_ratio = self.balance / self.initial_balance

        # The internal wallet features
        portfolio_features = np.array([
            1.0 if self.position > 0 else 0.0,
            unrealized_pnl_pct,
            cash_ratio,
            self.bars_in_trade / 100.0
        ])

        full_obs = np.hstack((obs, portfolio_features))
        return full_obs.astype(np.float32)

    def _execute_trade(self, action, current_price):
        """🟢 SUB-CHECKPOINT 2.2: REALISTIC FRICTION & INVALID ACTION PENALTIES"""
        slippage_pct = 0.0005
        step_reward_penalty = 0.0

        # ---------------------------------------------------------
        # SCENARIO A: 3-ACTION SPACE (Conviction Trading)
        # ---------------------------------------------------------
        if settings.ACTION_SPACE_TYPE == "discrete_3":

            if action == 1:  # Attempt to BUY ALL
                if self.balance > 0:
                    investment = self.balance * settings.CASH_RISK_FRACTION
                    actual_buy_price = current_price * (1 + slippage_pct)
                    self.position = investment / actual_buy_price
                    self.balance -= investment
                    self.entry_price = actual_buy_price
                    self.bars_in_trade = 1
                    step_reward_penalty -= 0.01
                else:
                    step_reward_penalty -= 0.05 # 🚫 Invalid: Buy with no cash

            elif action == 0:  # Attempt to SELL ALL
                if self.position > 0:
                    actual_sell_price = current_price * (1 - slippage_pct)
                    self.balance += self.position * actual_sell_price
                    self.position = 0
                    self.entry_price = 0.0
                    self.bars_in_trade = 0
                    step_reward_penalty -= 0.01
                else:
                    step_reward_penalty -= 0.05 # 🚫 Invalid: Sell with no shares

            elif action == 2: # HOLD
                if self.position > 0:
                    self.bars_in_trade += 1

        # ---------------------------------------------------------
        # SCENARIO B: 5-ACTION SPACE (Scaling In/Out)
        # ---------------------------------------------------------
        elif settings.ACTION_SPACE_TYPE == "discrete_5":

            if action in [3, 4]: # Attempt to BUY (50% or 100%)
                if self.balance > 0:
                    fraction = 1.0 if action == 4 else 0.5
                    investment = (self.balance * fraction) * settings.CASH_RISK_FRACTION
                    actual_buy_price = current_price * (1 + slippage_pct)
                    new_shares = investment / actual_buy_price

                    total_cost = (self.position * self.entry_price) + investment
                    self.position += new_shares
                    self.entry_price = total_cost / self.position if self.position > 0 else 0
                    self.balance -= investment
                    self.bars_in_trade = 1 if self.bars_in_trade == 0 else self.bars_in_trade
                    step_reward_penalty -= 0.01 if action == 4 else 0.005
                else:
                    step_reward_penalty -= 0.05 # 🚫 Invalid: Buy with no cash

            elif action in [0, 1]: # Attempt to SELL (100% or 50%)
                if self.position > 0:
                    fraction = 1.0 if action == 0 else 0.5
                    shares_to_sell = self.position * fraction
                    actual_sell_price = current_price * (1 - slippage_pct)
                    self.balance += shares_to_sell * actual_sell_price
                    self.position -= shares_to_sell

                    if action == 0: # Full liquidation resets trade stats
                        self.entry_price = 0.0
                        self.bars_in_trade = 0

                    step_reward_penalty -= 0.01 if action == 0 else 0.005
                else:
                    step_reward_penalty -= 0.05 # 🚫 Invalid: Sell with no shares

            elif action == 2: # HOLD
                if self.position > 0:
                    self.bars_in_trade += 1

        return step_reward_penalty

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        done = False
        info = {'step': self.current_step, 'action': action, 'price': current_price}

        # 1. Execute the trade and get the friction penalty
        reward = self._execute_trade(action, current_price)

        # 2. Calculate Portfolio Change
        current_portfolio_value = self.balance + (self.position * current_price)
        step_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

        # 🟢 SUB-CHECKPOINT 2.4: CONFIGURABLE REWARD LOGIC
        if settings.REWARD_STRATEGY == "absolute_asymmetric":
            # 2x penalty on losses to teach capital preservation
            if step_return > 0:
                reward += step_return * 100
            else:
                reward += step_return * 200

        elif settings.REWARD_STRATEGY == "pure_pnl":
            # Just reward raw profit 1:1 (A baseline strategy)
            reward += step_return * 100

        else:
            raise ValueError("Invalid REWARD_STRATEGY in settings.py")

        self.prev_portfolio_value = current_portfolio_value

        # 3. Advance Step & Check Bankruptcy
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        if current_portfolio_value <= self.initial_balance * 0.5:
            done = True
            reward -= 10.0

        info['portfolio_value'] = current_portfolio_value
        return self._next_observation(), float(reward), done, False, info
