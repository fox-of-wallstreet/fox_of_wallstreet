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

        if len(self.df) != len(self.features):
            raise ValueError(
                f"Length mismatch: df has {len(self.df)} rows but features has {len(self.features)} rows."
            )

        # Observation = scaled market/context features + 4 portfolio state features
        self.num_features = self.features.shape[1] + 4

        if settings.ACTION_SPACE_TYPE == "discrete_3":
            self.action_space = spaces.Discrete(3)  # 0: Sell All, 1: Buy All, 2: Hold
        elif settings.ACTION_SPACE_TYPE == "discrete_5":
            self.action_space = spaces.Discrete(5)  # 0: Sell 100%, 1: Sell 50%, 2: Hold, 3: Buy 50%, 4: Buy 100%
        else:
            raise ValueError("Invalid ACTION_SPACE_TYPE in settings.py")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.initial_balance = settings.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.prev_portfolio_value = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.features[self.current_step].copy()

        current_price = self.df.loc[self.current_step, 'Close']
        unrealized_pnl_pct = (
            (current_price - self.entry_price) / self.entry_price
            if self.position > 0 and self.entry_price > 0
            else 0.0
        )
        cash_ratio = self.balance / self.initial_balance

        portfolio_features = np.array([
            1.0 if self.position > 0 else 0.0,
            unrealized_pnl_pct,
            cash_ratio,
            self.bars_in_trade / settings.MAX_BARS_IN_TRADE_NORM
        ])

        full_obs = np.hstack((obs, portfolio_features))
        return full_obs.astype(np.float32)

    def _execute_trade(self, action, current_price):
        slippage_pct = settings.SLIPPAGE_PCT
        step_reward_penalty = 0.0

        if settings.ACTION_SPACE_TYPE == "discrete_3":

            if action == 1:  # BUY ALL
                if self.balance > 0:
                    investment = self.balance * settings.CASH_RISK_FRACTION
                    actual_buy_price = current_price * (1 + slippage_pct)
                    self.position = investment / actual_buy_price
                    self.balance -= investment
                    self.entry_price = actual_buy_price
                    self.bars_in_trade = 1
                    step_reward_penalty -= settings.TRADE_PENALTY_FULL
                else:
                    step_reward_penalty -= settings.INVALID_ACTION_PENALTY

            elif action == 0:  # SELL ALL
                if self.position > 0:
                    actual_sell_price = current_price * (1 - slippage_pct)
                    self.balance += self.position * actual_sell_price
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.bars_in_trade = 0
                    step_reward_penalty -= settings.TRADE_PENALTY_FULL
                else:
                    step_reward_penalty -= settings.INVALID_ACTION_PENALTY

            elif action == 2:  # HOLD
                if self.position > 0:
                    self.bars_in_trade += 1

        elif settings.ACTION_SPACE_TYPE == "discrete_5":

            if action in [3, 4]:  # BUY 50% or 100%
                if self.balance > 0:
                    fraction = 1.0 if action == 4 else 0.5
                    investment = (self.balance * fraction) * settings.CASH_RISK_FRACTION
                    actual_buy_price = current_price * (1 + slippage_pct)
                    new_shares = investment / actual_buy_price

                    total_cost = (self.position * self.entry_price) + investment
                    self.position += new_shares
                    self.entry_price = total_cost / self.position if self.position > 0 else 0.0
                    self.balance -= investment
                    self.bars_in_trade = 1 if self.bars_in_trade == 0 else self.bars_in_trade

                    if action == 4:
                        step_reward_penalty -= settings.TRADE_PENALTY_FULL
                    else:
                        step_reward_penalty -= settings.TRADE_PENALTY_HALF
                else:
                    step_reward_penalty -= settings.INVALID_ACTION_PENALTY

            elif action in [0, 1]:  # SELL 100% or 50%
                if self.position > 0:
                    fraction = 1.0 if action == 0 else 0.5
                    shares_to_sell = self.position * fraction
                    actual_sell_price = current_price * (1 - slippage_pct)
                    self.balance += shares_to_sell * actual_sell_price
                    self.position -= shares_to_sell

                    if self.position <= settings.MIN_POSITION_THRESHOLD:
                        self.position = 0.0
                        self.entry_price = 0.0
                        self.bars_in_trade = 0

                    if action == 0:
                        step_reward_penalty -= settings.TRADE_PENALTY_FULL
                    else:
                        step_reward_penalty -= settings.TRADE_PENALTY_HALF
                else:
                    step_reward_penalty -= settings.INVALID_ACTION_PENALTY

            elif action == 2:  # HOLD
                if self.position > 0:
                    self.bars_in_trade += 1

        return step_reward_penalty

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        done = False
        info = {'step': self.current_step, 'action': action, 'price': current_price}

        reward = self._execute_trade(action, current_price)

        current_portfolio_value = self.balance + (self.position * current_price)
        step_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

        if settings.REWARD_STRATEGY == "absolute_asymmetric":
            if step_return > 0:
                reward += step_return * 100
            else:
                reward += step_return * 200

        elif settings.REWARD_STRATEGY == "pure_pnl":
            reward += step_return * 100

        else:
            raise ValueError("Invalid REWARD_STRATEGY in settings.py")

        self.prev_portfolio_value = current_portfolio_value

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        if current_portfolio_value <= self.initial_balance * 0.5:
            done = True
            reward -= settings.BANKRUPTCY_PENALTY

        info['portfolio_value'] = current_portfolio_value

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._next_observation()

        return obs, float(reward), done, False, info
