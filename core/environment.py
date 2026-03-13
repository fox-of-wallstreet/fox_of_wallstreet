"""
Custom Gymnasium Trading Environment.
All risk parameters, action space, and reward strategy
are driven exclusively by config/settings.py.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config import settings



class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Internal portfolio state features appended to every observation:
    # [position_status, unrealized_pnl_pct, cash_ratio, bars_in_trade_normalized]
    NUM_PORTFOLIO_FEATURES = 4

    def __init__(self, df, features):
        super().__init__()

        self.df       = df.reset_index(drop=True)
        self.features = features.values if isinstance(features, pd.DataFrame) else features

<<<<<<< HEAD
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
=======
        # -------------------------------------------------------
        # Shape guard — now works because settings defines it
        # -------------------------------------------------------
        if self.features.shape[1] != settings.EXPECTED_MARKET_FEATURES:
            raise ValueError(
                f"🚨 DATA SHAPE MISMATCH: Expected {settings.EXPECTED_MARKET_FEATURES} "
                f"market features, got {self.features.shape[1]}. "
                f"Check FEATURES_LIST in settings.py matches processor output."
            )

        self.num_features = self.features.shape[1] + self.NUM_PORTFOLIO_FEATURES

        # -------------------------------------------------------
        # Action space — driven by settings
        # discrete_3: 0=Sell All, 1=Buy All,  2=Hold
        # discrete_5: 0=Sell100, 1=Sell50, 2=Hold, 3=Buy50, 4=Buy100
        # -------------------------------------------------------
        if settings.ACTION_SPACE_TYPE == "discrete_3":
            self.action_space = spaces.Discrete(3)
        elif settings.ACTION_SPACE_TYPE == "discrete_5":
            self.action_space = spaces.Discrete(5)
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        else:
            raise ValueError(
                f"❌ Invalid ACTION_SPACE_TYPE '{settings.ACTION_SPACE_TYPE}'. "
                f"Expected one of {settings.VALID_ACTION_SPACES}."
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
<<<<<<< HEAD
        self.current_step = 0
        self.initial_balance = settings.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
=======
        self.current_step        = 0
        self.initial_balance     = settings.INITIAL_BALANCE
        self.balance             = self.initial_balance
        self.position            = 0.0
        self.entry_price         = 0.0
        self.bars_in_trade       = 0
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        self.prev_portfolio_value = self.initial_balance
        return self._next_observation(), {}

    # -------------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------------
    def _next_observation(self):
<<<<<<< HEAD
        obs = self.features[self.current_step].copy()

        current_price = self.df.loc[self.current_step, 'Close']
        unrealized_pnl_pct = (
            (current_price - self.entry_price) / self.entry_price
            if self.position > 0 and self.entry_price > 0
            else 0.0
=======
        obs           = self.features[self.current_step].copy()
        current_price = self.df.loc[self.current_step, "Close"]

        unrealized_pnl_pct = (
            (current_price - self.entry_price) / (self.entry_price + 1e-8)
            if self.position > 0 else 0.0
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        )
        cash_ratio = self.balance / self.initial_balance

        portfolio_features = np.array([
            1.0 if self.position > 0 else 0.0,
            unrealized_pnl_pct,
            cash_ratio,
<<<<<<< HEAD
            self.bars_in_trade / settings.MAX_BARS_IN_TRADE_NORM
=======
            min(self.bars_in_trade / settings.MAX_BARS_NORMALIZATION, 1.0),
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        ])

        return np.hstack((obs, portfolio_features)).astype(np.float32)

    # -------------------------------------------------------
    # STOP LOSS / TAKE PROFIT
    # Market-driven forced close — happens before the agent acts.
    # Returns (was_triggered, reward_penalty)
    # -------------------------------------------------------
    def _check_sl_tp(self, current_price):
        if self.position <= 0:
            return False, 0.0

        unrealized_pnl_pct = (current_price - self.entry_price) / (self.entry_price + 1e-8)

        hit_stop_loss   = unrealized_pnl_pct <= -settings.STOP_LOSS_PCT
        hit_take_profit = unrealized_pnl_pct >=  settings.TAKE_PROFIT_PCT

        if hit_stop_loss or hit_take_profit:
            actual_sell_price = current_price * (1 - settings.SLIPPAGE_PCT)
            self.balance     += self.position * actual_sell_price
            self.position     = 0.0
            self.entry_price  = 0.0
            self.bars_in_trade = 0

            label = "🛑 STOP LOSS" if hit_stop_loss else "🎯 TAKE PROFIT"
            print(f"{label} triggered at {current_price:.2f} | PnL: {unrealized_pnl_pct:.2%}")
            return True, 0.0  # No extra penalty — this is expected market behaviour

        return False, 0.0

    # -------------------------------------------------------
    # TRADE EXECUTION
    # -------------------------------------------------------
    def _execute_trade(self, action, current_price):
<<<<<<< HEAD
        slippage_pct = settings.SLIPPAGE_PCT
        step_reward_penalty = 0.0

        if settings.ACTION_SPACE_TYPE == "discrete_3":

            if action == 1:  # BUY ALL
=======
        penalty = 0.0

        if settings.ACTION_SPACE_TYPE == "discrete_3":

            if action == 1:  # Buy All
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
                if self.balance > 0:
                    investment        = self.balance * settings.CASH_RISK_FRACTION
                    actual_buy_price  = current_price * (1 + settings.SLIPPAGE_PCT)
                    self.position     = investment / actual_buy_price
                    self.balance     -= investment
                    self.entry_price  = actual_buy_price
                    self.bars_in_trade = 1
<<<<<<< HEAD
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
=======
                    penalty           -= 0.01
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY  # Buy with no cash

            elif action == 0:  # Sell All
                if self.position > 0:
                    actual_sell_price  = current_price * (1 - settings.SLIPPAGE_PCT)
                    self.balance      += self.position * actual_sell_price
                    self.position      = 0.0
                    self.entry_price   = 0.0
                    self.bars_in_trade = 0
                    penalty           -= 0.01
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY  # Sell with no shares

            elif action == 2:  # Hold
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
                if self.position > 0:
                    self.bars_in_trade += 1

        elif settings.ACTION_SPACE_TYPE == "discrete_5":

<<<<<<< HEAD
            if action in [3, 4]:  # BUY 50% or 100%
=======
            if action in [3, 4]:  # Buy 50% or 100%
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
                if self.balance > 0:
                    fraction     = 1.0 if action == 4 else 0.5
                    investment   = (self.balance * fraction) * settings.CASH_RISK_FRACTION
                    actual_buy_price = current_price * (1 + settings.SLIPPAGE_PCT)
                    new_shares   = investment / actual_buy_price

<<<<<<< HEAD
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
=======
                    total_cost       = (self.position * self.entry_price) + investment
                    self.position   += new_shares
                    self.entry_price = total_cost / self.position if self.position > 0 else 0.0
                    self.balance    -= investment
                    self.bars_in_trade = 1 if self.bars_in_trade == 0 else self.bars_in_trade
                    penalty -= 0.01 if action == 4 else 0.005
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY

            elif action in [0, 1]:  # Sell 100% or 50%
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
                if self.position > 0:
                    fraction         = 1.0 if action == 0 else 0.5
                    shares_to_sell   = self.position * fraction
                    actual_sell_price = current_price * (1 - settings.SLIPPAGE_PCT)
                    self.balance    += shares_to_sell * actual_sell_price
                    self.position   -= shares_to_sell

<<<<<<< HEAD
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
=======
                    if action == 0:  # Full liquidation
                        self.entry_price   = 0.0
                        self.bars_in_trade = 0

                    penalty -= 0.01 if action == 0 else 0.005
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY

            elif action == 2:  # Hold
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
                if self.position > 0:
                    self.bars_in_trade += 1

        return penalty

    # -------------------------------------------------------
    # STEP
    # -------------------------------------------------------
    def step(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        info          = {"step": self.current_step, "action": action, "price": current_price}

<<<<<<< HEAD
        reward = self._execute_trade(action, current_price)

=======
        # 1. Check stop loss / take profit BEFORE agent acts
        sl_tp_triggered, reward = self._check_sl_tp(current_price)

        # 2. Execute agent action only if SL/TP didn't already close the position
        if not sl_tp_triggered:
            reward += self._execute_trade(action, current_price)

        # 3. Calculate portfolio value and step return
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        current_portfolio_value = self.balance + (self.position * current_price)
        step_return = (
            (current_portfolio_value - self.prev_portfolio_value)
            / (self.prev_portfolio_value + 1e-8)
        )

<<<<<<< HEAD
        if settings.REWARD_STRATEGY == "absolute_asymmetric":
            if step_return > 0:
                reward += step_return * 100
            else:
                reward += step_return * 200

=======
        # 4. Apply reward strategy
        if settings.REWARD_STRATEGY == "absolute_asymmetric":
            reward += step_return * 100 if step_return > 0 else step_return * 200
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        elif settings.REWARD_STRATEGY == "pure_pnl":
            reward += step_return * 100
        else:
            raise ValueError(
                f"❌ Invalid REWARD_STRATEGY '{settings.REWARD_STRATEGY}'. "
                f"Expected one of {settings.VALID_REWARD_STRATEGIES}."
            )

        self.prev_portfolio_value = current_portfolio_value

<<<<<<< HEAD
=======
        # 5. Advance step
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

<<<<<<< HEAD
        if current_portfolio_value <= self.initial_balance * 0.5:
            done = True
            reward -= settings.BANKRUPTCY_PENALTY

        info['portfolio_value'] = current_portfolio_value

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._next_observation()

        return obs, float(reward), done, False, info
=======
        # 6. Bankruptcy check
        if current_portfolio_value <= self.initial_balance * settings.BANKRUPTCY_THRESHOLD_PCT:
            done    = True
            reward -= settings.BANKRUPTCY_PENALTY

        info["portfolio_value"] = current_portfolio_value
        info["sl_tp_triggered"] = sl_tp_triggered
        return self._next_observation(), float(reward), done, False, info
>>>>>>> 7daa4dace0a1ce7ecaa224d0aedcd641f8074485
