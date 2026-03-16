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
    # [cash_ratio, position_size, inventory_fraction, unrealized_pnl, last_action]
    NUM_PORTFOLIO_FEATURES = 5

    def __init__(self, df, features):
        super().__init__()

        self.df       = df.reset_index(drop=True)
        self.features = features.values if isinstance(features, pd.DataFrame) else features

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
        else:
            raise ValueError(
                f"❌ Invalid ACTION_SPACE_TYPE '{settings.ACTION_SPACE_TYPE}'. "
                f"Expected one of {settings.VALID_ACTION_SPACES}."
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step         = 0
        self.initial_balance      = settings.INITIAL_BALANCE
        self.balance              = self.initial_balance
        self.position             = 0.0
        self.entry_price          = 0.0
        self.bars_in_trade        = 0
        self.last_action          = 0
        self.prev_portfolio_value = self.initial_balance
        return self._next_observation(), {}

    # -------------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------------
    def _next_observation(self):
        obs           = self.features[self.current_step].copy()
        current_price = self.df.loc[self.current_step, "Close"]

        portfolio_value    = self.balance + (self.position * current_price)
        cash_ratio         = self.balance / (self.initial_balance + 1e-8)
        position_size      = (self.position * current_price) / (self.initial_balance + 1e-8)
        inventory_fraction = (self.position * current_price) / (portfolio_value + 1e-8)
        unrealized_pnl     = (
            (current_price - self.entry_price) / (self.entry_price + 1e-8)
            if self.position > 0 else 0.0
        )
        max_action = 4 if settings.ACTION_SPACE_TYPE == "discrete_5" else 2
        last_action_norm = self.last_action / max_action

        portfolio_features = np.array([
            cash_ratio,
            position_size,
            inventory_fraction,
            unrealized_pnl,
            last_action_norm,
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
        penalty = 0.0

        if settings.ACTION_SPACE_TYPE == "discrete_3":

            if action == 1:  # Buy All
                if self.balance > 0:
                    investment        = self.balance * settings.CASH_RISK_FRACTION
                    min_investment    = settings.INITIAL_BALANCE * settings.MIN_INVESTMENT_FRACTION
                    if investment < min_investment:
                        penalty -= settings.INVALID_ACTION_PENALTY  # Effectively out of cash
                    else:
                        actual_buy_price  = current_price * (1 + settings.SLIPPAGE_PCT)
                        self.position     = investment / actual_buy_price
                        self.balance     -= investment
                        self.entry_price  = actual_buy_price
                        self.bars_in_trade = 1
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
                if self.position > 0:
                    self.bars_in_trade += 1

        elif settings.ACTION_SPACE_TYPE == "discrete_5":

            if action in [3, 4]:  # Buy 50% or 100%
                if self.balance > 0:
                    fraction       = 1.0 if action == 4 else 0.5
                    investment     = (self.balance * fraction) * settings.CASH_RISK_FRACTION
                    min_investment = settings.INITIAL_BALANCE * settings.MIN_INVESTMENT_FRACTION
                    if investment < min_investment:
                        penalty -= settings.INVALID_ACTION_PENALTY  # Effectively out of cash
                    else:
                        actual_buy_price = current_price * (1 + settings.SLIPPAGE_PCT)
                        new_shares   = investment / actual_buy_price

                        total_cost       = (self.position * self.entry_price) + investment
                        self.position   += new_shares
                        self.entry_price = total_cost / self.position if self.position > 0 else 0.0
                        self.balance    -= investment
                        self.bars_in_trade = 1 if self.bars_in_trade == 0 else self.bars_in_trade
                        penalty -= 0.01 if action == 4 else 0.005
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY

            elif action in [0, 1]:  # Sell 100% or 50%
                if self.position > 0:
                    fraction         = 1.0 if action == 0 else 0.5
                    shares_to_sell   = self.position * fraction
                    actual_sell_price = current_price * (1 - settings.SLIPPAGE_PCT)
                    self.balance    += shares_to_sell * actual_sell_price
                    self.position   -= shares_to_sell

                    if action == 0:  # Full liquidation
                        self.entry_price   = 0.0
                        self.bars_in_trade = 0

                    penalty -= 0.01 if action == 0 else 0.005
                else:
                    penalty -= settings.INVALID_ACTION_PENALTY

            elif action == 2:  # Hold
                if self.position > 0:
                    self.bars_in_trade += 1

        return penalty

    # -------------------------------------------------------
    # STEP
    # -------------------------------------------------------
    def step(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        info          = {"step": self.current_step, "action": action, "price": current_price}

        # 1. Check stop loss / take profit BEFORE agent acts
        sl_tp_triggered, reward = self._check_sl_tp(current_price)

        # 2. Execute agent action only if SL/TP didn't already close the position
        if not sl_tp_triggered:
            reward += self._execute_trade(action, current_price)

        self.last_action = action

        # 3. Calculate portfolio value and step return
        current_portfolio_value = self.balance + (self.position * current_price)
        step_return = (
            (current_portfolio_value - self.prev_portfolio_value)
            / (self.prev_portfolio_value + 1e-8)
        )

        # 4. Apply reward strategy
        if settings.REWARD_STRATEGY == "absolute_asymmetric":
            reward += step_return * 100 if step_return > 0 else step_return * 200
        elif settings.REWARD_STRATEGY == "pure_pnl":
            reward += step_return * 100
        else:
            raise ValueError(
                f"❌ Invalid REWARD_STRATEGY '{settings.REWARD_STRATEGY}'. "
                f"Expected one of {settings.VALID_REWARD_STRATEGIES}."
            )

        self.prev_portfolio_value = current_portfolio_value

        # 5. Advance step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # 6. Bankruptcy check
        if current_portfolio_value <= self.initial_balance * settings.BANKRUPTCY_THRESHOLD_PCT:
            done    = True
            reward -= settings.BANKRUPTCY_PENALTY

        info["portfolio_value"] = current_portfolio_value
        info["sl_tp_triggered"] = sl_tp_triggered
        return self._next_observation(), float(reward), done, False, info
