import os
import sys
import random
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators


INITIAL_BALANCE = 10000.0
SLIPPAGE_PCT = 0.0005


def load_test_df():
    csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}. Run data_engine.py first.")

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    mask = (
        (df["Date"] >= pd.to_datetime(settings.TEST_START_DATE)) &
        (df["Date"] <= pd.to_datetime(settings.TEST_END_DATE))
    )
    test_df = df.loc[mask].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError("Test dataframe is empty.")

    test_df = add_technical_indicators(test_df)

    if test_df.empty:
        raise ValueError("Test dataframe is empty after preprocessing.")

    return test_df


def summarize_results(name, portfolio_values, trades):
    final_val = portfolio_values[-1]
    total_return = (final_val / INITIAL_BALANCE - 1) * 100

    peak = portfolio_values[0]
    max_drawdown = 0.0
    for v in portfolio_values:
        peak = max(peak, v)
        dd = (v / peak - 1) * 100
        max_drawdown = min(max_drawdown, dd)

    return {
        "Strategy": name,
        "Final Value": round(final_val, 2),
        "Return %": round(total_return, 2),
        "Trades": trades,
        "Max Drawdown %": round(max_drawdown, 2),
    }


def run_buy_and_hold(df):
    prices = df["Close"].values
    buy_price = prices[0] * (1 + SLIPPAGE_PCT)
    shares = INITIAL_BALANCE / buy_price

    portfolio_values = shares * prices
    return summarize_results("Buy & Hold", portfolio_values, trades=1)


def run_always_cash(df):
    portfolio_values = np.full(len(df), INITIAL_BALANCE)
    return summarize_results("Always Cash", portfolio_values, trades=0)


def run_random_policy(df):
    balance = INITIAL_BALANCE
    position = 0.0
    portfolio_values = []
    trades = 0

    prices = df["Close"].values

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        valid_actions = [0, 1, 2]
    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        valid_actions = [0, 1, 2, 3, 4]
    else:
        raise ValueError("Unsupported ACTION_SPACE_TYPE")

    for price in prices:
        action = random.choice(valid_actions)

        if settings.ACTION_SPACE_TYPE == "discrete_3":
            if action == 1 and balance > 0:  # buy all
                actual_buy_price = price * (1 + SLIPPAGE_PCT)
                position = balance / actual_buy_price
                balance = 0.0
                trades += 1

            elif action == 0 and position > 0:  # sell all
                actual_sell_price = price * (1 - SLIPPAGE_PCT)
                balance += position * actual_sell_price
                position = 0.0
                trades += 1

        elif settings.ACTION_SPACE_TYPE == "discrete_5":
            if action in [3, 4] and balance > 0:
                fraction = 0.5 if action == 3 else 1.0
                investment = balance * fraction * settings.CASH_RISK_FRACTION
                actual_buy_price = price * (1 + SLIPPAGE_PCT)
                new_shares = investment / actual_buy_price
                position += new_shares
                balance -= investment
                trades += 1

            elif action in [0, 1] and position > 0:
                fraction = 0.5 if action == 1 else 1.0
                shares_to_sell = position * fraction
                actual_sell_price = price * (1 - SLIPPAGE_PCT)
                balance += shares_to_sell * actual_sell_price
                position -= shares_to_sell
                if position < 1e-8:
                    position = 0.0
                trades += 1

        portfolio_values.append(balance + position * price)

    return summarize_results("Random Policy", portfolio_values, trades)


def run_rsi_macd_rule(df):
    balance = INITIAL_BALANCE
    position = 0.0
    portfolio_values = []
    trades = 0

    for _, row in df.iterrows():
        price = row["Close"]
        rsi = row["RSI"]
        macd_hist = row["MACD_Hist"]

        # simple long-only rule
        if position == 0 and rsi < 30 and macd_hist > 0:
            actual_buy_price = price * (1 + SLIPPAGE_PCT)
            position = balance / actual_buy_price
            balance = 0.0
            trades += 1

        elif position > 0 and rsi > 70 and macd_hist < 0:
            actual_sell_price = price * (1 - SLIPPAGE_PCT)
            balance += position * actual_sell_price
            position = 0.0
            trades += 1

        portfolio_values.append(balance + position * price)

    return summarize_results("RSI + MACD Rule", portfolio_values, trades)


def main():
    df = load_test_df()

    results = [
        run_buy_and_hold(df),
        run_always_cash(df),
        run_random_policy(df),
        run_rsi_macd_rule(df),
    ]

    results_df = pd.DataFrame(results)
    print("\nBaseline comparison:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
