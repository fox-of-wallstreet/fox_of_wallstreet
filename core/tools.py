'''
Place for global things.
'''
import os
import sys
from config import settings

def fnline():
    '''
    For logging and tracing.
    Returns current filename and line number.
    E.g.: backtest.py(144))
    '''
    return os.path.basename(sys.argv[0]) + '(' + str(sys._getframe(1).f_lineno) + '):'

def get_features_list():
    '''
    Return the list of features.
    If time frame is 1 hour return some more.
    '''
    base_features = [
        'Log_Return',
        'Volume_Z_Score',
        'RSI',
        'MACD_Hist',
        'BB_Pct',
        'ATR_Pct',
        'Realized_Vol_Short',
        'Realized_Vol_Long',
        'Vol_Regime',
        'Dist_MA_Fast',
        'Dist_MA_Slow',
        'QQQ_Ret',
        'ARKK_Ret',
        'Rel_Strength_QQQ',
        'VIX_Z',
        'TNX_Z',
        'Sentiment_EMA',
        'News_Intensity'
    ]

    if settings.TIMEFRAME == "1h":
        return base_features + ['Sin_Time', 'Cos_Time', 'Mins_to_Close']
    elif settings.TIMEFRAME == "1d":
        return base_features
    else:
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

def get_stack_size():
    if settings.TIMEFRAME == "1h":
        return 5
    elif settings.TIMEFRAME == "1d":
        return 10
    else:
        raise ValueError(f"Unsupported TIMEFRAME: {settings.TIMEFRAME}")

# function to build the name of the artifacts folder specifing the model/agent setup
def build_experiment_name():
    symbol = settings.SYMBOL
    tf = settings.TIMEFRAME

    if settings.ACTION_SPACE_TYPE == "discrete_3":
        action = "d3"
    elif settings.ACTION_SPACE_TYPE == "discrete_5":
        action = "d5"
    else:
        raise ValueError(f"Unsupported ACTION_SPACE_TYPE: {settings.ACTION_SPACE_TYPE}")

    if settings.REWARD_STRATEGY == "absolute_asymmetric":
        reward = "asym"
    elif settings.REWARD_STRATEGY == "pure_pnl":
        reward = "pnl"
    else:
        raise ValueError(f"Unsupported REWARD_STRATEGY: {settings.REWARD_STRATEGY}")

    timesteps = settings.TOTAL_TIMESTEPS
    if timesteps >= 1_000_000:
        t_tag = f"t{timesteps//1_000_000}M"
    else:
        t_tag = f"t{timesteps//1000}k"

    seed_tag = f"s{settings.RANDOM_SEED}"
    version_tag = f"v{settings.EXPERIMENT_VERSION}"

    return f"ppo_{symbol}_{tf}_{action}_{reward}_{t_tag}_{seed_tag}_{version_tag}"
