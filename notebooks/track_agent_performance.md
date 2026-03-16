### This is a protocoll to track the performance of different agents ###


### Default Settings

#### ENVIRONMENT DEFAULTS
INITIAL_BALANCE = 10000.0
SLIPPAGE_PCT = 0.0005
TRADE_PENALTY_FULL = 0.01
TRADE_PENALTY_HALF = 0.005
INVALID_ACTION_PENALTY = 0.05
BANKRUPTCY_PENALTY = 10.0
MIN_POSITION_THRESHOLD = 1e-8
MAX_BARS_IN_TRADE_NORM = 100.0

##### PPO DEFAULTS
PPO_LEARNING_RATE = 0.0003
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
PPO_ENT_COEF = 0.01



### Start experimenting:

## 1. Current Reference Agent
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_5
Reward_func: assymetric
Training timesteps = 1.000.000
Enviornment parameters: Defaults
PPO Agent parameters: Defaults

Results:
Total Return: -8.93%
Total Real Transactions: 170


## 2. PPO parameters optimized
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_5
Reward_func: assymetric
Training timesteps = 1.000.000
Enviornment parameters: Defaults
PPO Agent parameters: optimized
      PPO_LEARNING_RATE = 0.0007
      PPO_BATCH_SIZE = 128
      PPO_GAMMA = 0.91
      PPO_ENT_COEF = 0.0012


Results:
Total Return: 6.27%
Total Real Transactions: 188


## 3. Panelty increased
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_5
Reward_func: assymetric
Training timesteps = 1.000.000
Enviornment parameters: penelty enhanced
      TRADE_PENALTY_FULL = 0.05 #0.01
      TRADE_PENALTY_HALF = 0.02 #0.005
PPO Agent parameters: optimized

Results:
Total Return: -21.51%
Total Real Transactions: 187


# 4. Action discrete 3
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_3
Reward_func: assymetric
Training timesteps = 1.000.000
Enviornment parameters: default
PPO Agent parameters: optimized
      PPO_LEARNING_RATE = 0.0007
      PPO_BATCH_SIZE = 128
      PPO_GAMMA = 0.91
      PPO_ENT_COEF = 0.0012

Results:
Total Return: -5.85
Total Real Transactions: 50


# 5. Action discrete 3 V2
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_3
Reward_func: assymetric
Training timesteps = 1.000.000
Enviornment parameters: default
PPO Agent parameters: optimized
      PPO_LEARNING_RATE = 5.2e-05 #3e-4
      PPO_BATCH_SIZE = 256 #64
      PPO_GAMMA = 0.9 #0.99
      PPO_ENT_COEF = 0.046 #0.01

Results:
Total Return: -99.04%
Total Real Transactions: 18


# 5. Action discrete 3 V3
----------------------------
Parameters:
Symbol: TSLA
Timeframe: 1h
Action Space = discrete_3
Reward_func: assymetric
#### Training timesteps = 100.000
Enviornment parameters: default
PPO Agent parameters: optimized
      PPO_LEARNING_RATE = 5.2e-05 #3e-4
      PPO_BATCH_SIZE = 64
      PPO_GAMMA = 0.965
      PPO_ENT_COEF = 0.0007 #0.01

Results:
