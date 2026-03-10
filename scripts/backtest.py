'''
Missing module docstring.
'''

import os
import os.path
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Ensure Python can find your folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.processor import add_technical_indicators, prepare_features
from core.environment import TradingEnv

def run_backtest():
    '''
    Missing function or method docstring.
    '''
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"🧪 STARTING FINAL EXAM: {settings.EXPERIMENT_NAME}")

    csv_path = f"data/{settings.SYMBOL.lower()}_{settings.TIMEFRAME}_hybrid.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Cannot find {csv_path}. Run data_engine.py first!")

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Slice to Testing Dates
    mask = (df['Date'] >= pd.to_datetime(settings.TEST_START_DATE, utc=True)) & (df['Date'] <= pd.to_datetime(settings.TEST_END_DATE, utc=True))
    test_df = df.loc[mask].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError(f"❌ Test dataframe is empty! Check your TEST_START_DATE ({settings.TEST_START_DATE}) and TEST_END_DATE.")

    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"📅 Testing Data: {len(test_df)} rows loaded.")

    # Process features
    test_df = add_technical_indicators(test_df)
    features_list = [
        'Log_Return', 'Volume_Z_Score', 'RSI', 'MACD_Hist', 'BB_Pct', 'ATR_Pct',
        'QQQ_Ret', 'ARKK_Ret', 'Rel_Strength_QQQ', 'VIX_Level', 'TNX_Level',
        'Sentiment_EMA', 'News_Intensity', 'Sin_Time', 'Cos_Time', 'Mins_to_Close'
    ]

    # Load the Scaler from the Vault (is_training=False)
    scaled_features = prepare_features(test_df, features_list, is_training=False)

    # Build the Environment
    base_env = TradingEnv(df=test_df, features=scaled_features)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecFrameStack(vec_env, n_stack=5)

    # Load the Brain
    model_path = settings.MODEL_PATH
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"🧠 Loading trained model from {model_path}.zip")
    if os.path.isfile(f"{model_path}.zip"):
        model = PPO.load(model_path, env=env)
    else:
        print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f'Cannot PPO.load({model_path}): No such file or directory.')
        return

    obs = env.reset()
    done = False
    trade_history = []

    # 🟢 Dynamic Mapping based on Control Room
    if settings.ACTION_SPACE_TYPE == "discrete_3":
        action_map = {0: "🔴 SELL ALL", 1: "🟢 BUY ALL"}
    else:
        action_map = {0: "🔴 STRONG SELL (100%)", 1: "🔴 LIGHT SELL (50%)", 3: "🟢 LIGHT BUY (50%)", 4: "🟢 STRONG BUY (100%)"}

    prev_position = 0.0

    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', "📈 Simulating live trading...")

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

        step_info = info[0]
        actual_action = step_info['action']

        # Pull actual internal position to verify it wasn't a rejected trade
        current_position = env.get_attr('position')[0]

        # Log if position changed AND it wasn't a "HOLD" (Action 2)
        if current_position != prev_position and actual_action != 2:
            trade_history.append({
                "Date": test_df.loc[step_info['step'], 'Date'],
                "Action": action_map.get(actual_action, "UNKNOWN"),
                "Price": round(step_info['price'], 2),
                "Portfolio_Value": round(step_info['portfolio_value'], 2)
            })
            prev_position = current_position

    initial_val = base_env.initial_balance
    final_val = step_info['portfolio_value']
    total_return = ((final_val - initial_val) / initial_val) * 100

    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', "="*50)
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"🏆 BACKTEST RESULTS: {settings.EXPERIMENT_NAME} 🏆")
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"Final Portfolio Value: ${final_val:.2f}")
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"Total Return: {total_return:.2f}%")
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"Total Real Transactions: {len(trade_history)}")
    print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', "="*50)

    if trade_history:
        df_trades = pd.DataFrame(trade_history)
        ledger_path = os.path.join(settings.ARTIFACT_DIR, "backtest_ledger.csv")
        df_trades.to_csv(ledger_path, index=False)
        print(os.path.basename(__file__) + '(' + str(sys._getframe(0).f_lineno) + '):', f"\n💾 Full Ledger saved to Vault: {ledger_path}")

if __name__ == "__main__":
    run_backtest()
