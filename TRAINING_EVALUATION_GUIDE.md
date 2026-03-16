# Training Evaluation Guide
How to read training output, judge backtest results, and diagnose problems.

---

## 1. Reading the PPO Training Log

Every iteration SB3 prints a table. Here's what to watch and what the thresholds mean.

### `approx_kl` — How much the policy changed this update
The most important stability signal. KL divergence measures how different the new policy is from the old one.

| Value | Meaning |
|---|---|
| 0.01 – 0.05 | ✅ Healthy — policy updating steadily |
| 0.05 – 0.10 | ⚠️ Getting large — watch the trend |
| > 0.10 | ❌ Policy thrashing — LR probably too high |
| Rising each iteration | ❌ Diverging — will degrade, consider stopping |

If `approx_kl` starts low (0.01–0.03) and stays flat or declines → good run.
If it starts at 0.05 and climbs to 0.15+ over 50 iterations → bad run, stop early.

---

### `clip_fraction` — How often PPO had to clip gradient updates
PPO clips updates when the policy tries to change too aggressively. A high clip fraction means the LR is too large.

| Value | Meaning |
|---|---|
| 0.05 – 0.15 | ✅ Healthy |
| 0.15 – 0.25 | ⚠️ High — monitor |
| > 0.25 | ❌ Too high — LR too large or policy unstable |
| Declining over time | ✅ Good — policy stabilizing |
| Rising over time | ❌ Bad — policy destabilizing |

---

### `entropy_loss` — How much the agent is exploring
Entropy measures randomness in the policy. Negative values are normal (SB3 reports as negative loss).

| Value | Meaning |
|---|---|
| -1.5 to -0.8 | ✅ Healthy exploration early in training |
| Slowly declining toward -0.3 | ✅ Normal convergence — agent focusing |
| Suddenly collapsing to -0.3 early | ❌ Policy collapsed — committed to few actions |
| Staying flat at -1.5+ throughout | ⚠️ Not converging — agent still random |

---

### `explained_variance` — Value function quality
How well the critic (value network) predicts future returns. Higher = critic has learned the data.

| Value | Meaning |
|---|---|
| 0.3 – 0.7 | ✅ Healthy — critic learning but not memorized |
| 0.7 – 0.9 | ⚠️ Getting high — watch for overfitting |
| > 0.93 | ❌ Value function memorized training data |
| Declining at end of run | ✅ Good sign — policy still adjusting |
| Rising and flat at 0.94+ | ❌ Overfit — agent exploiting training patterns |

A high explained_variance with high approx_kl is the worst combination: the critic is confident but wrong on new data, and gives the policy bad gradient targets.

---

### `learning_rate` — Just confirm Optuna params loaded
Should match `settings.py` / Optuna best params. If it shows the default 3e-4 when you expected 2.37e-4, `USE_OPTUNA_BEST_PARAMS` wasn't True or the DB wasn't found.

---

### Quick diagnostic rule
At iteration ~20 (≈40k steps), check:
- `approx_kl` < 0.05 → proceed
- `clip_fraction` < 0.20 → proceed
- `entropy_loss` still negative and > -0.5 → proceed

If all three pass at iteration 20, the run is healthy. If any fail, stop and fix.

---

## 2. Hyperparameter Sweet Spots (TSLA 1h discrete_5)

These are empirically derived from our runs + Mogens' notes.

### Learning Rate
| Range | Behavior |
|---|---|
| 1e-5 – 1e-4 | Conservative, slow convergence, very stable |
| **1e-4 – 3e-4** | **✅ Recommended range** |
| 3e-4 – 1e-3 | Unstable — KL climbs over long runs |
| > 1e-3 | Diverges, policy collapse |

**Key insight:** Optuna at short eval (50k steps) tends to pick high LR because aggressive updates look good short-term. Clamp the search space upper bound to 3e-4 to prevent this.

### Gamma (Discount Factor)
Controls how far ahead the agent thinks. `1 / (1 - gamma)` = effective planning horizon in bars.

| Gamma | Horizon | Behavior |
|---|---|---|
| 0.90 – 0.94 | 10–17 bars | ✅ Good for intraday TSLA — responsive |
| 0.94 – 0.97 | 17–33 bars | ⚠️ Gets slower to react |
| 0.97 – 0.999 | 33–1000 bars | ❌ Too forward-looking for 1h data — degrades |
| **Search linearly**, not log-scale | | Log-scale biases toward 0.999 |

### Batch Size
| Size | Behavior |
|---|---|
| 32 | Noisy gradients, fast but unstable |
| **64–128** | **✅ Recommended** |
| 256 | Smooth but slow to adapt |

### `ent_coef` (Entropy Coefficient)
| Range | Behavior |
|---|---|
| 1e-4 – 1e-3 | ✅ Healthy — encourages exploration without chaos |
| > 0.05 | Too much randomness — agent won't converge |

---

## 3. Judging Backtest Results

### Step 1: Compare to Buy-and-Hold
This is the baseline. If the agent loses less than buy-and-hold in a down market, that's positive alpha. If it underperforms in an up market, it's not learning direction.

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/tsla_1h_prices.csv', parse_dates=['Date'])
test = df[(df['Date'] >= '2025-11-01') & (df['Date'] <= '2026-03-06')]
bh = (test['Close'].iloc[-1] / test['Close'].iloc[0] - 1) * 100
print(f'Buy-and-hold: {bh:.2f}%')
"
```

| Agent vs B&H | Verdict |
|---|---|
| Agent > B&H in up market | ✅ Capturing upside |
| Agent > B&H (less loss) in down market | ✅ Capital preservation |
| Agent < B&H in both | ❌ Not learning |
| Agent >> B&H in training period but not test | ❌ Overfit |

---

### Step 2: Analyze the Ledger

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('artifacts/<run_folder>/backtest_ledger.csv')
print(df['Action'].value_counts())
print('SL/TP triggered:', df['SL_TP_Triggered'].sum())
print('Total events:', len(df))
"
```

**What to look for:**

| Signal | Meaning |
|---|---|
| `SL_TP_Triggered = 0` | ✅ Agent managing its own exits — learned policy |
| `SL_TP_Triggered` > 30% of events | ⚠️ Agent relying on mechanical rules, not learned exits |
| Events > 50% of total bars | ⚠️ Overtrading — slippage drag will erode returns |
| SELL_100 very rare vs BUY_100 | ⚠️ Agent builds positions but reluctant to fully exit |
| BUY events >> SELL events | ⚠️ Net-long bias — fine in bull markets, risky in bear |

**Healthy ledger looks like:**
- Events on ~20–35% of bars
- SELL_100 and BUY_100 roughly balanced
- SL/TP triggered near 0

---

### Step 3: Check Trade Count vs Events
`Logged Events` counts every position change. If events = 385 in 580 bars, the agent is acting on 66% of bars — that's overtrading.

Overtrading causes:
1. Slippage accumulation on every entry/exit
2. Whipsaw losses on noisy bars
3. Missing sustained trends because the agent exits too early

**Target:** events on 20–40% of bars for hourly data.

---

## 4. Diagnosing Common Problems

### Problem: `approx_kl` climbing throughout training
**Cause:** LR too high.
**Fix:** Clamp Optuna LR upper bound to 3e-4, delete DB, re-optimize.

### Problem: Agent overtrades (>50% of bars)
**Cause:** No cost for frequent trading — slippage alone isn't enough.
**Options:**
1. Add explicit transaction cost penalty per valid trade in environment
2. Increase `gamma` slightly toward 0.95 (longer horizon = less reactive)
3. Reduce `CASH_RISK_FRACTION` so each trade is smaller and less rewarding

### Problem: Backtest worse than previous run despite better training metrics
**Cause:** Settings changed between runs (slippage, gamma) — not apples-to-apples.
**Fix:** Only change one variable at a time. Keep a log of what changed.

### Problem: SL/TP firing constantly during Optuna trials
**Not a problem** — during 50k optimization trials, the agent hasn't learned to exit. Mechanical SL/TP doing the work is expected. In the full 200k train run, the agent should learn its own exits (SL/TP triggered → 0).

### Problem: `explained_variance` > 0.93 throughout
**Cause:** Training data too small or too repetitive — value network memorizes the sequence.
**Fix:** More training data variety, or reduce `N_STACK` to reduce input redundancy.

### Problem: `entropy_loss` collapsed early (e.g. -0.3 by iteration 20)
**Cause:** Agent committed to Hold too early — common when invalid action penalties are too high, making non-trade actions always safer.
**Fix:** Lower `INVALID_ACTION_PENALTY` slightly, or increase `ent_coef` in Optuna search.

---

## 5. The Evaluation Checklist

After every train+backtest cycle:

```
Training:
[ ] approx_kl stayed below 0.05 after iteration 20
[ ] clip_fraction declining by end of run
[ ] entropy_loss still > -0.7 at end (still exploring)
[ ] explained_variance not stuck at > 0.93

Backtest:
[ ] Total return better than buy-and-hold
[ ] Events < 40% of total bars
[ ] SL/TP triggered < 10% of events
[ ] SELL actions not drastically lower than BUY actions
[ ] No single action dominating (e.g. 90% HOLD)

Comparison:
[ ] Only one variable changed from previous run
[ ] Same test period for fair comparison
[ ] Buy-and-hold baseline recalculated for the test period
```

---

## 6. The Correct Workflow

```
1. Change ONE setting
2. rm artifacts/optuna_study.db   (if env/search space changed)
3. python scripts/optimize.py
4. Check best params — LR < 3e-4? gamma in 0.90–0.94?
5. python scripts/train.py
6. Watch first 20 iterations — KL < 0.05?
7. python scripts/backtest.py
8. Compare ledger + return vs buy-and-hold
9. Log what changed and what the outcome was
10. Only then change the next variable
```

Never change slippage, gamma search range, and reward strategy at the same time — you won't know which caused the change in results.
