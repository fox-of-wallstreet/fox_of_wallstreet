# Environment and Feature Experiment Playbook

This guide explains how to run controlled experiments on environment behavior and feature sets in Fox of Wallstreet.

## 1. Core Rule

Change settings in one place only: config/settings.py.

All training and backtest behavior is driven by those values.

## 2. Fast Iteration Profile

For quick debugging runs, reduce compute first.

Suggested temporary values in config/settings.py:

- TOTAL_TIMESTEPS = 50000 to 150000
- N_STACK = 3
- OPTUNA_TRIALS = 5
- OPTUNA_EVAL_TIMESTEPS = 5000 to 10000

When strategy direction looks good, return to full training settings.

## 3. Environment Risk Controls

Primary environment knobs:

- CASH_RISK_FRACTION
- STOP_LOSS_PCT
- TAKE_PROFIT_PCT
- BANKRUPTCY_THRESHOLD_PCT
- BANKRUPTCY_PENALTY
- SLIPPAGE_PCT
- INVALID_ACTION_PENALTY
- REWARD_STRATEGY

Practical ranges:

- CASH_RISK_FRACTION
  - Conservative: 0.30 to 0.60
  - Aggressive: 0.80 to 1.00
- STOP_LOSS_PCT
  - Tight: 0.03 to 0.08
  - Relaxed: 0.12 to 0.25
- TAKE_PROFIT_PCT
  - Tight: 0.08 to 0.15
  - Relaxed: 0.25 to 0.50
- SLIPPAGE_PCT
  - Typical intraday stress test: 0.0005 to 0.002

Notes:

- In current implementation, stop loss and take profit are always active.
- To effectively disable SL or TP without code changes, set very large values such as 10.0.

## 4. Feature Set Controls

Feature composition is driven by:

- _BASE_FEATURES
- USE_NEWS_FEATURES
- USE_MACRO_FEATURES
- USE_TIME_FEATURES

Derived automatically:

- FEATURES_LIST
- EXPECTED_MARKET_FEATURES

Good pattern:

1. Start with base technical features only.
2. Add one feature block at a time (news, then macro, then time).
3. Compare against previous run metrics.

## 5. Suggested Experiment Matrix

Run these in order:

1. Baseline
   - USE_NEWS_FEATURES = False
   - USE_MACRO_FEATURES = False
   - USE_TIME_FEATURES = True
   - REWARD_STRATEGY = pure_pnl
2. Add news
   - USE_NEWS_FEATURES = True
3. Add macro
   - USE_MACRO_FEATURES = True
4. Reward shaping test
   - REWARD_STRATEGY = absolute_asymmetric
5. Risk relaxation test
   - Wider STOP_LOSS_PCT and TAKE_PROFIT_PCT
   - Lower CASH_RISK_FRACTION if drawdowns are unstable

Keep one change per run whenever possible.

## 6. Reproducibility and Compatibility

Because experiment names are timestamped, each train run writes to a unique artifact folder.

Backtest now resolves the latest compatible run and performs metadata compatibility checks.

If backtest fails compatibility:

- Either revert settings to match trained metadata.
- Or retrain with current settings and backtest again.

## 7. Safe Workflow for Every Experiment

1. Edit config/settings.py.
2. Refresh raw checkpoints as needed:
   - python scripts/data_engine.py
   - python scripts/news_engine.py
   - python scripts/macro_engine.py
3. Train:
   - python scripts/train.py
4. Backtest:
   - python scripts/backtest.py
5. Record key metrics from console and metadata.json.

Backtest now also writes a report bundle into the resolved training run folder:

- artifacts/<RUN_ID>/reports/figures
- artifacts/<RUN_ID>/reports/tables
- artifacts/<RUN_ID>/reports/summary

Key outputs:

- figures/actions_overlay.png
- figures/equity_vs_benchmark.png
- figures/drawdown_curve.png
- figures/trade_return_hist.png (only when complete cycles exist)
- tables/equity_timeseries.csv
- summary/report_index.json

Note: train/backtest checkpoints are now signature-aware and auto-rebuild when key settings or raw data mtimes change.

## 8. What to Track Per Run

Track these in your notes table (or use artifacts/experiment_journal.csv):

- Experiment name
- Feature flags and feature count
- Reward strategy
- Risk knobs (SL, TP, cash risk)
- Final portfolio value
- Total return
- Number of ledger events
- Trades per 100 bars
- Completed cycle count
- Average cycle return and cycle holding duration
- Qualitative behavior notes (overtrading, frequent stop-outs, long holds)

Journal note:

- Training auto-creates or updates a row in artifacts/experiment_journal.csv.
- Backtest updates the same run row with final value/return/events.

## 9. Common Failure Patterns

- Very negative returns with many events:
  - Often overtrading or too-aggressive risk fraction.
  - Try lower CASH_RISK_FRACTION and wider SL/TP.
- Noisy reward learning:
  - Try pure_pnl first, then move to absolute_asymmetric.
- Feature mismatch or shape errors:
  - Ensure every feature in settings is registered in processor FEATURE_REGISTRY.

## 10. Optional Next Improvement

If desired, add explicit booleans in settings:

- ENABLE_STOP_LOSS = True
- ENABLE_TAKE_PROFIT = True

Then gate forced-close logic in environment with those flags.

## 11. Artifact Folder Hygiene

Use scripts/artifact_manager.py for optional cleanup:

- List runs:
   - python scripts/artifact_manager.py list
- Preview empty folder cleanup:
   - python scripts/artifact_manager.py prune-empty --dry-run
- Delete empty folders:
   - python scripts/artifact_manager.py prune-empty
- Keep only latest N model runs (preview):
   - python scripts/artifact_manager.py keep-latest --keep 5 --dry-run
