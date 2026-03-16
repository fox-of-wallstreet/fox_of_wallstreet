Gemini search:

This is a classic "academia vs. reality" problem in quantitative finance. You are absolutely right to be skeptical of papers boasting massive Sharpe ratios in-sample; the transition to out-of-sample (OOS) backtests is where most complex reinforcement learning (RL) trading systems fall apart.

Based on recent literature (2022–2024) and practical quant consensus, here is the unvarnished truth about how these reward functions actually perform when the rubber meets the road.

### 1. The Core Question: Do Sharpe-Proxy Rewards Outperform Simple PnL OOS?

**Short answer:** No. For directional trading, simple, step-level PnL (or step-level log returns) consistently generalizes better out-of-sample than Sharpe-based reward functions. The extra complexity rarely transfers.

**Why it fails OOS:**

* **Gaming the Denominator:** Proximal Policy Optimization (PPO) is notoriously good at exploiting the math of a reward function rather than learning market dynamics. Because Sharpe involves dividing by volatility (variance of returns), a PPO agent will often learn that the easiest way to maximize the reward is to drive the denominator to zero. It does this by simply sitting in cash (holding) or making micro-trades in low-volatility regimes, artificially inflating the proxy without actually capturing alpha.
* **Hyperparameter Fragility:** Step-level Sharpe (often implemented as a *Differential Sharpe Ratio*) requires maintaining moving averages of expected returns and variance. This introduces extreme sensitivity to hyperparameters (like the decay rate). What looks like a robust policy in-sample is usually just the agent overfitting to the specific volatility regimes of the training data.
* **The "Simpler is Better" Consensus:** Recent practical literature (and industry blogs like IBKR Quant) note that simple PnL—or even just the binary sign of the PnL (e.g., +1 for profitable step, -1 for loss)—often results in faster convergence and more robust OOS performance.

### 2. Step-Level PnL vs. Episode-Level (Realized) Reward

You must use **step-level (unrealized) PnL** if you are using PPO.

* **The Reward Sparsity Problem:** If you only grant a reward when a trade is closed (realized PnL), you introduce severe reward sparsity. PPO relies heavily on Generalized Advantage Estimation (GAE) to train its critic network (the value function). If the agent takes 50 steps to close a trade, the critic has no idea which of those 50 steps were "good" or "bad."
* **The Result of Sparsity:** Papers from 2023 and 2024 tackling reward sparsity in RL note that without dense signals, PPO agents either collapse into a single action (e.g., permanently holding) or thrash randomly.
* **The Fix:** Step-level unrealized PnL densifies the signal. The agent gets immediate feedback on the trajectory of its open position, allowing the value network to map current market states to expected future returns accurately.

### 3. Sharpe-Ratio Rewards: Per-Step vs. Rolling Window

If you *do* attempt to use risk-adjusted rewards, the implementation matters, but both have fatal flaws for directional trading:

* **Rolling Window Sharpe:** Calculating Sharpe over a trailing window (e.g., 20 steps) and feeding it as a reward fundamentally violates the Markov Decision Process (MDP) assumption. The reward the agent receives for a specific action today is contaminated by an action it took 19 steps ago. This "smeared" feedback confuses PPO's advantage estimations.
* **Per-Step (Differential) Sharpe:** This mathematically approximates the gradient of the Sharpe ratio at each step. While it keeps the MDP mathematically sound, it is highly non-convex. As discussed in recent algorithmic trading forums and papers, optimizing a non-convex reward often traps the PPO agent in local optima, making it highly unstable across different market regimes (e.g., failing spectacularly when transitioning from a 2022 bear market to a 2023/2024 bull market).

### 4. The Action Space Caveat (`discrete` vs `continuous`)

When reading papers that claim Sharpe-based rewards work, **check their action space**. The success of a reward function is highly dependent on what the agent is actually allowed to do:

* **Continuous (Portfolio Allocation):** If the action space is continuous (e.g., outputting a vector of portfolio weights between 0 and 1), Sharpe-based rewards *do* have merit. The agent is managing a basket of assets and can continuously hedge to smooth out the equity curve.
* **`discrete_3` (Buy, Hold, Sell) or `discrete_5`:** In discrete directional trading, Sharpe-proxy rewards almost always underperform simple PnL OOS. In a `discrete_3` space, the agent cannot finely tune its risk exposure; it is either in or out. Penalizing volatility here just teaches the agent to avoid the market. For `discrete_5` (e.g., Strong Buy, Buy, Hold, Sell, Strong Sell), Sharpe rewards often induce excessive churn as the agent rapidly flips between states trying to game the rolling variance.

### Summary Recommendation

Stick to **step-level unrealized PnL**. If you want to control risk, don't do it in the reward function. Instead, control risk through **action masking** (preventing the agent from taking risky trades), **position sizing logic** outside the RL loop, or by adding a simple, symmetric transaction cost penalty to the step reward to discourage churn.


# Perplexity search:

Here's a detailed synthesis of what the literature actually says, split into your specific questions.

***

## Step-Level PnL vs. Realized/Episode-Level Reward

The tradeoff is well-characterized. **Realized PnL** (reward only on closed trades) is theoretically correct — it avoids the agent over-optimizing unrealized positions — but creates severe **reward sparsity** because buy/sell actions are rare relative to hold steps. This is widely cited as a core failure mode: "Naively applying reward-hungry RL algorithms will fail" under sparse signals. [dennybritz](https://dennybritz.com/posts/wildml/learning-to-trade-with-reinforcement-learning/)

**Unrealized/step-level PnL** (the running `position × Δprice` per step) provides dense feedback but introduces a **short-horizon bias** — under a discount factor γ < 1, the agent is incentivized to take profit early even when the optimal policy would hold. This makes it particularly problematic with `γ < 0.99`. [dennybritz](https://dennybritz.com/posts/wildml/learning-to-trade-with-reinforcement-learning/)

The practical consensus from FinRL-Meta (NeurIPS 2022 benchmark) is to use **portfolio value change** \( r_t = v_t - v_{t-1} \) at each step, which is effectively step-level unrealized PnL — it's the default across FinRL's three reward types along with log-return and Sharpe. [papers.neurips](https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf)

***

## Sharpe-Ratio Reward: Per-Step vs. Rolling Window

The most directly relevant paper is **Rodinos et al. (Aristotle University / Speedlab AG)** — a PPO+LSTM study on 14 cryptocurrency pairs (2017–2022) that explicitly compares three reward schemes: [cidl.csd.auth](https://cidl.csd.auth.gr/resources/conference_pdfs/Paper%20-%20A_Sharpe_Ratio_Based_Reward_Scheme_in_Deep_Reinforcement_.pdf)

| Reward scheme | Monthly Ann. Sharpe | Hourly Ann. Sharpe |
|---|---|---|
| PnL-only (baseline) | 1.462 ± 0.055 | 2.374 ± 0.079 |
| PnL + Sharpe additive | 1.499 ± 0.060 | 2.484 ± 0.090 |
| **Proposed (dynamic Sharpe direction)** | **1.617 ± 0.056** | **2.641 ± 0.083** |

Their key design: they compute an **expanding-window Sharpe approximation within the episode**, starting at step `m/2` and growing to `m`. Critically, their "proposed" scheme is *not* simply PnL + Sharpe — they **compare consecutive Sharpe values** and add the term if Sharpe improved, *subtract* it if it deteriorated. This directional shaping provides the agent a signal about risk trajectory, not just instantaneous return. [cidl.csd.auth](https://cidl.csd.auth.gr/resources/conference_pdfs/Paper%20-%20A_Sharpe_Ratio_Based_Reward_Scheme_in_Deep_Reinforcement_.pdf)

A 2026 PMC paper (Charkhestani et al.) takes a different approach: they compute rewards as **step-level realized portfolio return**, then use the **episode-level Sharpe to normalize and scale those per-step rewards** after the episode. This aligns training objective with risk-adjusted performance and reportedly improves learning stability. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12909882/)

***

## Does the Complexity Actually Transfer Out-of-Sample?

**The honest answer is: weakly, and conditionally.** Key caveats from the literature:

- The Rodinos et al. results are on a **2021–2022 crypto test set** — a period including both the 2021 bull run and the 2022 bear market. That's one OOS regime, and the improvement in annualized Sharpe (~10%) is real but modest. [cidl.csd.auth](https://cidl.csd.auth.gr/resources/conference_pdfs/Paper%20-%20A_Sharpe_Ratio_Based_Reward_Scheme_in_Deep_Reinforcement_.pdf)
- A 2024/2025 thesis (IJbema, LIACS) studying PPO/DDPG/SAC on stock portfolio allocation found that **off-policy methods (DDPG, SAC) generalize better OOS** than PPO, and critically notes that none of the DRL models consistently beat a simple Buy-and-Hold benchmark across diverse OOS assets. [theses.liacs](https://theses.liacs.nl/pdf/2024-2025-IJbemaJJonathan.pdf)
- A comparative study (ScienceXcel 2025) found PPO achieves the highest cumulative return and Sharpe (1.75 vs DDPG 1.60 vs DQN 1.45), but this is **in-sample/limited backtest**, with a discrete action space and no multi-regime OOS test. [sciencexcel](https://www.sciencexcel.com/articles/DtDBMCSPcpMLzp6QD5dApsOuAZ7nd2p4Zd3l7uKE.pdf)
- The ACM survey (Sun et al., 2023, cited 145×) notes that Sharpe-ratio rewards are **common in portfolio management papers** but the field broadly struggles with the simulation-to-reality gap due to non-stationarity. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3582560)

The practical risk with Sharpe-proxy reward shaping: the rolling volatility estimate becomes **very noisy at short episode lengths** (e.g., 100 steps as used by Rodinos et al.), which can cause the Sharpe term to dominate the PnL signal mid-episode. The scale factor `α = 0.01` in their formula is doing heavy lifting to prevent this. [cidl.csd.auth](https://cidl.csd.auth.gr/resources/conference_pdfs/Paper%20-%20A_Sharpe_Ratio_Based_Reward_Scheme_in_Deep_Reinforcement_.pdf)

***

## Action Space Caveat for `discrete_5`

This is a real concern. **Virtually all papers** in this space use `discrete_3` (long / neutral / short, or buy / hold / sell) — the Rodinos et al. paper uses exactly `{long, neutral, short}`, and the FinRL ecosystem defaults to `{buy, hold, sell}` or continuous allocation fractions. [arxiv](https://www.arxiv.org/pdf/2504.02281.pdf)

**No papers in the 2022–2024 window directly test `discrete_5`** (e.g., strong buy / buy / hold / sell / strong sell). The implicit assumption is that `discrete_5` introduces a **policy entropy problem** — with 5 actions but only 3 meaningfully different position states, the agent needs more samples to learn which "strength" is worth taking. Sharpe-shaping that works well in `discrete_3` may behave differently here because the Sharpe term may inadvertently suppress the "strong" actions early in training when variance is high.

***

## Practical Takeaway for Your Setup

Given what the evidence supports:

- **Step-level unrealized PnL is the safest baseline** — it's the most generalizable OOS because it doesn't add variance to the reward signal during non-stationary regimes. [papers.neurips](https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf)
- If you add a Sharpe term, use the **directional approach** (Rodinos-style: reward if Sharpe improves, penalize if it falls) rather than just adding the raw Sharpe value, which can create unstable gradients. [reddit](https://www.reddit.com/r/algotrading/comments/8705zw/sharpe_ratio_as_a_reward_function_for/)
- For `discrete_5`, be careful with scale factor `α` — you'll likely need to tune it separately, as papers calibrated on `discrete_3` environments won't give you a safe default.
- Consider **episode-level Sharpe normalization** of step rewards (Charkhestani 2026 approach) as a lower-risk alternative to mid-episode Sharpe shaping — it's less invasive and preserves the dense step-level signal. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12909882/)
