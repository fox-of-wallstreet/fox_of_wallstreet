# Final Baseline Feature Set

This baseline keeps a compact set of features that are normalized, relatively stable, and meaningful for a PPO-style trading agent on hourly and daily large-cap stock data.  The goal is to tell the agent about price change, momentum, anchored fair value, volatility, cross-asset context, exogenous news flow, forecast context, and current portfolio state without relying on raw price levels. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)

## Core market

### `Log_Return`
- Category: core market.
- Source: current and previous `Close`.
- Engineering: yes.
- Formula: `log(Close_t / Close_{t-1})`.
- Notes: tells the agent the latest directional price change in a more stable form than raw price level. [tandfonline](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2490818)

### `Volume_Z_Score`
- Category: core market.
- Source: `Volume`.
- Engineering: yes.
- Formula: `(Volume_t - rolling_mean_20) / rolling_std_20`.
- Notes: tells the agent whether the current move is happening on unusually high or low participation. [trendspider](https://trendspider.com/learning-center/anchored-vwap-trading-strategies/)

### `RSI`
- Category: core market.
- Source: `Close`.
- Engineering: yes.
- Method: 14-period exponentially smoothed relative strength index.
- Notes: tells the agent whether momentum is strong, weak, extended, or recovering. [trendspider](https://trendspider.com/learning-center/anchored-vwap-trading-strategies/)

### `MACD_Hist`
- Category: core market.
- Source: `Close`.
- Engineering: yes.
- Formula: `(EMA_12 - EMA_26) - EMA_9_of_MACD`.
- Notes: tells the agent whether momentum is accelerating or fading, not just whether price is up or down. [liquidityfinder](https://liquidityfinder.com/news/macd-histogram-using-momentum-strength-to-read-trend-health-8ee55)

### `ATR_Pct`
- Category: core market.
- Source: `High`, `Low`, previous `Close`, current `Close`.
- Engineering: yes.
- Formula: `ATR_14 / Close_t`.
- Notes: tells the agent how large recent bar movement is relative to price, which helps interpret calm versus noisy conditions. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)

## AVWAP

### `AVWAP_Dist`
- Category: anchored value / market structure. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap)
- Source: `Close` and the internally maintained active AVWAP anchor. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap)
- Engineering: yes.
- Formula: `(Close_t / Active_AVWAP_t) - 1`. [alchemymarkets](https://alchemymarkets.com/education/indicators/anchored-vwap/)
- Notes: tells the agent whether price is above or below anchored fair value, and how far it has moved from the last accepted structural anchor. [alchemymarkets](https://alchemymarkets.com/education/indicators/anchored-vwap/)

### `AVWAP_Dist_ATR`
- Category: anchored value / market structure. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap)
- Source: `Close`, active AVWAP, and `ATR_14`. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
- Engineering: yes.
- Formula: `(Close_t - Active_AVWAP_t) / ATR_14_t`. [chartschool.stockcharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
- Notes: tells the agent whether the distance from AVWAP is small noise or a meaningful volatility-scaled extension. [alchemymarkets](https://alchemymarkets.com/education/indicators/anchored-vwap/)

## Trend and volatility

### `Dist_MA_Slow`
- Category: trend distance.
- Source: `Close`.
- Engineering: yes.
- Formula: `(Close_t / MA_50_t) - 1`.
- Notes: tells the agent whether price is aligned with the broader trend and how stretched it is from that slower trend anchor. [mql5](https://www.mql5.com/en/market/product/56837)

### `Realized_Vol_Short`
- Category: volatility state.
- Source: `Log_Return`.
- Engineering: yes.
- Formula: rolling 10-bar standard deviation of `Log_Return`, annualized.
- Notes: tells the agent how jumpy the market has been lately. [de.tradingview](https://de.tradingview.com/script/t89zQK0W-Realized-Volatility-StdDev-of-Returns/)

### `Vol_Regime`
- Category: volatility regime.
- Source: `Realized_Vol_Short` and `Realized_Vol_Long`.
- Engineering: yes.
- Formula: `Realized_Vol_Short / Realized_Vol_Long`.
- Notes: tells the agent whether recent volatility is elevated or subdued relative to the longer baseline. [arxiv](https://arxiv.org/html/2504.18958v1)

## Cross-asset and macro

### `QQQ_Ret`
- Category: cross-asset.
- Source: `QQQ` close from `yfinance`.
- Engineering: yes.
- Formula: `log(QQQ_Close_t / QQQ_Close_{t-1})`.
- Notes: tells the agent what the broader tech/market proxy is doing right now.

### `Rel_Strength_QQQ`
- Category: relative strength.
- Source: `Log_Return` and `QQQ_Ret`.
- Engineering: yes.
- Formula: 20-bar rolling sum of target returns minus 20-bar rolling sum of `QQQ_Ret`.
- Notes: tells the agent whether the stock is outperforming or underperforming the market proxy.

### `VIX_Z`
- Category: macro stress.
- Source: `^VIX` close from `yfinance`.
- Engineering: yes.
- Formula: `(VIX_Close_t - rolling_mean_20) / rolling_std_20`.
- Notes: daily-only feature that tells the agent whether market stress is elevated relative to its recent norm.

### `TNX_Z`
- Category: rates context.
- Source: `^TNX` close from `yfinance`.
- Engineering: yes.
- Formula: `(TNX_Close_t - rolling_mean_20) / rolling_std_20`.
- Notes: daily-only feature that tells the agent whether the rates backdrop is unusually high or low relative to recent history.

## News and time

### `News_Intensity`
- Category: news.
- Source: Alpaca headlines.
- Engineering: yes.
- Formula: count of headlines mapped into each market bar.
- Notes: tells the agent whether something exogenous is happening even when sentiment itself is weak or neutral.

### `Sentiment_Mean`
- Category: news.
- Source: Alpaca headlines scored by FinBERT.
- Engineering: yes.
- Method: headline-level sentiment scores are averaged within each market bucket after headlines are mapped to the next tradable bar.
- Notes: tells the agent the current bucket’s average news tone without adding the extra persistence of a sentiment EMA.

### `Sin_Time`
- Category: time.
- Source: bar timestamp.
- Engineering: yes.
- Formula: `sin(2π * time_index / period)`. [towardsdatascience](https://towardsdatascience.com/cyclical-encoding-an-alternative-to-one-hot-encoding-for-time-series-features-4db46248ebba/)
- Notes: tells the agent where the current bar sits in the repeating session or weekly cycle while preserving cyclical continuity. [developer.nvidia](https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/)

### `Cos_Time`
- Category: time.
- Source: bar timestamp.
- Engineering: yes.
- Formula: `cos(2π * time_index / period)`. [skforecast](https://skforecast.org/latest/faq/cyclical-features-time-series.html)
- Notes: works with `Sin_Time` to give a unique cyclical encoding of time without treating the clock as linear. [feature-engine.trainindata](https://feature-engine.trainindata.com/en/1.7.x/user_guide/creation/CyclicalFeatures.html)

## Forecast block

### `Forecast_Return_Hk`
- Category: forecast.
- Source: historical engineered numeric features.
- Engineering: yes.
- Method: random-forest regressor trained to predict `log(Close_{t+h} / Close_t)`.
- Notes: tells the agent the forecast model’s estimate of forward return. [perplexity](https://www.perplexity.ai/search/dc9dc1e9-14a2-436c-bcf0-c201d43eb2e0)

### `Forecast_Vol_Hk`
- Category: forecast.
- Source: historical engineered numeric features.
- Engineering: yes.
- Method: random-forest regressor trained to predict the standard deviation of the next `h` log returns.
- Notes: tells the agent the forecast model’s estimate of near-future volatility. [perplexity](https://www.perplexity.ai/search/dc9dc1e9-14a2-436c-bcf0-c201d43eb2e0)

### `Forecast_Uncertainty`
- Category: forecast.
- Source: random-forest ensemble dispersion.
- Engineering: yes.
- Method: standard deviation across individual tree predictions from the return model.
- Notes: tells the agent how uncertain the forecast block currently is. [perplexity](https://www.perplexity.ai/search/dc9dc1e9-14a2-436c-bcf0-c201d43eb2e0)

## Portfolio state

### `Cash_Ratio`
- Category: portfolio state.
- Source: user-provided environment state.
- Engineering: no.
- Notes: tells the agent how much dry powder is available.

### `Position_Size`
- Category: portfolio state.
- Source: user-provided environment state.
- Engineering: no.
- Notes: tells the agent its current directional exposure.

### `Inventory_Fraction`
- Category: portfolio state.
- Source: user-provided environment state.
- Engineering: no.
- Notes: tells the agent how much of the portfolio is currently tied up in holdings.

### `Unrealized_PnL`
- Category: portfolio state.
- Source: user-provided environment state.
- Engineering: no.
- Notes: tells the agent whether the current position is working or hurting.

### `Last_Action`
- Category: portfolio state.
- Source: user-provided environment state.
- Engineering: no.
- Notes: tells the agent its immediate action history, which can help reduce unstable flip-flopping.

## Practical notes

- This baseline is built around transformed and normalized signals rather than raw price levels, which is usually the cleaner default for trading ML and RL state design. [quantbeckman](https://www.quantbeckman.com/p/what-type-of-returns-should-i-feed)
- `VIX_Z` and `TNX_Z` are intended for daily bars only under your current data setup.
- `Sentiment_Mean` is kept, while `Sentiment_EMA` is intentionally removed from the baseline to avoid adding an overly persistent news feature.
- `Mins_to_Close` is intentionally excluded from the baseline because `Sin_Time` and `Cos_Time` already encode general intraday seasonality, and the countdown-to-close feature is more useful as a later ablation if close-specific behavior becomes important. [econ.kobe-u.ac](https://www.econ.kobe-u.ac.jp/wp/wp-content/uploads/2023/06/1722.pdf)
