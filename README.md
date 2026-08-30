# PCA Statistical Arbitrage Strategy

This strategy investigates whether residuals derived from rolling PCA can be used in a mean-reversion strategy across a selection of different stocks. It focuses on not only building the strategy, but also on the testing and evaluation process.

The strategy was developed in two phases. 
**Version 1** establishes the initial research process. Although an intial, historic result seemed promising, the rolling evaluation produced a Sharpe of -0.02 and di not demonstrate a reliable positive edge. Nevertheless, it allowed me to scrutinise and utilise statistical inference in order to evaluate the strategy and check the validity of data.
**Version 2** retains the original mean-reversion idea, but redesigns the signal by using 'residual pressure' scaled by recent stock-specific residual volatility, and dynamically selects the amount of PCA factors used to describe market movement based on a threshold of explained variance. The strategy buys the three candidates with the strongest positive signal (based on the negative of the scaled residual pressure), and shorts the three with the most negative, using a 'dollar-neutral' portfolio.

Version 2 produced positive discovery results, achieving net Sharpe ratio of 0.804, CAGR of 6.55% on a separate testing dataset. However, the one sided p-value of 0.057 and bootstrap Sharpe confidence interval of [-0.148, 1.799] narrowly miss the thresholds which may suggest statistical significance.

The project therefore demonstrates the overall quantitative research process rather than caliming a proved trading edge, as discussed in the motivation section. It covers hypothesis formation, backtesting, strategy revision, external evaluation, transaction cost modelling, robustness testing and statistical evaluation.


## TL;DR


* Built two versions of a rolling PCA-based statistical arbitrage strategy. 
* Residuals calculated using rolling PCA and least squares regression in order to evaluate mean-reversion.
* Version 1 used cumulative z-scores and a rolling autocorrelation stock selection strategy. Evaluation produced a Sharpe of -0.02, showing that initially promising values did not generalise.
* Version 2 used dynamically selected PCA factors to calculate residuals, extending the hypothesis by measuring five-day residual pressure and trading the expected reversal.
* Discovery Sharpe ratios: 0.509 for Technology, 0.575 for Financial and 0.296 for Energy.
* On a UK external testing dataset, version 2 achieved a net Sharpe ratio of 0.804, CAGR of 6.55% and maximum drawdown of -13.54%
* External p-value was 0.057 and bootstrap Sharpe confidence interval was [-0.148, 1.779], so considered promising but not statistically validated.
* The main outcome is a transparent research process showing the evaluation and development of an unsucessful first idea into a more defensibe second strategy, highlighting the evaluation process using frequentist inference.

## Motivation

The project was built in order to develop intuition for quantitative research and to learn more about the general process of forming a hypothesis, backtesting and revising the strategy when the data doesn’t fully agree. The project has therefore evolved significantly from the original design as a result of this iterative process, as documented below. I also aimed to gain experience in criticising and evaluating my own data – after having studied the process of generating confidence intervals and hypothesis testing, I wanted to apply this to a real trading strategy, to develop the habit of scepticism when returns are very positive etc. The project therefore involves perhaps simplistic strategies (PCA, mean-reversion), combined in what I hope to be an insightful way, and developed slightly further using my own ideas to do with regimes and diversification. Therefore, the project's goal was less to prove that a simple strategy works, but more to learn how to test whether a promising result can truly suggest an edge, and to identify when conclusions may need to be revised.

---

# Version 1

The strategy uses PCA to decompose a basket of tech stocks into shared market factors, generating stock specific residuals. 

## Methodology

### 1. Rolling PCA

Rather than running PCA once on the full set of data (which would’ve allowed the model to use future data relative to the specific trading day), the strategy involved refitting the PCA every 5 trading days using the 60 previous days. Before fitting, returns in the 60 day window were standardised so that stocks with larger day to day movements didn't dominate the PCA just because they were more volatile. After each refit, the top 3 principal components are kept. These represent the shared patterns of stock movement.

### 2. Residuals via factor regression

For each rolling window, each stocks daily return is regressed using ordinary least squares regression, against the 3 PCA factors, producing a set of weights for each of the stocks. These weights were then used with the data from the next five days (out of sample). The process then continues, with the window sliding forward, discarding the previous weights and refitting. Using this prediction for each set of five days, the residual can be found by subtracting the prediction from the actual data. This signal is what is then used to determine the trades that are made.

### 3. Walk-forward stock selection

The strategy didn’t use a fixed stock basket throughout evaluation. After 252 initial trading days of residual history, it reassessed the eligible stocks each five trading days. On each reassessment date it calculated each stock’s autocorrelation using the residuals from before that date. This autocorrelation was ‘lag-one’ meaning that it was compared with the data from the day before, across all of the available past data. A negative autocorrelation is indication that stocks with highly positive residuals tend to be followed by negative residuals. This was used as a selection rule rather than proof. Stocks meeting this rule were then used for trading over the next five days.

### 4. Mean-reversion signal

The residuals are cumulatively summed over time for each of the current chosen stocks. A rolling z-score (over 20 days) is calculated using this in order to determine the position. For example, +/- 1.5 standard deviations for entry and +/- 0.5 for exit. Naturally this creates the opportunity cost of missing the potential profit from an overshoot. However, this trade-off is due to the reduced risk of holding the position through a reversal which could cause losses.

### 5. Portfolio construction, timing and transaction costs

On each day, active positions were given equal absolute weights so that the total summed to 1. Signals were also shifted forward one day before calculating returns. This was done to prevent returns used to create a signal from also being counted as profit from that signal. 
To calculate transaction costs, the turnover from one day to the next (absolute change in portfolio weights) was calculated. The costs were calculated as turnover multiplied by 5 basis points (0.05%) per unit traded. This cost was an assumption made for simplicity.

### 6. Exploratory research experiments

Before the final walk-forward evaluation, two related hypotheses were explored. The experiments form part of the project’s overall pipeline but were never actually used as part of the final trading rule for the first version of the strategy.

#### 6a. Regime filter experiment

PC1's explained variance ratio was used as an indicator of market cohesion. The hypothesis was that periods where the stocks were highly correlated could be riskier when trading with a portfolio of stocks from the same sector, as it ancels 

Using PC1's rolling explained variance ratio as an indicator of market cohesion, a regime filter was tested to see whether this could improve the performance of the base strategy. This was an assumption that was (perhaps wrongly, as discussed later), based on intuition when considering the fact that highly correlated periods could be riskier when trading with a portfolio of stocks from the same sector, as it cancels out the diversification from holding a range of different stocks from different markets. 

Before using the signal to create this filter, it was interesting to look into the different regimes identified by PC1’s explained variance ratio.  The highest negative cohesion reading from the data (74.5%) occurred in June 2025, which was then researched and identified as coinciding with the major US-China tariff de-escalation that triggered rising prices for semiconductor manufacturing firms. This was an exploratory observation rather than evidence that the cohesion measure could predict returns.

However, when using this to create a regime filter, the strategy reported weaker performance. After comparing the returns in each of the identified regimes, it seemed as though high cohesion days were not consistently worse, which contradicted the hypothesis. This may have been due to the fact that when the whole basket moves together, price variations are explained more clearly by PC1, leaving purer stock-specific residuals to use for signal generation. I then proceeded to explore how exactly performance varied with cohesion, and whether the relationship was a continuous one which may have been oversimplified by attempting to follow a binary rule. The observed relationship was U Shaped, with stronger results at both high and low cohesion levels, which suggests that using a binary filter is too simplistic of an approach.

This was not sufficient evidence to retain a regime filter, especially not a binary one. However, the experiment was still insightful for the process of testing hypotheses and investigating market structure, so has been included here for reference.

#### 6b. Residual behaviour and initial stock selection research

Earlier in the project, I used a chronological split of data in order to test whether the mean-reversion assumption holds for the stocks. Using an earlier period of data, the one lag autocorrelation method was used to select stocks, with a later period of data then being used to evaluate the chosen basket. This produced an encouraging result, but was then replaced by the walk forward method discussed in section 3, as the fixed historical basket does not account for changing market conditions. The static result was therefore treated as a research observation rather than proof of an edge.

#### 6c. Factor evaluation experiment

I also then explored how PCA factor loadings change through time. In particular I examined whether the second principal component added any relevant information by distinguishing between different areas of the market in certain periods.

This was an exploratory attempt to understand how the market sector might evolve after considering this when changing from the static autocorrelation test to the walk forward version. It was found that PC2 cleanly split the basket into software and semicondustor industries. Initial testing showed the two sectors moving in opposite directions, but when tested over time, it seemed as though this split between sectors varied not only in intensity but also in sign, sometimes moving opposite and sometimes toghether. The analysis therefore did not establish a stable, tradeable relationship and so this was not used in the final strategy.

### 7. Final version 1 walk forward evaluation

An initial evaluation of the strategy, using only the selected stocks as discussed previously, resulted in a perhaps naive Sharpe of 1.2. However, this couldn't be used in isolation as it tested only one period with a fixed basket and so was not reliable at all. The stock selection method from section 3 was used as discussed, with stock eligibility being assessed using past residuals. This eligibility controlled whether a stock could open a new position, but existing positions were not automatically closed if the stock later became ineligible. Instead it remained open until the normal Z-score exit condition was hit. This was done to prevent the five day selection schedule from creating an unintended five day holding period for all stocks.

The strategy was then evaluated across 688 trading days, providing the following, less favourable results:

*Annualized arithmetic mean return: -0.44%
*CAGR: -2.81%
*Annualized volatility: 21.96%
*Sharpe ratio: -0.02
*Maximum drawdown: -39.27%
*Average daily turnover: 32.89%
*Average net exposure: 1.82%

This suggests that the original stock selection idea did not generalise reliable when it was repeatedly evaluated through changing market conditions.

### 8. Statistical evaluation of Version 1

#### Hypothesis test

The primary question was:
Null hypothesis: the strategy has no positive average daily return.
Alternative hypothesis: the strategy has a positive average daily return.

I used a Newey-West adjusted test due to the fact that the daily returns are likely to be related with eachother due to volatility clustering and since positions remain open for multiple days.

The walk-forward mean daily return was around -0.002% with the one sided p-value being 0.5122. This is well above the 5% threshold and so does not provide evidence against the null hypothesis of no positive return. It does not prove that the strategy has a negative true edge.

#### Bootstrap test for Sharpe

I estimated uncertainty around the Sharpe ratio using a bootstrap test. This method resampled consecutive five-day blocks of returns, retaining some of the dependence, which resulted in a 95% confidence interval of -1.33 to 1.21. This interval inludes zero meaning that historical data is consistent with both negative and positive true Sharpe ratios.

### 9. Robustness checks

The fixed walk-forward Version 1 specification was tested without changing its rules after observing the results. These checks examine whether the conclusion was sensitive to transaction costs, time period, or individual stocks.

#### Transaction cost sensitivity:

| Assumed cost | Sharpe ratio | CAGR |
|---|---:|---:|
| 0 bp | 0.17 | 1.31% |
| 5 bp | -0.02 | -2.81% |
| 10 bp | -0.21 | -6.76% |
| 20 bp | -0.58 | -14.19% |
| 30 bp | -0.96 | -21.03% |

This shows how the strategy showed only modest positive returns before costs and became weaker once costs were added.

#### Stability across periods

| Period | Sharpe ratio | CAGR | Maximum drawdown |
|---|---:|---:|---:|
| 3 Nov 2023 – 20 Mar 2025 | -0.87 | -18.60% | -27.78% |
| 21 Mar 2025 – 4 Aug 2026 | 0.77 | 16.04% | -19.36% |

This data shows that performance between the two periods was not stable through time.

#### Dependence on individual stocks

I removed one stock at a time whilst keeping the remaining rules fixed. The results changed quite a lot depending on which stock was removed. For example, removing NVDA reduced the Sharpe ratio to -0.43. This is diagnostic evidence only - removing underperforming stocks would create a new strategy as it would bias the Version 1 results.

Overall, the robustness checks support the same cautious conclusion: Version 1 was sensitive to reasonable trading costs, market period, and individual constituents.

### Key findings

#### Final Version 1 Conclusion

The original static result did not generalise well under walk forward evaluation. After correcting portfolio construction, the initial walk forward Sharpe ratio was -0.02 after using 5bps costs. The hypothesis test and bootstrap confidence intervals also did not provide evidence of a positive edge.

#### Regime filter conclusion

The regime filter hypothesis was tested and not retained. The binary market cohesion filter did not improve results consistently enough to justify inclusion. It remains part of the documented research process because it tested a plausible hypothesis.

#### What the project demonstrates

The main outcome of version 1 is methodological rather than a validated trading signal. A promising initial historical result was challenged using walk forward evaluations. The final evidence didn’t support a reliable historical edge, so the strategy is treated as an exploratory research result.

## Limits and Evaluation

### Historical and research-selection limitations

The walk forward design prevents stock selection decisions from using future data that would not have been available at that time. However it does not make the final evaluation a completely untouched test. The strategy, thresholds and research pipeline were developed while examining the same overall data set. The version one result should therefore be treated as exploratory historical evidence rather than a definitive proof of performance.

### Execution and market assumptions

The backtest uses daily data and also delays signals by one trading day, which prevents lookahead bias. However, it assumes a simplified 5-bps transaction cost without considering short availability or other market constraints.

The tradeable stocks also consist of present-day technology stocks and so they may contain survivorship bias. A historical tradeable basket constructed from stocks available at each point in time maybe would have been a stronger design.

### Interpretation

The final version 1 design does not support a reliable positive trading edge. this does not prove that the underlying mean-reversion idea doesnt work, it just shows that this specific implementation didn't produce robust results after evaluation using statistical inference. The value of this part of the project lies in the research process by having tested an initially promising result.

## Next steps: creating a new version

Equipped with new knowledge of the research process, I continued to develop a new hypothesis branched off of the first version in order to see if I could create a strategy with a detectable, positive edge.

---

# Version 2

## Hypothesis

Version 1 tested whether stock specific price variations (identified using PCA residuals) mean reverted. After completing this and seeing that the results did not validate, I extended the idea by distinguishing between daily noise and sustained residual deviation in a particular direction over time.
The hypothesis is that several consecutive days of abnormal performance may temporarily move a stock away from its value predicted by its common factors. This could be due to temporary investor overreaction or other similar market events, which may eventually reverse over the following weeks. The hypothesis is therefore similar to that of version 1, but extended so that the direction is not just determined by the residual threshold alone, but also by the strength of the residual pressure over recent periods of time.
In addition to this, I also wanted to explore the impact of PCA on the information carried by the residuals. Therefore, it was hypothesised that manually restricting the percentage of the variance explained by the PCA and dynamically choosing enough factors such that this threshold of the variance was explained by PCA, may have an impact on the effectiveness of the residuals in a mean reversion strategy.

## Methodology

### 1. Rolling PCA

A rolling PCA model was fitted using the previous 126 trading days. In contrast to version 1, a threshold was use in order to control how much of the variance was to be explained by the PCA. 55% was chosen to test, in order balance removing common risk against preserving the information held by the residuals. It was frozen at this in order to reduce selection bias, but the tuning of this parameter remains a limitation which could be extended and investigated further.
As a result, version 2 dynamically selects only enough components to explain 55% of the variance. The model refits every 10 days using this rule, and only information available up to that point. 
This therefore allows for the number of factors to adjust as the structure of the market changes.

### 2. Residual and signal calculations

For each stock, the residuals were calculated by subtracting the explained PCA components from the observed return. Residuals are summed over the last 5 trading days, with a negative sum representing stock-specific underperformance and a positive sum representing the opposite. This sum then needed to be contextualised, which I chose to do by dividing it by the residual volatility over the last 20 days (the standard deviation of residuals over the last 20 days) in order to make sure that naturally more volatile stocks didn’t get receive more extreme scores consistently. These final results were called the residual pressure.
Signals were then created based on these final results – strongly negative residual pressure generated a positive signal, as we expect the stock to revert imminently in line with the hypothesis, and vice versa. The signal was just taken as the negative of the residual pressure.

### 3. Portfolio construction

After 252 days of history, the strategy then ranks the stocks based on the residual pressure values found previously. Every twenty days, the following procedure takes place:
1.	Three stocks with the strongest positive expected-return signal (based on the negative of residual pressure) are bought.
2.	Three stocks with the strongest negative expected-return signals are sold short.
3.	The long side receives 50% of the portfolio weight.
4.	The short side receives 50% of the portfolio weight.
5.	Positions are held until the next scheduled rebalance.
By doing this we aim to remail independent or neutral on the market and so the gains do not purely come from complete market performance (which is already contributed to by removing the common market factors).

### 4. Timings

Positions are delayed by one trading day as before. Positions remain open until each 20-day rebalance, when stocks no longer in the top or bottom three are closed, with newly selected stocks opened. An interesting extension of this may be to weight stocks differently depending on how they are ranked, or to create an exit strategy that is based on signals, as this removed the key limitation that the stock may be held after it undergoes reversal. 
Twenty days was chosen as the rebalance period as research showed that faster trading generally produced higher turnover and weaker net results.

## Results

### 1. Discovery findings

The frozen strategy was evaluated across three development universes.

| Universe | Sharpe | CAGR | Maximum drawdown |
|---|---:|---:|---:|
| Technology | 0.509 | 5.62% | -17.60% |
| Financials | 0.575 | 4.09% | -10.20% |
| Energy | 0.296 | 2.94% | -16.06% |

The strategy produced an initial positive sharpe across all three, but the technology result was unstable across time, meaning that the strategy may be dependent on market regime and that more testing was naturally required.

### 2. Frozen external evaluation

The strategy rules remained frozen before evaluating on an unused universe of 15 UK stocks from 2017 to 2021.

| Metric | Result |
|---|---:|
| Annualised arithmetic mean return | 6.69% |
| CAGR | 6.55% |
| Net Sharpe ratio | 0.804 |
| Gross Sharpe ratio | 0.900 |
| Maximum drawdown | -13.54% |
| Average daily turnover | 6.33% |
| First-half Sharpe | 1.580 |
| Second-half Sharpe | 0.404 |

Results remained positive, but also showed instability over time as previously mentioned.

## Evaluation

### Statistical evaluation

The primary hypotheses were: 
 Null hypothesis - the strategy's average daily return is not positive.
 Alternative hypothesis - the strategy's average daily return is positive.

Again, a one sided Newey-West adjusted test was used due to dependence over time. 

| Statistical measure | Result |
|---|---:|
| HAC test statistic | 1.580 |
| One-sided p-value | 0.057 |
| 95% bootstrap Sharpe interval | [-0.148, 1.779] |

As shown, the test narrowly avoided the standard 5% significance level, however this interval is naturally just a chosen threshold, and so is still somewhat meaningful as a result.
The bootstrap confidence interval also crossed zero, meaning that the strategy remains consistent with a slightly negative Sharpe and a strongly positive one.

The conclusion is therefore that Version 2 produced a promising external performance, but the evidence is not quite strong enough to establish a positive edge.

It also should be noted that these values are subject to research-selection bias, after exploring multiple candidate extensions of Version 1.

### 2. Robustness checks

#### Transaction cost sensitivity

| Assumed cost | Sharpe | CAGR |
|---|---:|---:|
| 0 bp | 0.900 | 7.40% |
| 5 bp | 0.804 | 6.55% |
| 10 bp | 0.707 | 5.70% |
| 20 bp | 0.512 | 4.02% |

Performance remained positive under higher transaction cost assumptions, with declining Sharpe and CAGR (compound annual growth rate) as expected.

#### Dependence on individual stocks

The strategy was recalculated after removing each UK stock individually. Sharpe ratios ranged from 0.472 to 1.179, with a median of approximately 0.754, with each result remaining positive, which is promising.

#### Stability across periods

| Period | Sharpe | CAGR | Maximum drawdown |
|---|---:|---:|---:|
| First half | 1.580 | 9.50% | -6.83% |
| Second half | 0.404 | 3.67% | -13.54% |

Both periods were profitable, but the difference between Sharpe rations does suggest regime dependency.

These checks are diagnostic rather than additional independent validation tests.

### 3. Limitations

* the external sample covers only five years.
* survivorship bias due to choosing current UK stocks.
* transaction costs are simplified.
* exit signals are times instead of based on exit signals.
* performance differed between subperiods.
* strategy exploration may have induced selection-bias.
* the confidence interval still includes zero.

## Conclusion

Version 2 is perhaps a more defensibe extension of Version 1, with more promising data whilst preserving the original mean reversion hypothesis, but providing more justification for holding specific stocks. It tested the hypothesis that sustained stock-sopecific 'residual pressure' may move prices away from their values implied by PCA factors for a temporary period. Under this hypothesis, stocks with strongly negative residual pressures should then revert, outperforming other stocks and vice versa.

The strategy achieved positive results across three development universes and produced a net Sharpe ratio of 0.804 with a 6.55% CAGR on fresh UK data, suggesting some directional consistency with the hypothesis. The strategy was also reasonably robust when checked, as detailed in the previous section.

Several parts of the extended strategy may have accounted for this. Using dynamic PCA may have helped in removing dominant, shared movements before calculating signals. The use of residual pressure here may also have helped to more accurately predict mean-reversion rather than just using thresholds. Scaling by residual volatility may also have helped in preventing naturally volatile stocks from dominating the rankings consistently. Holding the positions for 20 days may also have helped to reduce excessive turnover and allow for reversion to occur before closing positions. As an extension, it would be interesting to see whether performance could have been further improved by using specific exit signals instead of timed exits to prevent positions from being held after mean reversion.

However, its p-value narrowly missed the 5% threshold, its bootstrap confidence interval crossed zero. Version 2 should therefore be interpreted as a promising out-of-sample research result rather than a statistically proved trading edge. 

The evidence must therefore be considered as consistent with the proposed mechanism and hypothesis, but not enough to prove that the strategy has a persistent positive edge. Version 2 could be interpreted as a promising extension of the first version, rather than a statistically validated trading strategy.








