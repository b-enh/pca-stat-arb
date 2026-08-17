# PCA Statistical Arbitrage Strategy

A statistical arbitrage strategy, developed using a rolling PCA factor decomposition and validated against test data, separate from the original sample used for strategy development. Deployed for automatic paper trading via Alpaca API.

## TL;DR

* Residuals calculated via rolling PCA, least squares regression.
* Mean-reversion used and evaluated, using autocorrelation to identify stocks compatible with the strategy. This identification will later be used on a rolling basis to facilitate changing market conditions.
* Binary regime filter evaluated after identifying a U-Shaped relationship between performance and market cohesion, which could be used to create a dynamic and continuous regime classification.
* Annualised return of 27.79% and a Sharpe of 1.20.
* Bootstrapped using Sharpe values to check confidence intervals due to highly positive returns, 95% confidence interval within [-0.64, 3.04] with 89.5% of samples above 0. See end for key evaluation, hypothesis testing and discussion.


## Motivation

The project was built in order to develop intuition for quantitative research and to learn more about the general process of forming a hypothesis, backtesting and revising the strategy when the data doesn’t fully agree. The project has therefore evolved significantly from the original design as a result of this iterative process, as documented below. I also aimed to gain experience in criticising and evaluating my own data – after having studied the process of generating confidence intervals and hypothesis testing, I wanted to apply this to a real trading strategy, to develop the habit of scepticism when returns are very positive etc. The project therefore involves perhaps simplistic strategies (PCA, mean-reversion), combined in what I hope to be an insightful way, and developed slightly further using my own ideas to do with regimes and diversification.

## Overview

The strategy uses PCA to decompose a basket of tech stocks into shared market factors, generating stock specific residuals. 

**Pipeline:** raw prices > rolling PCA factor decomposition > residual calculation via a least-squares regression > z-score mean reversion signal generation > validation of the selected stocks > backtesting > live paper trading.

## Methodology

### 1. Rolling PCA

Rather than running PCA once on the full set of data (which would’ve allowed the model to use future data relative to the specific trading day), the strategy involved refitting the PCA every 5 trading days using the 60 previous days. After each refit, the top 3 principal components are kept. These represent the shared patterns of stock movement.

### 2. Residuals via factor regression

For each rolling window, each stocks daily return is regressed using least squares, against the 3 PCA factors, producing a set of weights for each of the stocks. These weights were then used with the data from the next five days (out of sample). The process then continues, with the window sliding forward, discarding the previous weights and refitting. Using this prediction for each set of five days, the residual can be found by subtracting the prediction from the actual data. This signal is what is then used to determine the trades that are made.

### 3. Mean-reversion signal

The residuals are cumulatively summed over time for each of the individual stocks. A rolling z-score (over 20 days) is calculated using this in order to determine the position. For example, +/- 1.5 standard deviations for entry and +/- 0.5 for exit. Naturally this creates the opportunity cost of missing the potential profit from an overshoot. However, this trade-off is due to the reduced risk of holding the position through a reversal which could cause losses.

### 4. Testing the core strategy

After a simple initial test to ensure that the mean reversion strategy was working before building on top of it, the Sharpe ratio was found to be negative (-0.28). This led to two separate investigations of what could have been causing this issue:

#### 4a: Testing the hypothesis that highly correlated periods would be riskier.
Using PC1's rolling explained variance ratio as an indicator of market cohesion, a regime filter was tested to see whether this could improve the performance of the base strategy. This was an assumption that was (perhaps wrongly, as discussed later), based on intuition when considering the fact that highly correlated periods could be riskier when trading with a portfolio of stocks from the same sector, as it cancels out the diversification from holding a range of different stocks from different markets. 

Before using the signal to create this filter, it was interesting to look into the different regimes identified by PC1’s explained variance ratio.  The highest negative cohesion reading from the data (74.5%) occurred in June 2025, which was then researched and identified as coinciding with the major US-China tariff de-escalation that triggered rising prices for semiconductor manufacturing firms. This demonstrated that the signals generated were showing market structure instead of just noise.

However, when using this to create a regime filter, performance was worsened, lowering the Sharpe ratio to -0.63. After comparing the returns in each of the identified regimes, it seemed as though high cohesion days performed better on average, which was the reverse of the hypothesis. This may have been due to the fact that when the whole basket moves together, price variations are explained more clearly by PC1, leaving purer stock-specific residuals to use for signal generation. However, this alone was not enough to justify the low Sharpe ratio calculated previously, and so another test was used.

#### 4b. Testing the strategy's core assumption
Testing the strategy’s core assumption – do the stocks actually mean-revert. Using autocorrelation, it was found that a few of the stocks, such as AAPL and AMZN showed weak positive autocorrelation with previous dates when lagged by 1, suggesting momentum rather than reversion. As a future extension of the project, it would be interesting to see whether regimes could be identified in which some stocks could have a momentum strategy applied, and others a mean-reversion strategy, as it seems as though momentum comes into play especially after firm specific news or press coverage.

### 5. Train/test split for stock selection

As a result, the tradeable stocks were then restricted, which had to be done carefully as selecting stocks based on behaviour over the same period used for testing would cause bias. The dataset was therefore split, now involving an earlier selection period which narrowed the tradeable stocks down to the 7 stocks that exhibited mean-reverting properties. This resulted in a much higher Sharpe ratio of 1.2 net of transaction costs. This high figure is clearly a positive finding, however it must be evaluated and scrutinised, as discussed later, particularly due to the changing market conditions and therefore the possibility of a stock changing its behaviour and moving away from mean-reverting tendencies.

### 6. Regime analysis

Since the earlier regime filter was being tested on a broken strategy, it was retested on the newly chosen stocks for completeness rather than expecting a drastic improvement. the test period was too short to give conclusive results and so I tested on the full dataset of around 940 days and compared this to a strategy without the regime filter. It produced negligible effect with a small drawdown improvement. I also wanted to check if using a binary filter limited the capabilities of using regime filtering, and so wanted to check whether performance scaled continuously. It was found that the performance scaled in a U-Shaped relationship with cohesion, rather than a usable monotonic trend, and so I did not build further ontop of this, however it will be interesting to investigate this further in the future.

### 7. Live paper trading

The validated strategy runs automatically once a day via Alpaca. Each run re-derives the rolling PCA and residuals from data in order to determine signals. 

### Key findings

**The core strategy validated well out-of sample.** On the test period of 281 trading days, the strategy produced an annualised return of 27.79% and a Sharpe ratio of 1.20 net of transaction costs. Performance remained intact after excluding the single best trading day, suggesting that the result wasn't due to one lucky outlier. However, for quite a basic strategy, the returns and Sharpe ratio seemed to be very high and so these results are later evaluated using bootstrap sampling to construct confidence intervals around the Sharpe ratio.

**The regime-timing hypothesis was tested twice, with conflicting results.** The second test on the refined stock selection found only a small benefit from excluding the highest cohesion days, which may have been noise. The project therefore does not provide strong evidence for whether a binary regime filter adds value reliably. It may be the case that a continuous filter provides better results – the performance vs. cohesion relationship formed a U shaped trend, with strongest performance at both extremes. This was, however, tested using a small sample of data, and so this relationship could also evolve with time. It would therefore perhaps be beneficial to look into the potential benefits of developing a dynamic, continuous regime filter in the future.

**Rolling PCA indicates how factor and therefore market structure evolves with time.** Using PC2’s loading allowed for the identification of different market patterns coinciding with real world events, which could also be used to influence future strategies involving the relationship between software and semiconductor industries.

## Limits and Evaluation

### Small-sample uncertainty
The high Sharpe ratio and figure for annualised returns requires evaluation, as while waiting for paper trading to continue to gain good insight into how well the strategy is performing, it is hard to identify whether there was a reasonable edge with testing on a somewhat short test period. To quantify this, I bootstrapped the daily returns, resampling in order to compute the Sharpe ratio each time. I found that the 95% confidence interval was between -0.64 and 3.04, with 89.5% of the resamples yielding a positive Sharpe. This is clearly a wide interval meaning that, with a longer testing period, it may be shown that the calculated Sharpe is closer to 0 than originally calculated. However, the confidence interval is clearly positioned much more above 0 than it is below it, meaning that the true value is much more likely to be contained within the positive interval. One thing to note is that this bootstrap treats daily returns as independent which is naturally not the case due to clustered volatility and the fact that the strategy holds multi-day positions. A one-sample t-test against a null hypothesis of the strategy having a 0 mean daily return gave a one sided p-value of 0.10, which is not significant at the 5% level. Combined with the confidence interval, this suggests that the sample size is too small to distinguish the strategy's edge from zero with confidence. A larger sample is required, ideally supplied by the ongoing paper trading. 


### Selection risk beyond lookahead bias
The train test split prevents direct lookahead bias, but it does not mean that there is no bias at all from selecting 7 stocks that fit the prediction and trading only with these based on their previous performance. As markets change, these stocks may stop showing mean-reverting tendencies. However, what these tests show are that clearly by testing using autocorrelation and selecting stocks to trade in the period of time after such a selection, the mean-reverting strategy works much better than by selecting stocks randomly. Therefore, such selection could be done on a rolling basis in order to select stocks to trade to ensure that they remain mean-reverting.














