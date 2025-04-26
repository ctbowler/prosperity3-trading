# IMC Prosperity 2025 Trading Algorithms 
- This repository contains my trading algorithms developed for the IMC Prosperity Trading Competition. Strategies focus on market making, volatility arbitrage, spread trading, and multi-asset momentum detection across a range of structured products, options, and baskets.
- All algorithms were implemented in Python and adhere to the Object Oriented Structure of the IMC Trading Bots (see wiki: https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-19ee8453a0938114a15eca1124bf28a1).

# Strategies
## Rainforest Resin Market Making
- **Context:** Rainforest Resin was an extremely stable product and appeared to have minimal external shocks, which allowed us to deduce a low volatility market making strategy.
  The bid-ask spreads were tight but consistent, and exhibited relatively symmetric order book dynamics.

- **Idea:** Maintain inventory neutrality by passively posting limit orders slightly inside the natural bid-ask spread.
  Dynamically adjust quoting behavior based on inventory imbalances, becoming more aggressive when heavily long or short.

- **Math:**
Mid-price estimation:
<p align="center">
$$
\text{Midprice} = \frac{\text{Best Bid} + \text{Best Ask}}{2}
$$
</p>

Undercutting the spread to improve fill probability:
<p align="center">
$$
\text{Improved Bid} = \text{Best Bid} + 1 
$$
$$
\text{Improved Ask} = \text{Best Ask} - 1
$$
</p>

Inventory risk adjustment based on position:
<p align="center">
$$
\text{Target Spread} \propto -\text{Inventory}
$$
</p>

- **Profitability:** Rainforest Resin provided stable, low-volatility profits.
  We were consistently able to accumulate gains through small tick-size improvements without large directional risk exposure.
  In rounds with large liquidity, this strategy experienced minimal slippage and low adverse selection risk.

- **Looking back:** 
This strategy could have been further improved by introducing dynamic spread widening based on short-term volatility bursts.
When volumes spiked or when adverse signals were present (e.g., sudden one-sided order books), we could have slightly relaxed my quotes instead of maintaining symmetric undercutting; nonetheless, the Resin market proved extremely favorable to classical low-volatility market making.




## Squid Ink Mean Reversion
- **Context:** Squid Ink prices showed erratic short-term movements but tended to revert to a moving equilibrium.
  Unlike Rainforest Resin, the Squid Ink order book was typically thin and subject to larger volatility bursts, making low volatility market making (as we did for Resin) riskier without additional filters.

- **Idea:** Compute the rolling Exponential Moving Average (EMA) and rolling volatility of mid-prices.
  Enter trades based on Z-score deviations between current price and EMA, but exit early on momentum shifts to avoid losses from strong breakouts.

- **Math:**
Mid-price calculation:
<p align="center">
$$
\text{Midprice}_t = \frac{\text{Best Bid}_t + \text{Best Ask}_t}{2}
$$
</p>

EMA of midprices (span $n$):
<p align="center">
$$
\text{EMA}_t = \alpha \times \text{Midprice}_t + (1 - \alpha) \times \text{EMA}_{t-1}
$$
</p>
where
<p align="center">
$$
\alpha = \frac{2}{n+1}
$$
</p>

Rolling volatility:
<p align="center">
$$
\sigma_t = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
$$
</p>

Z-score trading signal:
<p align="center">
$$
z_t = \frac{\text{Midprice}_t - \text{EMA}_t}{\sigma_t}
$$
</p>

Trade triggers:
<p align="center">
$$
z_t > \text{threshold} \text{ → Short Squid Ink}
$$

$$
 z_t < -\text{threshold} \text{ → Long Squid Ink}
$$
</p>
Exit on Z-score crossing zero or on momentum reversal.

- **Profitability:**  
This strategy performed strongly in moderately volatile conditions.
The majority of profits came from mean-reversion after overextensions, especially during mid-day low volume periods when Squid Ink naturally reverted toward its EMA.
However, during high volatility shocks, the static thresholds sometimes triggered premature entries.

- **Looking back:**  
In retrospect, the static Z-score threshold could have been dynamically adjusted based on short-term volatility regimes.
During "quiet" periods, a lower threshold would have captured more subtle reversions, while during volatile bursts, a higher threshold would have prevented entries against breakouts.
Additionally, integrating a short-term momentum indicator would have improved exit timing and profitability.
Nonetheless, the EMA/volatility Z-score approach proved highly effective relative to the erratic Squid Ink market conditions.







## Picnic Basket Pairs Trading
- **Context:** Picnic Basket 1 and Picnic Basket 2 were composite assets made up of CROISSANTS, JAMS, and DJEMBES.
  The two baskets were highly correlated but constructed with slightly different component ratios, creating predictable spread dynamics.
  Although the composite formulas were given for each basket, the data did not support the idea that their composite spreads were mean reverting; rather, the spread between their midprices served as a more useful indicator. 
  This setup was ideal for **statistical arbitrage** through mean reversion of the spread.

- **Idea:**  Track the historical spread between the two baskets' midprices, model the spread as a stationary process, and trade based on extreme deviations from the historical mean.
  Adjust trade aggressiveness based on recent volatility to avoid entering during unstable periods.

- **Math:**
Midprice estimation for each basket:
<p align="center">
$$
\text{Midprice}_{\text{Basket}} = \frac{\text{Best Bid} + \text{Best Ask}}{2}
$$
</p>

Spread calculation:
<p align="center">
$$
\text{Spread}_t = \text{Midprice}_{\text{Basket1}, t} - \text{Midprice}_{\text{Basket2}, t}
$$
</p>

Rolling Z-score of spread:
<p align="center">
$$
z_t = \frac{\text{Spread}_t - \mu_{\text{Spread}}}{\sigma_{\text{Spread}}}
$$
$$
\text{where } \mu_{\text{Spread}} \text{ and } \sigma_{\text{Spread}} \text{ are rolling mean and standard deviation.}
$$
</p>

Trade triggers:
<p align="center">
$$ 
z_t > \text{Upper Threshold} \text{ → Short Basket 1, Long Basket 2}
$$
$$ 
z_t < \text{Lower Threshold} \text{ → Long Basket 1, Short Basket 2}
$$
$$
\text{Exit trades when} |z_t| \text{falls below a small threshold, indicating mean reversion.}
$$
</p>
  
- **Profitability:**  
Pairs trading on the baskets was consistently profitable in rounds with stable market conditions.
The spread mean-reverted reliably, and controlling volatility exposure helped limit losses during spread widening periods.
Profits were especially strong when using tighter exit thresholds to secure reversion gains before potential spread rebounding.

- **Looking back:**  
While the pairs trading strategy worked well, it could have been enhanced by dynamically sizing trades based on spread volatility or half-life estimation (if modeled as an Ornstein-Uhlenbeck process) of the spread mean-reversion.
Additionally, introducing cointegration testing would have allowed for more adaptive spread selection, especially if external shocks had caused component prices (like CROISSANTS or JAMS) to decorrelate unexpectedly.
Practically, it's computationally stressful to perform cointegration tests periodically but in a real trading environment it may be worth it. Nonetheless, static Z-score based spread trading proved highly effective in the structured environment provided by the baskets.




## Volcanic Rock Voucher Delta Hedging
- **Context:** European call options were simulated as Volcanic Rock Vouchers with the underlying asset being Volcanic Rock. Vouchers were available at the following strike prices:
  9500, 9750, 10000, 10250, and 10500
- **Idea:** Delta-hedge voucher positions against the underlying asset based on Black-Scholes estimated theoretical prices, minimizing first-order exposure to market movements. Independently, monitor deviations between market voucher prices and theoretical values to signal spread reversion trading opportunities.
- **Math:**
  The price of the underlying asset, $S(t)$, is modeled by the following stochastic differential equation (SDE):

<p align="center">
$$
dS(t) = \mu S(t)dt + \sigma S(t) dW(t) 
$$
</p>

  With $\mu =$ drift rate and $\sigma =$ volatility. This is also recognized as a Geometric Brownian Motion process which guarantees positivity in $S(t)$.

One can derive a model for the Voucher (option) price, $V(t)$, through a multivariable second order Taylor expansion adapted to stochastic processes, using the identity $(dW)^2 = dt$, in conjunction with the stock returns SDE (1) (i.e., Itô's Lemma):

<p align="center">
$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
$$
</p>

This equation is known as the Black-Scholes Equation. The details are omitted here, but a closed-form solution can be found by converting the Black-Scholes model to the standard heat equation and proceeding via convolution with a Gaussian distribution to yield:

<p align="center">
$$
V(S_0, 0) = S_0 N(d_1) - K e^{-rT} N(d_2)
$$
</p>

<p align="center">
$$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}},
\quad
d_2 = d_1 - \sigma \sqrt{T}
$$
</p>

<p align="center">
$$
\begin{aligned}
S_0 & : \text{Current stock price at time } t = 0 \\
K & : \text{Strike price (exercise price of the option)} \\
r & : \text{Risk-free interest rate (annualized)} \\
T & : \text{Time to expiration (in years)} \\
\sigma & : \text{Volatility of the stock (annualized)} \\
N(\cdot) & : \text{Cumulative distribution function (CDF) of the standard normal distribution}
\end{aligned}
$$
</p>

For delta-hedging, I computed:
<p align="center">
$$
\Delta = \frac{\partial V}{\partial S} = N(d_1) 
$$
</p>

This measures the local sensitivity of the voucher (option) price to changes in the underlying asset price.
By computing Delta, I dynamically hedge my voucher orders by holding the appropriate number of underlying assets needed to minimize first-order exposure to market price fluctuations<sup>1</sup>. 
Hence, we model the asset's expected return as the risk-free rate, $$r$$,  when computing the theoretical voucher value, $$V(S, t)$$.
- **Profitability:** Using theoretical price to set spread reversion thresholds worked well in both backtesting and on the IMC submission, securing a profit of 118k (seashells) on just round 3.
  Taking long positions when the theoretical price exceeded the market price of a voucher (and vice versa for shorting) was often profitable on both the voucher and the underlying asset hedge.
   However, in round 4, the spread reversion thresholds for the 9500 and 9750 vouchers were indicating many false signals; nonetheless, the total pnl from the underlying asset and other vouchers was net positive. Vouchers were not submitted for the final round as most of my time went into refining baskets and squid ink.  

- **Looking back:**  Although Black-Scholes is a stylized model, it was well-suited for pricing the vouchers, as later confirmed by IMC in the competition wiki. Incorporating a stochastic volatility model would have been unnecessary and impractical:
(i) there was limited evidence of latent volatility fluctuations during trading, and
(ii) more complex methods such as Markov Chain Monte Carlo (MCMC) would have introduced substantial runtime complexity without clear performance gains.
While the Black-Scholes model was effective, I believe my implementation could have been further improved to increase profitability.
 The competition wiki indicated that implied volatility—obtained by inverting the Black-Scholes formula—might serve as a useful signal for trend detection.
  In my original implementation, I used Black-Scholes theoretical prices computed from rolling historical volatility as a mean-reversion signal.
  However, this approach caused my vega estimates to lag behind the actual market conditions, limiting its effectiveness as an additional indicator due to the highly volatile nature of the vouchers. 
  Directly extracting implied volatility from market prices and using it (and Vega) as a primary signal might have been a more effective strategy for capturing trend and reversion opportunities.
 However, I was not able to develop an implied volatility reversion strategy that outperformed my original theoretical price-based model in terms of stability and profitability.

<sub> 1: *"Appropriate" here refers to minimizing first-order exposure **under the assumptions** of the Black-Scholes model: constant volatility, continuous trading, frictionless markets, and no jumps.</sub>*




## Magnificent Macarons Fair-Value Trading
- **Context:** The price of Magnificent Macarons was linked to two environmental factors — the sunlight index and sugar price — which made it possible to estimate a rough fair value at any time.
Instead of just reacting to price moves like in pure technical trading, I could build a model that tried to predict what macarons "should" be worth based on external data.
The problem was that the macaron market was really illiquid, which meant sharp price jumps and big gaps in the order book happened a lot.

- **Idea:** I built a simple fair-value model by running a linear regression of macaron prices against sunlight and sugar levels.
Whenever the actual market mid-price got too far above or below the estimated fair value, I entered trades expecting a reversion.
To avoid getting stuck during big breakout moves, I added a momentum slope filter — only trading when momentum wasn’t fighting against the mean-reversion idea.


- **Math:**
Fair value model (given):
<p align="center">
$$
\text{FairValue}_t = -2.836 \times \text{SunlightIndex}_t + 3.327 \times \text{SugarPrice}_t + 118.173
$$
</p>

Spread between market midprice and fair value:
<p align="center">
$$
\text{Spread}_t = \text{FairValue}_t - \text{Midprice}_t
$$
</p>

Momentum slope of 100-period rolling average:
<p align="center">
$$
\text{MomentumSlope}_t = \text{MA}_{100,t} - \text{MA}_{100,t-5}
$$
</p>

Trade triggers:
<p align="center">
$$
\text{Spread}_t > \text{Threshold} \text{ and momentum slope} < 0  \text{ → **Long Macarons**}
$$
$$
\text{Spread}_t < -\text{Threshold} \text{ and momentum slope}  > 0 \text{ → **Short Macarons**}
$$
</p>
Threshold dynamically adjusted based on recent volatility and position.


- **Profitability:**  
The macarons fair-value strategy worked well when sunlight or sugar shocks caused clear mispricings in the market.
Most profits came from simple mean-reversion back to the fair value after temporary moves.
The strategy performed best when momentum signals helped avoid chasing into big breakouts.
However, low liquidity sometimes made it hard to get good fills, and exits during volatile periods occasionally caused slippage.
In some cases, strong-looking momentum was actually just noise, and the strategy entered positions right before major breakouts, causing noticeable losses.

- **Looking back:**  
One of the biggest challenges was assuming the fair value model always held in the short term.
During illiquid periods, market prices could drift far from fair value just because of order book imbalances, not real mispricing.
In the future, using extra filters like bid-ask volume imbalance or recent trade flow pressure could help avoid bad entries.
Also, scaling position sizes based on how far the market was from fair value — instead of treating every signal the same — might have helped reduce losses on weaker signals.
Overall, despite some missteps around breakout events, the fair-value plus momentum approach was very effective for trading macarons most of the time.


