# IMC Prosperity 2025 Trading Algorithms 
- This repository contains my trading algorithms developed for the IMC Prosperity Trading Competition. Strategies focus on market making, volatility arbitrage, spread trading, and multi-asset momentum detection across a range of structured products, options, and baskets.
- All algorithms were implemented in Python and adhere to the Object Oriented Structure of the IMC Trading Bots (see wiki: https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-19ee8453a0938114a15eca1124bf28a1).

# Strategies
## Rainforest Resin Market Making
## Squid Ink Mean Reversion 
## Picnic Basket Pairs Trading
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
  
