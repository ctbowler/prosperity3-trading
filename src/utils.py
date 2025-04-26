"""
utils.py

Helper functions for financial calculations, linear regression, EMA, and Black-Scholes option pricing.

Author: Carson Bowler
"""

import math
import numpy as np
from math import log, sqrt, exp
from statistics import NormalDist

def norm_cdf(x: float) -> float:
    """
    Computes the cumulative distribution function for a standard normal distribution.

    Author: Carson Bowler
    """
    return (1.0 + math.erf(x / math.sqrt(2))) / 2

def compute_ema(prices: list[float]) -> float:
    """
    Computes the Exponential Moving Average (EMA) of a list of prices.

    Author: Carson Bowler
    """
    if not prices:
        return 0.0
    alpha = 2.0 / (len(prices) + 1)
    ema_val = prices[0]
    for p in prices[1:]:
        ema_val = alpha * p + (1 - alpha) * ema_val
    return ema_val

def compute_std(prices: list[float]) -> float:
    """
    Computes the standard deviation of a list of prices.

    Author: Carson Bowler
    """
    if len(prices) < 2:
        return 1e-9
    mean_val = sum(prices) / len(prices)
    var = sum((p - mean_val) ** 2 for p in prices) / (len(prices) - 1)
    return max(math.sqrt(var), 1e-9)

def calc_next_price_regression(cache: list[int], dim: int) -> int | None:
    """
    Predicts the next price using linear regression over the last 'dim' points.

    Author: Carson Bowler
    """
    if len(cache) < dim:
        return None

    X = np.array(range(dim)).reshape(-1, 1)
    y = np.array(cache[-dim:])
    X = np.hstack([np.ones((dim, 1)), X])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    next_time = np.array([[1, dim]])
    next_price = next_time @ beta
    return int(round(next_price[0]))

class BlackScholes:
    """
    Black-Scholes option pricing and implied volatility estimation.

    Author: Eric Liu (see: https://github.com/ericcccsliu/imc-prosperity-2)
    """

    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=100, tolerance=1e-5):
        low_vol = 0.01
        high_vol = 6
        volatility = 0.5 * (low_vol + high_vol)
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = 0.5 * (low_vol + high_vol)
        return volatility

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)
