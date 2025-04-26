"""
constants.py

Global constants for position limits, trading parameters, and product configurations.

Author: Carson Bowler
"""

# Position limits for each product
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "MAGNIFICENT_MACARONS": 75,
}

# Default empty position dictionary
EMPTY_POSITION = {product: 0 for product in POSITION_LIMITS.keys()}

# EMA and volatility parameters
EMA_LENGTH = 200
VOL_LENGTH = 200

# Trading thresholds
ZSCORE_THRESHOLD = 1
K = 2  # Scaling factor for trade size
MAX_TRADE_SIZE = 15

# Special parameters for basket trading
BASKET1_COMPONENTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_COMPONENTS = {"CROISSANTS": 4, "JAMS": 2}
