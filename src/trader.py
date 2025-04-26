from datamodel import OrderDepth, UserId, TradingState, Order, Trade, Listing, ProsperityEncoder, Symbol, Observation # IMC Data Model,
from blackscholes import BlackScholes
from typing import Dict, List, Any
import collections
from collections import defaultdict
import math
import copy
import numpy as np
import json
import pandas as pd
import jsonpickle
import numpy as np
from math import log, sqrt, exp


empty_dict = {
    "RAINFOREST_RESIN": 0, "KELP": 0, "SQUID_INK": 0,
    "VOLCANIC_ROCK": 0, "VOLCANIC_ROCK_VOUCHER_9500": 0,
    "VOLCANIC_ROCK_VOUCHER_9750": 0, "VOLCANIC_ROCK_VOUCHER_10000": 0,
    "VOLCANIC_ROCK_VOUCHER_10250": 0, "VOLCANIC_ROCK_VOUCHER_10500": 0,
    "CROISSANTS": 0, "JAMS": 0, "DJEMBES": 0,
    "PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0
}
position = copy.deepcopy(empty_dict)
kelp_cache = []
squid_ink_cache = []
kelp_dim = 5
squid_ink_lr_dim = 5
squid_ink_breakout_dim = 50
squid_ink_swing_dim = 5

EMA_LENGTH = 200
VOL_LENGTH = 200
ZSCORE_THRESHOLD = 1
K = 2
MAX_TRADE_SIZE = 15
last_z_sign = 0



# Trader class for handling trading logic and order generation
# Author: Carson Bowler, Elijah Schwab 
class Trader:
    def __init__(self):
        self.avg_price = 0.0
        self.cpos = 0.0
        self.POS_LIMIT = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
    
    # This specific function was created by: https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    


    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2))) / 2
    
    
    def calc_next_price_regression(self, cache: List[int], dim: int) -> Any:
        """
        Calculate next price using linear regression on the last 'dim' time steps.
        If there is insufficient data, return None.
        """
        if len(cache) < dim:
            return None

        # Create feature matrix X (with an intercept) and target vector y
        X = np.array(range(dim)).reshape(-1, 1)
        y = np.array(cache[-dim:])
        # Add constant term for intercept
        X = np.hstack([np.ones((dim, 1)), X])
        # Compute regression coefficients: beta = (X^T*X)^{-1}*X^T*y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        # Predict price at the next time step (time index = dim)
        next_time = np.array([[1, dim]])
        next_price = next_time @ beta
        return int(round(next_price[0]))

    def compute_ema(self, prices: List[float]) -> float:
        if not prices:
            return 0.0
        alpha = 2.0 / (len(prices) + 1)
        ema_val = prices[0]
        for p in prices[1:]:
            ema_val = alpha * p + (1 - alpha) * ema_val
        return ema_val

    def compute_std(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 1.0
        mean_val = sum(prices) / len(prices)
        var = sum((p - mean_val) ** 2 for p in prices) / (len(prices) - 1)
        return max(math.sqrt(var), 1e-9)

    def values_extract(self, order_dict, buy: int = 0) -> Any:
        tot_vol = 0
        best_val = -1
        mxvol = -1
        for price, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = price
        return tot_vol, best_val




    def compute_orders_rainforest_resin(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []
        asks = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        bids = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        sell_vol, best_sell_pr = self.values_extract(asks)
        buy_vol, best_buy_pr = self.values_extract(bids, 1)
        cpos = self.position[product]
        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        if mprice_actual <= 9998 and cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
            remaining_capacity = self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos
            if remaining_capacity > 0:
                orders.append(Order(product, best_sell_pr, remaining_capacity))
                cpos += remaining_capacity
                if cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
                    orders.append(Order(product, best_sell_pr + 1, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos))
            return orders

        if mprice_actual >= 10002 and cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
            remaining_capacity = self.POSITION_LIMIT['RAINFOREST_RESIN'] + cpos
            if remaining_capacity > 0:
                orders.append(Order(product, best_buy_pr, -remaining_capacity))
                cpos -= remaining_capacity
                if cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
                    orders.append(Order(product, best_buy_pr - 1, -(self.POSITION_LIMIT['RAINFOREST_RESIN'] + cpos)))
            return orders

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1
        generic_base_order_size = 25

        half_position_limit = self.POSITION_LIMIT['RAINFOREST_RESIN'] / 2
        is_significantly_long = cpos > half_position_limit
        is_significantly_short = cpos < -half_position_limit

        if is_significantly_long:
            sell_pr = max(undercut_sell - 1, acc_ask - 1)
            bid_pr = min(undercut_buy, acc_bid - 1)
        elif is_significantly_short:
            bid_pr = min(undercut_buy + 1, acc_bid + 1)
            sell_pr = max(undercut_sell, acc_ask + 1)
        else:
            bid_pr = min(undercut_buy, acc_bid - 1)
            sell_pr = max(undercut_sell, acc_ask + 1)

        for ask, vol in asks.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
                order_for = min(-vol, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
                cpos += order_for
                orders.append(Order(product, ask, order_for))

        if cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
            base_order_size = generic_base_order_size
            if cpos < 0:
                target_buy = min(base_order_size * 2, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
            else:
                target_buy = min(base_order_size, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
            if target_buy > 0:
                orders.append(Order(product, bid_pr, target_buy))
                cpos += target_buy

        cpos = self.position[product]
        for bid, vol in bids.items():
            if ((bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
                order_for = max(-vol, -self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
                cpos += order_for
                orders.append(Order(product, bid, order_for))

        if cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
            base_order_size = generic_base_order_size
            if cpos > 0:
                target_sell = min(base_order_size * 2, self.POSITION_LIMIT['RAINFOREST_RESIN'] + cpos)
            else:
                target_sell = min(base_order_size, self.POSITION_LIMIT['RAINFOREST_RESIN'] + cpos)
            if target_sell > 0:
                orders.append(Order(product, sell_pr, -target_sell))
                cpos -= target_sell

        return orders
        return []



    def compute_orders_kelp(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)
        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < self.POSITION_LIMIT[product]:
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.POSITION_LIMIT[product]:
            num = self.POSITION_LIMIT[product] - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                orders.append(Order(product, bid, order_for))

        if cpos > -self.POSITION_LIMIT[product]:
            num = -self.POSITION_LIMIT[product] - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
        return []




    def compute_orders_squid_ink(self, product, order_depth):
        orders = []
        sells_sorted = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buys_sorted = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        _, best_sell_pr = self.values_extract(sells_sorted, 0)
        _, best_buy_pr = self.values_extract(buys_sorted, 1)
        if best_sell_pr <= 0 or best_buy_pr <= 0:
            return orders
        current_mid = 0.5 * (best_sell_pr + best_buy_pr)
        cpos = self.position[product]
        limit = self.POSITION_LIMIT[product]
        needed_len = max(self.EMA_LENGTH, self.VOL_LENGTH)
        if len(self.squid_ink_cache) < needed_len:
            return orders
        slice_vals = self.squid_ink_cache[-needed_len:]
        ema_val = self.compute_ema(slice_vals)
        std_val = self.compute_std(slice_vals)
        z = (current_mid - ema_val) / std_val
        sign_z = 1 if z > 0 else -1 if z < 0 else 0
        if self.last_z_sign != 0 and sign_z != 0 and sign_z != self.last_z_sign:
            delta = -cpos
            if delta > 0:
                buy_price = best_sell_pr + 1
                orders.append(Order(product, buy_price, delta))
            elif delta < 0:
                sell_price = best_buy_pr - 1
                orders.append(Order(product, sell_price, delta))
            self.last_z_sign = 0
            return orders
        if abs(z) <= self.ZSCORE_THRESHOLD:
            self.last_z_sign = sign_z
            return orders
        base_qty = int(round(self.K * abs(z)))
        trade_qty = min(base_qty, self.MAX_TRADE_SIZE) // 5
        if z > 0:
            short_cap = limit + cpos
            can_sell = min(trade_qty, short_cap)
            if can_sell > 0:
                sell_price = best_buy_pr - 1
                orders.append(Order(product, sell_price, -can_sell))
                self.last_z_sign = sign_z
        else:
            buy_cap = limit - cpos
            can_buy = min(trade_qty, buy_cap)
            if can_buy > 0:
                buy_price = best_sell_pr + 1
                orders.append(Order(product, buy_price, can_buy))
                self.last_z_sign = sign_z
        return orders


    def compute_orders(self, product, order_depth, acc_bid=0, acc_ask=0):
        if product == "RAINFOREST_RESIN":
            return self.compute_orders_rainforest_resin(product, order_depth, acc_bid, acc_ask)
        elif product == "KELP":
            return self.compute_orders_kelp(product, order_depth, acc_bid, acc_ask)
        elif product == "SQUID_INK":
            return self.compute_orders_squid_ink(product, order_depth)
        return []
    

    def compute_orders_all_vouchers(self, state: TradingState, memory: dict) -> tuple[Dict[str, List[Order]], dict]:
        result = {k: [] for k in self.POS_LIMIT}
        

        order_depths = state.order_depths

        if "base_iv_history" not in memory:
            memory["base_iv_history"] = []
            memory["prev_spread"] = 0
            memory['mid_prices_vouchers'] = {"VOLCANIC_ROCK_VOUCHER_9500": [], "VOLCANIC_ROCK_VOUCHER_9750": [], "VOLCANIC_ROCK_VOUCHER_10000": [], "VOLCANIC_ROCK_VOUCHER_10250": [], "VOLCANIC_ROCK_VOUCHER_10500": []}

        TTE = 2 / 365
        product = "VOLCANIC_ROCK"
        vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]

        # Get VOLCANIC_ROCK mid price
        rock_book = order_depths.get(product)
        if not rock_book or not rock_book.buy_orders or not rock_book.sell_orders:
            return {}, memory

        best_bid = max(rock_book.buy_orders)
        best_ask = min(rock_book.sell_orders)
        St = (best_bid + best_ask) / 2

        # Prepare voucher books
        voucher_books = {}
        for v in ["VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750"]:
            book = order_depths.get(v)
            if not book or not book.buy_orders or not book.sell_orders:
                return {}, memory
            voucher_books[v] = book

        m_list, iv_list = [], []

        for voucher in vouchers:
            book = order_depths.get(voucher)
            if not book or not book.buy_orders or not book.sell_orders:
                continue
            bid = max(book.buy_orders)
            ask = min(book.sell_orders)
            Vt = (bid + ask) / 2
            K = int(voucher.split("_")[-1])
            try:
                iv = BlackScholes.implied_volatility(Vt, St, K, TTE)
                m = log(K / St) / sqrt(TTE)
                m_list.append(m)
                iv_list.append(iv)
            except:
                continue

        if len(m_list) < 3:
            return {}, memory

        a, b, c = np.polyfit(m_list, iv_list, deg=2)
        current_iv = c

        # Update memory
        memory["base_iv_history"].append(current_iv)
        if len(memory["base_iv_history"]) > 200:
            memory["base_iv_history"].pop(0)

        base_iv_array = np.array(memory["base_iv_history"])
        if len(base_iv_array) < 51:
            return {}, memory

        window = base_iv_array[-50:]
        rolling_mean = np.mean(window)
        volatility = np.std(window)
        spread = current_iv - rolling_mean
        prev_spread = memory["prev_spread"]
        memory["prev_spread"] = spread

        # Order logic
        position = state.position.get(product, 0)
        rock_pos = position
        orders = []


       
        # === Soft Orders ===
        if spread < -0.001 and position < self.POS_LIMIT[product]:
            rock_size = 5
            orders.append(Order(product, best_bid, rock_size))
        elif spread > 0.001 and position > -self.POS_LIMIT[product]:
            rock_size = -5
            orders.append(Order(product, best_ask, rock_size))
        else:
            rock_size = 0

        # === Delta Hedge ===
        deltas = {
            "VOLCANIC_ROCK_VOUCHER_10000": BlackScholes.delta(St, 10000, TTE, volatility),
            "VOLCANIC_ROCK_VOUCHER_10250": BlackScholes.delta(St, 10250, TTE, volatility),
            "VOLCANIC_ROCK_VOUCHER_10500": BlackScholes.delta(St, 10500, TTE, volatility),
            "VOLCANIC_ROCK_VOUCHER_9500": BlackScholes.delta(St, 9500, TTE, volatility),
            "VOLCANIC_ROCK_VOUCHER_9750": BlackScholes.delta(St, 9750, TTE, volatility),
        }

        

        # === Hard Orders ===
        price_shift = int(volatility * 10)
        volume_scale = max(1, int(volatility * 5))

        if -0.007 < spread < -0.004:
            sell_price = best_bid 
            if rock_pos < 0:
                volume_scale = int(volume_scale*2)
            else:
                volume_scale = int(volume_scale*1)
            orders.append(Order(product, best_ask, volume_scale))
            rock_size = -volume_scale
        elif spread > 0.005:
            buy_price = best_ask
            orders.append(Order(product, best_bid, -volume_scale))
            rock_size = volume_scale
        else:
            rock_size = 0

        if rock_size != 0:
            for voucher, delta in deltas.items():
                best_bid_voucher = max(voucher_books[voucher].buy_orders)
                best_ask_voucher = min(voucher_books[voucher].sell_orders)
                memory['mid_prices_vouchers'][voucher].append((best_bid_voucher + best_ask_voucher) / 2)
                if len(memory['mid_prices_vouchers'][voucher]) > 20:
                    memory['mid_prices_vouchers'][voucher].pop(0)
                    prod_mid = memory['mid_prices_vouchers'][voucher]
                    ma_fast = np.mean(prod_mid[-5:])
                    ma_slow = np.mean(prod_mid[-20:])
                    mom = ma_fast - ma_slow
                else:
                    mom = 0

                prev_hedge = state.position.get(voucher, 0)
                hedge = -delta * rock_size
                hedge = int(hedge)

                pos_limit = self.POS_LIMIT[voucher]
                max_qty = pos_limit - abs(prev_hedge)
                
                if hedge > 0:
                    size = min(hedge, max_qty)
                    price = min(state.order_depths.get(voucher).sell_orders)
                elif hedge < 0:
                    size = max(hedge, -max_qty)
                    price = max(state.order_depths.get(voucher).buy_orders)
                else:
                    continue

                if size != 0:
                    result.setdefault(voucher, []).append(Order(voucher, price, size))




                theo_price = BlackScholes.black_scholes_call(St, int(voucher.split("_")[-1]), TTE, volatility)
                margin = abs((theo_price - (best_bid + best_ask) / 2) / theo_price) * 100
                voucher_pos = state.position.get(voucher, 0)
                # Regular Voucher Orders
                if best_ask_voucher < theo_price and voucher_pos < 200 :
                    # Go Long
                    if margin > 10:
                        orders.append(Order(product, best_ask, 20))
                
                elif best_bid_voucher > theo_price and voucher_pos > -200 :
                    # Go Short
                    if margin > 10:
                        orders.append(Order(product, best_bid, -20))

        result[product] = orders
    
      
        return result, memory
    

    

    def compute_orders_macarons(self, state, memory):
        product = "MAGNIFICENT_MACARONS"
        result = {product: []}
        conversions = 0
        depth = state.order_depths.get(product)
        self.POS_LIMIT_MAC = 75
        self.CONV_LIMIT = 10
        self.BASE_SIZE = 10
        self.SPREAD = 2
        if "mid_prices_mac" not in memory:
            memory["mid_prices_mac"] = []
        if "avg_price_mac" not in memory:
            memory["avg_price_mac"] = 0

        osell = collections.OrderedDict(sorted(state.order_depths['MAGNIFICENT_MACARONS'].sell_orders.items()))
        obuy = collections.OrderedDict(sorted(state.order_depths['MAGNIFICENT_MACARONS'].buy_orders.items(), reverse=True))
        ask_vol, best_ask = self.values_extract(osell)
        bid_vol, best_bid = self.values_extract(obuy, 1)
        mid_price = (best_bid + best_ask) / 2 
        
        memory['mid_prices_mac'].append(mid_price)
     
        # Linear Regression for predicted value
        sugar = state.observations.conversionObservations[product].sugarPrice
        sun = state.observations.conversionObservations[product].sunlightIndex
        fair = -2.83629*sun + 3.327* sugar + 118.17
        spread = fair - mid_price
        
        # Momentum Slope Calculation
        if len(memory['mid_prices_mac']) > 105:
            memory['mid_prices_mac'].pop(0)
            mom_slope = pd.Series(memory['mid_prices_mac']).rolling(100).mean().diff(5).iloc[-1]
        else:
            mom_slope = 0
        ma_100 = pd.Series(memory['mid_prices_mac']).rolling(100).mean().iloc[-1]
        
        pos = state.position.get(product, 0)
        cpos = pos

        ##################### --- Trades --- #####################
        # SHORT TRADES
        # Capitalize on obvious mean reversion
        if (spread < - 50 and mom_slope >= 2) or (spread < -20 and mom_slope > 2) and pos > - self.POS_LIMIT_MAC :
            #print("Shorting")
            trade_size = min(self.BASE_SIZE, self.POS_LIMIT_MAC + pos)
            trade_price = best_bid
            result[product].append(Order(product, trade_price, -trade_size))
            cpos -= trade_size

        # Initial Condition for when Momentum not computed
        elif mom_slope == 0 and spread < 0 and pos > - self.POS_LIMIT_MAC:
            trade_size = min(self.BASE_SIZE, self.POS_LIMIT_MAC + pos)
            trade_price = best_bid
            result[product].append(Order(product, trade_price, -trade_size))
            cpos -= trade_size

            #conversions += trade_size
        
        elif spread < 0 and mom_slope > 1 and ma_100 < mid_price and pos > - self.POS_LIMIT_MAC:
            trade_size = min(self.BASE_SIZE, self.POS_LIMIT_MAC + pos)
            trade_price = best_bid
            result[product].append(Order(product, trade_price, -trade_size))
            cpos -= trade_size
        
        # LONG TRADES
        elif (spread > 20 and 0 > mom_slope > -2) or (spread > 0 and 0 < mom_slope < -1) and pos < self.POS_LIMIT_MAC:
            #print("Buying")
            trade_size = min(self.BASE_SIZE, self.POS_LIMIT_MAC - pos)
            trade_price = best_ask  
            result[product].append(Order(product, trade_price, trade_size))
            cpos += trade_size
            #cpos += -trade_size
            #conversions += -trade_size
        elif spread > 10 and mom_slope < - 1 and pos < self.POS_LIMIT_MAC: 
            trade_size = min(self.BASE_SIZE, self.POS_LIMIT_MAC - pos)
            trade_price = best_ask  
            result[product].append(Order(product, trade_price, trade_size))
            cpos += trade_size
            #cpos += -trade_size

        elif spread > 0 and pos < -50 and mom_slope <= -1.5:
            trade_size = min(self.BASE_SIZE*3, self.POS_LIMIT_MAC - pos)
            trade_price = best_ask  
            result[product].append(Order(product, trade_price, trade_size))
            cpos += trade_size
            #cpos += -trade_size

        return result, memory
    
    
    
    def extract_best_prices(self, depth):
            if not depth or not depth.buy_orders or not depth.sell_orders:
                return None, None, 0, 0
            best_bid = max(depth.buy_orders)
            bid_vol = depth.buy_orders[best_bid]
            best_ask = min(depth.sell_orders)
            ask_vol = depth.sell_orders[best_ask]
            return best_bid, best_ask, bid_vol, ask_vol

    def update_avg_entry(self, memory, sym, new_price, new_size, pos):
        if pos == 0:
            memory[sym]["avg_entry"] = float(new_price)
        elif (pos > 0 and new_size > 0) or (pos < 0 and new_size < 0):
            prev_pos = abs(pos)
            new_pos = prev_pos + abs(new_size)
            weighted = memory[sym]["avg_entry"] * prev_pos + float(new_price) * abs(new_size)
            memory[sym]["avg_entry"] = weighted / new_pos
        else:
            memory[sym]["avg_entry"] = float(new_price)
    
    
    def compute_orders_baskets(self,state, memory):
        result = {"PICNIC_BASKET1": [], "PICNIC_BASKET2": []}
        base_size = 20
        window = 100
        basket1 = "PICNIC_BASKET1"
        basket2 = "PICNIC_BASKET2"
        max_pos1 = 60
        max_pos2 = 100

        if "PICNIC_BASKET1" not in memory:
            memory["PICNIC_BASKET1"] = {"mid_prices": [], "avg_entry": 0.0}
        if "PICNIC_BASKET2" not in memory:
            memory["PICNIC_BASKET2"] = {"mid_prices": [], "avg_entry": 0.0}
       

        def get_mid(symbol):
            depth = state.order_depths.get(symbol)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                return None
            bid, ask, _, _ = self.extract_best_prices(depth)
            return (bid + ask) / 2

        mid1 = get_mid(basket1)
        mid2 = get_mid(basket2)

        if mid1 is not None and mid2 is not None:
            memory[basket1]["mid_prices"].append(mid1)
            memory[basket2]["mid_prices"].append(mid2)

        for sym in [basket1, basket2]:
            if len(memory[sym]["mid_prices"]) > 200:
                memory[sym]["mid_prices"].pop(0)

        result[basket1] = []
        result[basket2] = []

        prices1 = memory[basket1]["mid_prices"]
        prices2 = memory[basket2]["mid_prices"]
        

        if len(prices1) >= window and len(prices2) >= window:
            spread = pd.Series(prices1) - pd.Series(prices2)
            mean = spread.rolling(window).mean().iloc[-1]
            std = spread.rolling(window).std().iloc[-1]
            z = (spread.iloc[-1] - mean) / std

            pos1 = state.position.get(basket1, 0)
            pos2 = state.position.get(basket2, 0)

            bid1, ask1, bid_vol1, ask_vol1 = self.extract_best_prices(state.order_depths.get(basket1))
            bid2, ask2, bid_vol2, ask_vol2 = self.extract_best_prices(state.order_depths.get(basket2))

            if z > 2:
                size1 = min(base_size, max_pos1 + pos1, bid_vol1)
                size2 = min(base_size, max_pos2 - pos2, ask_vol2)
                if size1 > 0:
                    result[basket1].append(Order(basket1, int(bid1), -size1))
                    self.update_avg_entry(memory, basket1, bid1, -size1, pos1)
                if size2 > 0:
                    result[basket2].append(Order(basket2, int(ask2), size2))
                    self.update_avg_entry(memory, basket2, ask2, size2, pos2)

            elif z < -2:
                size1 = min(base_size, max_pos1 - pos1, ask_vol1)
                size2 = min(base_size, max_pos2 + pos2, bid_vol2)
                if size1 > 0:
                    result[basket1].append(Order(basket1, int(ask1), size1))
                    self.update_avg_entry(memory, basket1, ask1, size1, pos1)
                if size2 > 0:
                    result[basket2].append(Order(basket2, int(bid2), -size2))
                    self.update_avg_entry(memory, basket2, bid2, -size2, pos2)

            elif abs(z) < 0.5:
                # Exit if profitable
                if pos1 > 0 and bid1 > memory[basket1]["avg_entry"] + 10:
                    result[basket1].append(Order(basket1, int(bid1), -pos1))
                    memory[basket1]["avg_entry"] = 0.0
                elif pos1 < 0 and ask1 < memory[basket1]["avg_entry"] -10:
                    result[basket1].append(Order(basket1, int(ask1), -pos1))
                    memory[basket1]["avg_entry"] = 0.0

                if pos2 > 0 and bid2 > memory[basket2]["avg_entry"] + 10:
                    result[basket2].append(Order(basket2, int(bid2 + 1), -pos2))
                    memory[basket2]["avg_entry"] = 0.0
                elif pos2 < 0 and ask2 < memory[basket2]["avg_entry"] - 10:
                    result[basket2].append(Order(basket2, int(ask2 - 1), -pos2))
                    memory[basket2]["avg_entry"] = 0.0

     
        return result["PICNIC_BASKET1"], result['PICNIC_BASKET2'], memory


    # SUBMIT ORDERS
    def run(self, state: TradingState):

        for product in state.market_trades:
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                if trade.buyer == "SUBMISSION":
                    self.position[product] -= trade.quantity
                if trade.seller == "SUBMISSION":
                    self.position[product] += trade.quantity
        result = {"RAINFOREST_RESIN": [], "KELP": [], "SQUID_INK": [], "PICNIC_BASKET1": [], 'PICNIC_BASKET2': [], "VOLCANIC_ROCK_VOUCHER_9500": [], "VOLCANIC_ROCK_VOUCHER_9750": [],
                  "VOLCANIC_ROCK": [], "VOLCANIC_ROCK_VOUCHER_10000": [], "VOLCANIC_ROCK_VOUCHER_10250": [], "VOLCANIC_ROCK_VOUCHER_10500": [], "MAGNIFICENT_MACARONS": []}
        for key, val in state.position.items():
            self.position[key] = val
        acc_bid = {"RAINFOREST_RESIN": 10000, "KELP": 0, "SQUID_INK": 0}
        acc_ask = {"RAINFOREST_RESIN": 10000, "KELP": 0, "SQUID_INK": 0}
        for product in ["KELP", "SQUID_INK"]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                _, best_sell = self.values_extract(collections.OrderedDict(sorted(order_depth.sell_orders.items())))
                _, best_buy = self.values_extract(collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True)), 1)
                mid_price = (best_sell + best_buy) / 2
                if product == "KELP":
                    if len(self.kelp_cache) == self.kelp_dim:
                        self.kelp_cache.pop(0)
                    self.kelp_cache.append(mid_price)
                    kelp_pred = self.calc_next_price_regression(self.kelp_cache, self.kelp_dim)
                    if kelp_pred is not None:
                        acc_bid["KELP"] = kelp_pred - 1
                        acc_ask["KELP"] = kelp_pred + 1
                    else:
                        acc_bid["KELP"] = best_buy
                        acc_ask["KELP"] = best_sell
                else:
                    if len(self.squid_ink_cache) == self.squid_ink_breakout_dim + 5:
                        self.squid_ink_cache = self.squid_ink_cache[-(self.squid_ink_breakout_dim + 5):]
                    self.squid_ink_cache.append(mid_price)
                    squid_ink_pred = self.calc_next_price_regression(self.squid_ink_cache[-self.squid_ink_lr_dim:], self.squid_ink_lr_dim) if len(self.squid_ink_cache) >= self.squid_ink_lr_dim else None
                    if squid_ink_pred is not None:
                        acc_bid["SQUID_INK"] = squid_ink_pred - 1
                        acc_ask["SQUID_INK"] = squid_ink_pred + 1
                    else:
                        acc_bid["SQUID_INK"] = best_buy
                        acc_ask["SQUID_INK"] = best_sell
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                orders = self.compute_orders(product, order_depth, acc_bid.get(product, 0), acc_ask.get(product, 0))
                result[product] += orders
        
        # Decode shared memory once
        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {}

        orders_all, memory = self.compute_orders_all_vouchers(state, memory)
        for k, v in orders_all.items():
            result[k] += v

        orders_mac, memory = self.compute_orders_macarons(state, memory)
        result['MAGNIFICENT_MACARONS'] += orders_mac['MAGNIFICENT_MACARONS']
        
        orders_b1, orders_b2, memory = self.compute_orders_baskets(state, memory)
        result['PICNIC_BASKET1'] += orders_b1
        result['PICNIC_BASKET2'] += orders_b2
       

        traderData = jsonpickle.encode(memory)
        conversions = 0
        return result, conversions, traderData
