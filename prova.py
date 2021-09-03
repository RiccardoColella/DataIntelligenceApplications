import numpy as np
from P1asutilities import get_best_bid_price_possible_reward

#bids and prices range
prices = np.linspace(1, 10, num=10)
bids = np.linspace(0.1, 1, num=10)

bids, best_daily_price, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
bids = [bids]

print(bids, best_daily_price, best_possible_reward)
