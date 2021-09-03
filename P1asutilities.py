'''this file contains some functions used to calculate the best possible rewards,
best daily price, etc. ...
basically it is the same as p1.py but made as functions'''

from environment import Environment
import numpy as np

env = Environment()

def get_bid_and_price_revenue(bid, price, customer_class):
    '''returns the mean daily revenue given a bid , a price and a customer class'''
    cost_per_click = env.get_cost_per_click(bid, customer_class)
    new_users_daily = env.get_mean_new_users_daily(bid, customer_class)
    buy_percentage = env.get_conversion_rate(price, customer_class)
    mean_comebacks = env.get_mean_n_times_comeback(customer_class)

    purchases = new_users_daily * buy_percentage
    revenue = purchases * (1+mean_comebacks) * env.get_margin(price) - new_users_daily * cost_per_click

    return revenue

def get_best_bid_price_possible_reward(bids, prices):
    '''return best price, best bid, and best possible reward if no discrimination is made between customer classes'''
    revenuesmatrix = np.arange(bids.size * prices.size)
    revenuesmatrix = revenuesmatrix.reshape(bids.size, prices.size)

    for i in range(bids.size):
        for j in range(prices.size):
            revenuesmatrix[i][j] = get_bid_and_price_revenue(bids[i], prices[j], 1) + get_bid_and_price_revenue(bids[i], prices[j], 2) +  get_bid_and_price_revenue(bids[i], prices[j], 3)

    best_bid = bids[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[0]]
    best_price = prices[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[1]]
    best_possible_reward = np.max(revenuesmatrix)

    return best_bid, best_price, best_possible_reward
