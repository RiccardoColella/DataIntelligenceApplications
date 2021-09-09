'''this file contains some functions used to calculate the best possible rewards,
best daily price, etc. ... '''

from environment import Environment
import numpy as np

env = Environment()

def get_bid_and_price_revenue(bid, price, customer_class):
    '''returns the mean daily revenue given a bid , a price and a customer class'''
    cost_per_click = env.get_mean_cost_per_click(bid, customer_class)
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

def get_best_price(prices, customer_class):
    '''this function returns the best price among a vector of prices for a given customer class'''
    revenues = []
    for price in prices:
        revenue = env.get_margin(price) * env.get_conversion_rate(price, customer_class) * (1 + env.get_mean_n_times_comeback(customer_class) )
        revenues.append(revenue)

    max_revenue = max(revenues)
    return prices[revenues.index(max_revenue)]

def get_best_bid(bids, price, customer_class):
    '''return the bid associated to the best revenue given a price and a customer class'''
    revenues = []
    for bid in bids:
        revenues.append(get_bid_and_price_revenue(bid, price, customer_class))

    max_revenue = max(revenues)
    return bids[revenues.index(max_revenue)]

def get_best_bid_and_price(bids, prices, customer_class):
    '''return the best bid and price for a specific customer class'''
    best_price = get_best_price(prices, customer_class)
    best_bid = get_best_bid(bids, best_price, customer_class)
    return best_bid, best_price

def get_best_bid_price_users_possible_reward_per_class(bids, prices):
    '''return 4 list containing best price, best bid, best users number and best possible reward for every customer classes'''

    best_bid = [0 for i in range(3)]
    best_price = [0 for i in range(3)]
    best_users = [0 for i in range(3)]
    best_possible_reward = [0 for i in range(3)]

    for i in range(1, 4):
        best_bid[i-1], best_price[i-1] = get_best_bid_and_price(bids, prices, i)
        best_users =  env.get_mean_new_users_daily(best_bid[i-1], i)
        best_possible_reward[i-1] = get_bid_and_price_revenue(best_bid[i-1], best_price[i-1], i)

    return best_bid, best_price, best_users, best_possible_reward
