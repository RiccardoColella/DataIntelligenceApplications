# the following 15 lines just add verbose option

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
verbose = parser.parse_args().verbose

if verbose:
    def log(argument):
        print(argument)
else:
    def log(argument):
        return

# now the real code begins

from environment import Environment
import numpy as np

env = Environment()


def get_best_price(prices, customer_class):
    '''this function returns the best price among a vector of prices for a given customer class'''
    revenues = []
    for price in prices:
        revenue = env.get_margin(price) * env.get_conversion_rate(price, customer_class) * (1 + env.get_mean_n_times_comeback(customer_class) )
        revenues.append(revenue)

    max_revenue = max(revenues)
    return prices[revenues.index(max_revenue)]


def get_bid_and_price_revenue(bid, price, customer_class):
    '''returns the mean daily revenue given a bid , a price and a customer class'''
    cost_per_click = env.get_cost_per_click(bid, customer_class)
    new_users_daily = env.get_mean_new_users_daily(bid, customer_class)
    buy_percentage = env.get_conversion_rate(price, customer_class)
    mean_comebacks = env.get_mean_n_times_comeback(customer_class)

    purchases = new_users_daily * buy_percentage
    revenue = purchases * (1+mean_comebacks) * env.get_margin(price) - new_users_daily * cost_per_click

    log('bid, price, customer_class, revenue: ' + str(bid) +str(', ') + str(price) +str(', ')+ str(customer_class) +str(', ')+ str(revenue))

    return revenue


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


bids = np.linspace(0.1, 1, num=10)
prices = np.linspace(1, 10, num=10)

if __name__ == '__main__':
    # find the best joint bid and price strategy for all the customer classes
    for i in range(1, 4):
        print('The best joint bid and price strategy for class ' + str(i) + ' is ' + str(get_best_bid_and_price(bids, prices, i)))

    # find the best joint bid and price strategy if it is not possible to discrimate between the classes

    revenuesmatrix = np.arange(bids.size * prices.size)
    revenuesmatrix = revenuesmatrix.reshape(bids.size, prices.size)

    revenuesmatrix_per_class = []
    for i in range(1, 4):
        revenuesmatrix_per_class.append(np.arange(bids.size * prices.size))
        revenuesmatrix_per_class[-1] = revenuesmatrix_per_class[-1].reshape(bids.size, prices.size)

    for i in range(bids.size):
        for j in range(prices.size):
            revenuesmatrix[i][j] = get_bid_and_price_revenue(bids[i], prices[j], 1) + get_bid_and_price_revenue(bids[i], prices[j], 2) +  get_bid_and_price_revenue(bids[i], prices[j], 3)
            for k in range(1, 4):
                revenuesmatrix_per_class[k-1][i][j] = get_bid_and_price_revenue(bids[i], prices[j], k)

    best_bid = bids[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[0]]
    best_price = prices[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[1]]

    log("Revenue matrix per class:")
    for i in range(0,3):
        log('class' + str(i+1))
        log(revenuesmatrix_per_class[i])
    log("Revenue matrix:")
    log(revenuesmatrix)

    print("The best joint bid and price strategy is (" + str(best_bid) + ', ' + str(best_price) + ')')
