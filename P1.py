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

from P1utilities import *

env = Environment()

bids = env.bids
prices = env.prices

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
