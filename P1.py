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
    revenue = purchases * (1 + mean_comebacks) * env.get_margin(price) - new_users_daily * cost_per_click
    return revenue


def get_best_bid(bids, price, customer_class):
    '''return the bid associated to the best revenue given a price snd a customer class'''
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


# find the best joint bid and price strategy for all the customer classes
for i in range(1, 4):
    print(get_best_bid_and_price(bids, prices, i))

# find the best joint bid and price strategy if it is not possible to discrimate between the classes

revenuesmatrix = np.arange(bids.size * prices.size)
revenuesmatrix = revenuesmatrix.reshape(bids.size, prices.size)

for i in range(bids.size):
    for j in range(prices.size):
        for c in range(1, 4):
            revenuesmatrix[i][j] += get_bid_and_price_revenue(bids[i], prices[j], c)

best_bid = bids[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[0]]
best_price = prices[np.unravel_index(np.argmax(revenuesmatrix), revenuesmatrix.shape)[1]]

print("Revenue matrix:")
print(revenuesmatrix)
print("Best bid: " + str(best_bid) + " - Best price: " + str(best_price))
