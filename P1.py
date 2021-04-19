import environment

env = environment.Environment()


def get_best_price(prices, customer_class):
    revenues = []
    for price in prices:
        revenue = env.get_margin(price) * env.get_conversion_rate(price, customer_class)
        revenues.append(revenue)

    max_revenue = max(revenues)
    return prices[revenues.index(max_revenue)]


def get_best_price_common(prices):
    revenues = []
    for price in prices:
        revenue = 0
        revenue += get_best_price(price, 1)
        revenue += get_best_price(price, 2)
        revenue += get_best_price(price, 3)
        revenues.append(revenue)

    max_revenue = max(revenues)
    return prices[revenues.index(max_revenue)]


def get_bid_revenue(bid, price, customer_class):
    cost_per_click = env.get_cost_per_click(bid, customer_class)
    new_users_daily = env.get_new_users_daily(bid, customer_class)
    buy_percentage = env.buy(price, customer_class)
    percent_comebacks = env.get_mean_n_times_comeback(customer_class)

    purchases = new_users_daily * buy_percentage
    n_comebacks = purchases * percent_comebacks
    revenue = new_users_daily * cost_per_click - (purchases + n_comebacks) * env.get_margin(price)
    return revenue


def get_best_bid(bids, price, customer_class):
    revenues = []
    for bid in bids:
        revenues.append(get_bid_revenue(bid, price, customer_class))

    max_revenue = max(revenues)
    return bids[revenues.index(max_revenue)]


def get_best_bid_and_price(bids, prices, customer_class):
    best_price = get_best_price(prices, customer_class)
    best_bid = get_best_bid(bids, best_price, customer_class)
    return best_bid, best_price


def get_best_bid_having_common_price(bids, prices):
    best_price = get_best_price_common(prices)
    best_bid_1 = get_best_bid(bids, best_price, 1)
    best_bid_2 = get_best_bid(bids, best_price, 2)
    best_bid_3 = get_best_bid(bids, best_price, 3)
    return best_bid_1, best_bid_2, best_bid_3
