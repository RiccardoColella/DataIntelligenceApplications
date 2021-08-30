import numpy as np

from environment import Environment
from tsgaussp5 import TSLearnerGauss

T = 365

prices = [8]
bids = np.linspace(0.1, 1, num=10)

## TODO: parallelization like P3

tsgauss_learner = TSLearnerGauss(len(bids))

for t in range(T):
    if t % 20 == 0:
        print("Iteration day: " + str(t))

    ## TODO: call tsgaussp5 to choose the daily_bid
    daily_arm = tsgauss_learner.pull_arm()
    daily_bid = bids[daily_arm]

    # Get new users in the day t and their costs
    [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
    new_users = [new_user_1, new_user_2, new_user_3]

    # Get the total cost
    total_cost = 0
    for user in range(len(new_users)):
        total_cost += new_users[user] * cost[user]

    # Calculate the number of bought items
    daily_bought_items_per_class = [0, 0, 0]

    for user in range(len(new_users)):
        for c in range(new_users[user]):
            daily_bought_items_per_class[user] += env.buy(prices[0], user + 1)

    # Calculate the revenue
    daily_revenue = daily_bought_items * env.get_margin(prices[0]) - total_cost

    # Get delayed rewards
    next_30_days = [0] * 30
    for user in range(1, 4):
        next_30_days = list(
            map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class[user - 1], daily_price,user)))

    #update observations
    tsgauss_learner.update_observations(daily_arm, daily_revenue, next_30_days)
