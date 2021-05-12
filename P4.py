from environment import Environment
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from tsgaussp4 import TSLearnerGauss

env = Environment()

T = 365

prices = np.linspace(1, 10, num=10)
bids = [0.7]

tsgauss_learner = TSLearnerGauss(len(prices))

vector_daily_price_ts = []
vector_daily_revenue_ts = []

for t in range(T):
    print("Iteration day: " + str(t))
    # Get new users in the day t and their costs
    [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
    new_users = [new_user_1, new_user_2, new_user_3]
    [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
    cost = [cost1, cost2, cost3]

    # Get the total cost
    total_cost = 0
    for i in range(len(new_users)):
        total_cost += new_users[i] * cost[i]

    #per i primi 50 giorni non splittiamo di sicuro lol
    if t<50:
        daily_arm_ts = tsgauss_learner.pull_arm()
        daily_price_ts = prices[daily_arm_ts]
        vector_daily_price_ts.append(daily_price_ts)
    #capiamo se conviene splittare o no (panic)
    else:
        
