from ucb1 import ucb1_learner
from environment import Environment
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from TSGauss import TSLearnerGauss

env = Environment()

T = 365

prices = np.linspace(1, 10, num=10)
bids = [0.7]

# mettere P1 come funzione per poter usare ottimo per calcolare il regret

ucb1_learner = ucb1_learner(len(prices))
tsgauss_learner = TSLearnerGauss(len(prices))
vector_daily_price_ucb1 = []
vector_daily_revenue_ucb1 = []
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

    # Choose the arm and thus the price for UCB
    daily_arm_ucb1 = ucb1_learner.pull_arm()
    daily_price_ucb1 = prices[daily_arm_ucb1]
    vector_daily_price_ucb1.append(daily_price_ucb1)

    # Choose the arm and thus the price for Thomson Sampling
    daily_arm_ts = tsgauss_learner.pull_arm()
    daily_price_ts = prices[daily_arm_ts]
    vector_daily_price_ts.append(daily_price_ts)

    # Get n bought items
    daily_bought_items_perclass_ucb1 = [0, 0, 0]
    daily_bought_items_perclass_ts = [0, 0, 0]

    # Calculate the number of real bought items
    for i in range(len(new_users)):
        for c in range(new_users[i]):
            daily_bought_items_perclass_ucb1[i] += env.buy(daily_price_ucb1, i)
            daily_bought_items_perclass_ts[i] += env.buy(daily_price_ts, i)

    # Sum up the n. of bought items
    daily_bought_items_ucb1 = sum(daily_bought_items_perclass_ucb1)
    daily_bought_items_ts = sum(daily_bought_items_perclass_ts)

    # Calculate the revenue
    daily_revenue_ucb1 = daily_bought_items_ucb1 * env.get_margin(daily_price_ucb1)-total_cost
    daily_revenue_ts=daily_bought_items_ts*env.get_margin(daily_price_ts)-total_cost

    # Add to the vector the daily revenue
    vector_daily_revenue_ucb1.append(daily_revenue_ucb1)
    vector_daily_revenue_ts.append(daily_revenue_ts)

    # Get delayed rewards
    next_30_days = [0] * 30
    for i in range(1, 4):
        next_30_days = [next_30_days[j] + env.get_next_30_days(daily_bought_items_perclass_ucb1[i - 1], daily_price_ucb1, i)[j] for j in range(len(next_30_days))]
    # pointwise list sum
    ucb1_learner.update_observations(daily_arm_ucb1, daily_revenue_ucb1, next_30_days)

    print("Daily revenue UCB1: " + str(daily_revenue_ucb1 + sum(next_30_days)))
    print("Daily revenue Thomson Sampling: " + str(daily_revenue_ts + sum(next_30_days)))

    next_30_days = [0] * 30
    for i in range(1, 4):
        next_30_days = list(map(add, next_30_days, env.get_next_30_days(daily_bought_items_perclass_ts[i-1], daily_price_ts, i)))
    tsgauss_learner.update_observations(daily_arm_ts, daily_revenue_ts, next_30_days)


plt.figure()
plt.plot(ucb1_learner.collected_rewards)
plt.plot(tsgauss_learner.collected_rewards)
plt.xlim([0, T-30])
plt.legend(['UCB1', 'TS'])
plt.title('Collected reward')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(vector_daily_price_ucb1)
plt.plot(vector_daily_price_ts)
plt.xlim([0, T-30])
plt.legend(['UCB1', 'TS'])
plt.title('daily prices')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(vector_daily_revenue_ucb1)
plt.plot(vector_daily_revenue_ts)
plt.xlim([0, T-30])
plt.legend(['UCB1 ',' TS '])
plt.title('daily revenue')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(ucb1_learner.collected_rewards)
#plt.plot([i * 1000 for i in vector_daily_price_ucb1])
plt.xlim([0, T-30])
plt.title('UCB1 confronto prezzo revenue')
plt.xlabel('Days')
plt.show()


plt.figure()
plt.plot(tsgauss_learner.collected_rewards)
#plt.plot([i * 1000 for i in vector_daily_price_ts])
plt.xlim([0, T-30])
plt.title('TS confronto prezzo revenue')
plt.xlabel('Days')
plt.show()
