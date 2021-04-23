from ucb1 import ucb1_learner
from environment import Environment
import numpy as np

env = Environment()

T=365

prices=np.linspace(1,10,num=10)
bids=[1]

#mettere P1 come funzione per poter usare ottimo per calcolare il regret

learner=ucb1_learner(len(prices))

for t in range(T):
    [new_user_1,new_user_2,new_user_3] = env.get_all_new_users_daily(bids[0])
    new_users=[new_user_1,new_user_2,new_user_3]
    [cost1,cost2,cost3] = env.get_all_cost_per_click(bids[0])
    cost=[cost1,cost2,cost3]

    total_cost=0
    for i in range(0,3):
        total_cost+=new_users[i]*cost[i]

    daily_arm=learner.pull_arm()
    daily_price = prices[daily_arm]

    print(daily_price)

    daily_bought_items=0
    for i in range(0,3):
        for c in range(new_users[i]):
            daily_bought_items+=env.buy(daily_price,i)

    daily_revenue=daily_bought_items*env.get_margin(daily_price)-total_cost
    print(daily_revenue)
    learner.update_observations(daily_arm,daily_revenue)
