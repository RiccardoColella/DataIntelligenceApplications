from ucb1 import ucb1_learner
from environment import Environment
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from tsgauss import tsgauss_learner

env = Environment()

T=395

prices=np.linspace(1,10,num=10)
bids=[1]

#mettere P1 come funzione per poter usare ottimo per calcolare il regret

ucb1_learner=ucb1_learner(len(prices))
tsgauss_learner=tsgauss_learner(len(prices))
vector_daily_price_ucb1=[]
vector_daily_revenue_ucb1=[]
vector_daily_price_ts=[]
vector_daily_revenue_ts=[]

for t in range(T):
    [new_user_1,new_user_2,new_user_3] = env.get_all_new_users_daily(bids[0])
    new_users=[new_user_1,new_user_2,new_user_3]
    [cost1,cost2,cost3] = env.get_all_cost_per_click(bids[0])
    cost=[cost1,cost2,cost3]

    total_cost=0
    for i in range(0,3):
        total_cost+=new_users[i]*cost[i]

    daily_arm_ucb1=ucb1_learner.pull_arm()
    daily_price_ucb1 = prices[daily_arm_ucb1]
    vector_daily_price_ucb1.append(daily_price_ucb1)

    daily_arm_ts=tsgauss_learner.pull_arm()
    daily_price_ts = prices[daily_arm_ts]
    vector_daily_price_ts.append(daily_price_ts)

    daily_bought_items_perclass_ucb1=[0, 0, 0]
    daily_bought_items_perclass_ts=[0, 0, 0]

    for i in range(0,3):
        for c in range(new_users[i]):
            daily_bought_items_perclass_ucb1[i] += env.buy(daily_price_ucb1,i)
            daily_bought_items_perclass_ts[i] += env.buy(daily_price_ts,i)

    daily_bought_items_ucb1 = sum(daily_bought_items_perclass_ucb1)
    daily_bought_items_ts=sum(daily_bought_items_perclass_ts)


    daily_revenue_ucb1=daily_bought_items_ucb1*env.get_margin(daily_price_ucb1)-total_cost
    daily_revenue_ts=daily_bought_items_ts*env.get_margin(daily_price_ts)-total_cost

    vector_daily_revenue_ucb1.append(daily_revenue_ucb1)
    vector_daily_revenue_ts.append(daily_revenue_ts)

    next_30_days=[0]*30
    for i in range (1,4):
        next_30_days=list( map(add, next_30_days, env.get_next_30_days( daily_bought_items_perclass_ucb1[i-1], daily_price_ucb1, i)) )
        '''pointwise list sum'''
    ucb1_learner.update_observations(daily_arm_ucb1,daily_revenue_ucb1,next_30_days)


    next_30_days=[0]*30
    for i in range (1,4):
        next_30_days=list( map(add, next_30_days, env.get_next_30_days( daily_bought_items_perclass_ts[i-1], daily_price_ts, i)) )
    tsgauss_learner.update_observations(daily_arm_ts,daily_revenue_ts,next_30_days)


plt.figure()
plt.plot(ucb1_learner.collected_rewards)
plt.plot(tsgauss_learner.collected_rewards)
plt.xlim([0, 365])
plt.legend(['UCB1', 'TS'])
plt.title('Collected reward')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(vector_daily_price_ucb1)
plt.plot(vector_daily_price_ts)
plt.xlim([0, 365])
plt.legend(['UCB1', 'TS'])
plt.title('daily prices')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(vector_daily_revenue_ucb1)
plt.plot(vector_daily_revenue_ts)
plt.xlim([0, 365])
plt.legend(['UCB1 ',' TS '])
plt.title('daily revenue')
plt.xlabel('Days')
plt.show()

plt.figure()
plt.plot(ucb1_learner.collected_rewards)
plt.plot([i * 1000 for i in vector_daily_price_ucb1])
plt.xlim([0, 365])
plt.title('UCB1 confronto prezzo revenue')
plt.xlabel('Days')
plt.show()


plt.figure()
plt.plot(tsgauss_learner.collected_rewards)
plt.plot([i * 1000 for i in vector_daily_price_ts])
plt.xlim([0, 365])
plt.title('TS confronto prezzo revenue')
plt.xlabel('Days')
plt.show()
