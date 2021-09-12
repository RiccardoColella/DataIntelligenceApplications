'''
a function tha returns instantaneous and cumulative regret,
given the best possible reward and collected_rewards
'''

import numpy as np
from environment import Environment
from P1utilities import get_bid_and_price_revenue

def regret_calculator(best, rewards):
    length=len(list(rewards))
    instantaneous_regret = np.array([best for i in range(length)]) - np.array(rewards)

    cumulative_regret = np.array([best * i for i in range(length)]) - np.cumsum(rewards)

    return instantaneous_regret, cumulative_regret

def regret_per_class_calculator(best_list, rewards):
    instantaneous_regret=[]
    cumulative_regret=[]
    for i in range(len(best_list)):
        a, b = regret_calculator(best_list[i], rewards[i])
        instantaneous_regret.append(a)
        cumulative_regret.append(b)

    return instantaneous_regret, cumulative_regret

def regret_upper_bound_ucb1(bid, prices, best_price, best_possible_reward, T):

    delta = [best_possible_reward - sum([get_bid_and_price_revenue(bid, prices[i], a) for a in range(1,4)]) for i in range(len(prices))]

    upper_bound = [ sum( [0 if prices[i]==best_price else 8*np.log(t)/delta[i]+3.5*delta[i] + delta[i]*3  for i in range(len(delta))] ) for t in range(1,T)]
    #upper_bound = [ sum( [0 if prices[i]==best_price else 8*np.log(t)/delta[i] for i in range(len(delta))] ) + sum([0 if prices[i]==best_price else 4.29 * delta[i] for i in range(len(delta))]) for t in range(1,T)]

    return upper_bound
