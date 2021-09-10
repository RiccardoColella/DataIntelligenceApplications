'''
a function tha returns instantaneous and cumulative regret,
given the best possible reward and collected_rewards
'''

import numpy as np

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
