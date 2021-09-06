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
