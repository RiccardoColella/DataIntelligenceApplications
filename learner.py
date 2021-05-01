import numpy as np


class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = []
        self.n_pulled_arms = [0] * n_arms
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        '''this function updates the learner's parameter after a arm is pulled'''
        '''reminder: we overwrite this function everytime we create a specific learner'''

        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
