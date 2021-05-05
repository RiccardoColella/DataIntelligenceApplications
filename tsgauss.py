from learner import Learner
import numpy as np
import random

class tsgauss_learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.dev = np.zeros(n_arms)
        self.last30dayschoice = []
        self.delayedreward = [[]]
        self.rewards_per_arm = np.zeros(n_arms)

    def pull_arm(self):
        """ This function return the arm to be pulled at the next round
        reminder: always call this function and then call update_observations
        :return: the arm to be pulled at the next round
        """

        if self.t < self.n_arms:
            arm = self.t

        else:
            sampled_values=[]
            for i in range(self.n_arms):
                sampled_values.append(int(np.random.normal(loc=self.empirical_means[i], scale=self.dev[i], size=1)))
                arm=random.choice([i for i, j in enumerate(sampled_values) if j == max(sampled_values)])

        return arm

    def update_observations(self, pulled_arm, reward, delayedr):
        """ This function updates the learner's parameter after a arm is pulled
        :param pulled_arm: The arm pulled
        :param reward: The reward associated to the pulled arm
        :param delayedr: The future 30 rewards (delayed rewards)
        :return: none
        """

        if self.t < 30:
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.append(delayedr)
        else:
            self.last30dayschoice.pop(0)
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.pop(0)
            self.delayedreward.append(delayedr)

        self.t += 1
        self.rewards_per_arm[pulled_arm] = self.rewards_per_arm[pulled_arm] + reward
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.n_pulled_arms[pulled_arm] += 1

        if self.t>1:
            for i in range(len(self.delayedreward)-1):
                self.collected_rewards[-i-2] += self.delayedreward[-i-1][0]
                self.rewards_per_arm[self.last30dayschoice[-i-1]] += self.delayedreward[-i-1][0]
                self.delayedreward[-i-1].pop(0)
        else:
            #just remove the empty list at the beginning
            self.delayedreward.pop(0)

        for a in range(self.n_arms):
            if self.n_pulled_arms[a]>0:
                self.empirical_means[a]=self.rewards_per_arm[a]/self.n_pulled_arms[a]
                self.dev[a] = np.std(self.rewards_per_arm[a])
