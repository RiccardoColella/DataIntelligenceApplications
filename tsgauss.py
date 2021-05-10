from learner import *
import numpy as np


class TSLearnerGauss(Learner):
    """ Thomson Sampling Learner Class """

    def __init__(self, n_arms, sigma=5):
        """
        Initialize the Thompson Sampling Learner class with number of arms, arms, sigma, expected mean.
        :param n_arms:
        """

        super().__init__(n_arms)  # supercharge init from the Learner

        # Assignments and Initializations
        self.n_arms = n_arms
        self.sigma = sigma
        self.empirical_means = np.zeros(n_arms)
        self.dev = np.zeros(n_arms)
        self.last_30_days_choice = []
        self.delayed_reward = [[]]

    def pull_arm(self):
        """
        Pulls the current arm with the given budget and returns it.
        :return: The index of the pulled arm
        """

        sampled_values = [0] * self.n_arms
        for i in range(self.n_arms):
            sampled_values[i] = np.random.normal(loc=self.empirical_means[i], scale=self.dev[i], size=1)[0]

        return np.argmax(sampled_values)

    def update_observations(self, pulled_arm, reward, delayed_r):
        """
        Updates with the given pulled arm and reward.
        :param pulled_arm: The chosen arm
        :param reward: The assigned reward
        :param delayed_r: The reward from the next 30 days
        :return: NA
        """

        if self.t < 30:
            self.last_30_days_choice.append(pulled_arm)
            self.delayed_reward.append(delayed_r)
        else:
            self.last_30_days_choice.pop(0)
            self.last_30_days_choice.append(pulled_arm)
            self.delayed_reward.pop(0)
            self.delayed_reward.append(delayed_r)

        self.t += 1
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.rewards_per_arm[pulled_arm] = self.rewards_per_arm[pulled_arm] + reward
        self.n_pulled_arms[pulled_arm] += 1

        if self.t > 1:
            for i in range(len(self.delayed_reward)):
                self.collected_rewards[-i-1] += self.delayed_reward[-i - 1][0]
                self.rewards_per_arm[self.last_30_days_choice[-i - 1]] += self.delayed_reward[-i - 1][0]
                self.delayed_reward[-i - 1].pop(0)
        else:
            # just remove the empty list at the beginning
            self.delayed_reward.pop(0)

        self.empirical_means[pulled_arm] = self.dev[pulled_arm] * ((self.empirical_means[pulled_arm] /
                                                                self.dev[pulled_arm]) + reward / self.sigma ** 2)
        self.dev[pulled_arm] = 1 / (1 / self.dev[pulled_arm] + 1 / self.sigma ** 2)
