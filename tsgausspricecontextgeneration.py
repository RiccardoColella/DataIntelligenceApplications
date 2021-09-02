from learner import *
import numpy as np


class TSLearnerGauss(Learner):
    """ Thomson Sampling Learner Class """
    def __init__(self, n_arms, collected_rewards, mu, tau, sigma0, last30dayschoice, delayedreward, rewards_per_arm, t):
        """
        Initialize the Thompson Sampling Learner class with number of arms, arms, sigma, expected mean.
        :param n_arms:
        """

        super().__init__(n_arms)  # supercharge init from the Learner

        # Assignments and Initializations
        self.n_arms = n_arms
        self.sigma = sigma0
        self.tau = tau
        self.mu = mu
        self.collected_rewards = collected_rewards
        self.last30dayschoice = last30dayschoice
        self.delayedreward = delayedreward
        self.rewards_per_arm = rewards_per_arm
        self.t = t

    def pull_arm(self):
        """
        Pulls the current arm with the given budget and returns it.
        :return: The index of the pulled arm
        """
        if self.t <= 28:
            return(self.t%10)

        mean = np.random.normal(self.mu[:],self.tau[:])

        idx = np.argmax(np.random.normal(mean[:], self.sigma))
        return idx

    def update_observations(self, pulled_arm, reward, delayedr):
        """
        Updates with the given pulled arm and reward.
        :param pulled_arm: The chosen arm
        :param reward: The assigned reward
        :param delayed_r: The reward from the next 30 days
        :return: NA
        """
        if self.t <= 30:
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.append(sum(delayedr)+reward)
        else:
            self.last30dayschoice.pop(0)
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.pop(0)
            self.delayedreward.append(sum(delayedr)+reward)

        if self.t >= 30:
            self.collected_rewards=np.append(self.collected_rewards,self.delayedreward[0])
            self.rewards_per_arm[self.last30dayschoice[0]] += self.collected_rewards[-1]
            self.n_pulled_arms[self.last30dayschoice[0]] += 1

            arm = self.last30dayschoice[0]
            self.mu[arm] = (self.rewards_per_arm[arm] * self.tau[arm] ** 2 + self.sigma ** 2 * self.mu[arm]) / (self.n_pulled_arms[arm] * self.tau[arm] ** 2 + self.sigma ** 2)
            self.tau[arm] = (self.tau[arm] * self.sigma) ** 2 / (self.n_pulled_arms[arm] * self.tau[arm] ** 2 + self.sigma ** 2)

        self.t += 1
