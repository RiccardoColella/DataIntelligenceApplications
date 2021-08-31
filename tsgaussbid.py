from learner import *
import numpy as np


class TSLearnerGauss(Learner):
    """ Thomson Sampling Learner Class """

    def __init__(self, n_arms):
        """
        Initialize the Thompson Sampling Learner class with number of arms, arms, sigma, expected mean.
        :param n_arms:
        """

        super().__init__(n_arms)  # supercharge init from the Learner

        # Assignments and Initializations
        self.n_arms = n_arms
        self.sigma = 5
        self.tau = [10] * n_arms
        self.mu = [800] * n_arms
        self.last30dayschoice = []
        self.delayedreward = []
        self.rewards_per_arm = np.zeros(n_arms)
        self.percentage=0.2

    def pull_arm(self):
        """
        Pulls the current arm with the given budget and returns it.
        :return: The index of the pulled arm
        """
        #initializing
        idx=-1
        index=-2
        m = self.mu
        ta=self.tau

        if self.t<10:
            idx = np.random.randint(0,10)
        else:
            while idx != index:
                mean = np.random.normal(m[:],ta[:])
                index = np.argmax(np.random.normal(mean[:], self.sigma))
                if np.quantile(self.rewards_per_arm[index], self.percentage)>=0:
                    idx = index
                else:
                    m=m.delete(index)
                    ta=ta.delete(index)

        return idx

    def update_observations(self, pulled_arm, reward, delayedr):
        """
        Updates with the given pulled arm and reward.
        :param pulled_arm: The chosen arm
        :param reward: The assigned reward
        :param delayed_r: The reward from the next 30 days
        :return: NA
        """

        #print('arm, reward: ' + str(pulled_arm) + ', ' + str(sum(delayedr)+reward))

        if self.t <= 30:
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.append(sum(delayedr)+reward)
        else:
            self.last30dayschoice.pop(0)
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward.pop(0)
            self.delayedreward.append(sum(delayedr)+reward)

        #print('last30dayschoice: ' + str(self.last30dayschoice))
        #print('delayedreward: ' + str(self.delayedreward))

        if self.t >= 30:
            self.collected_rewards=np.append(self.collected_rewards,self.delayedreward[0])
            self.rewards_per_arm[self.last30dayschoice[0]] += self.collected_rewards[-1]
            self.n_pulled_arms[self.last30dayschoice[0]] += 1

            arm = self.last30dayschoice[0]
            self.mu[arm] = (self.rewards_per_arm[arm] * self.tau[arm] ** 2 + self.sigma ** 2 * self.mu[arm]) / (self.n_pulled_arms[arm] * self.tau[arm] ** 2 + self.sigma ** 2)
            self.tau[arm] = (self.tau[arm] * self.sigma) ** 2 / (self.n_pulled_arms[arm] * self.tau[arm] ** 2 + self.sigma ** 2)

        self.t += 1

        #print("Collected rewards:")
        #print(self.collected_rewards)

        #print('------')