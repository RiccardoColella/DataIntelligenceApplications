from learner import Learner
import numpy as np


class ucb1_learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
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
            upper_bound = self.empirical_means + self.confidence
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])

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
            self.last30dayschoice = self.last30dayschoice.pop(0)
            self.last30dayschoice.append(pulled_arm)
            self.delayedreward = self.delayedreward.pop(0)
            self.delayedreward.append(delayedr)

        self.t += 1
        self.rewards_per_arm[pulled_arm] = self.rewards_per_arm[pulled_arm] + reward
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.n_pulled_arms[pulled_arm] += 1

        for i in range(len(delayedr)):
            if i < 2:
                continue
            self.collected_rewards[-i - 2] += self.delayedreward[-i - 1][0]
            self.rewards_per_arm[self.last30dayschoice[-i - 1]] += self.delayedreward[-i - 1][0]
            self.delayedreward.pop([-i - 1][0])

        for a in range(self.n_arms):
            self.empirical_means[a] = self.rewards_per_arm[a]/self.n_pulled_arms[a]
            if self.n_pulled_arms > 0:
                self.confidence[a] = (2 * np.log(self.t) / self.n_pulled_arms) ** 0.5
