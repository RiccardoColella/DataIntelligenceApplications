from learner import Learner
import numpy as np


class UCB1Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.last30dayschoice = []
        self.delayedreward = []
        self.rewards_per_arm = np.zeros(n_arms)

    def pull_arm(self):
        """ This function return the arm to be pulled at the next round
        reminder: always call this function and then call update_observations
        :return: the arm to be pulled at the next round
        """

        if self.t <= 30:
            arm = self.t % self.n_arms

        else:
            upper_bound = self.empirical_means + self.confidence
            #print('upper_bound: ' + str(upper_bound))
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
            #print('arm:' + str(arm))

        return arm

    def update_observations(self, pulled_arm, reward, delayedr):
        """ This function updates the learner's parameter after a arm is pulled
        :param pulled_arm: The arm pulled
        :param reward: The reward associated to the pulled arm
        :param delayedr: The future 30 rewards (delayed rewards)
        :return: none
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

        # print(self.collected_rewards)

        self.t += 1

        for a in range(self.n_arms):
            if self.n_pulled_arms[a] > 0:
                self.empirical_means[a] = self.rewards_per_arm[a] / self.n_pulled_arms[a]
                self.confidence[a] = (2 * np.log(self.t) / self.n_pulled_arms[a]) ** 0.5
