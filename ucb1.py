from learner import Learner
import numpy as np

class ucb1_learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means=np.zeros(n_arms)
        self.confidence=np.zeros(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t

        else:
            upper_bound = self.empirical_means + self.confidence
            arm = np.random.choice(np.where(upper_bound==upper_bound.max())[0])

        return arm

    def update_observations(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(len(self.rewards_per_arm[pulled_arm])-1)+reward)/len(self.rewards_per_arm[pulled_arm])
        for a in range(self.n_arms):
            if len(self.rewards_per_arm[a])>0:
                self.confidence[a] = (2*np.log(self.t)/len(self.rewards_per_arm[a]))**0.5
