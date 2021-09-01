import numpy as np

n_arms = 10
sigma = 1
tau = [1] * n_arms
mu = [1867] * n_arms

mean = np.random.normal(mu[:],tau[:])
idx = np.argmax(np.random.normal(mean[:], sigma))
