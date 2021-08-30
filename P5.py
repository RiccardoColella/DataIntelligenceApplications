import numpy as np

from environment import Environment
from tsgaussp5 import TSLearnerGauss

T = 365

prices = [8]
bids = np.linspace(0.1, 1, num=10)

for t in range(T):
    if t % 20 == 0:
        print("Iteration day: {:3d} - execution: {:3d}".format(t, idx))
