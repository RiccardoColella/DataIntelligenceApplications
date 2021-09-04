# the following lines just add verbose option and others command line options

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")

# how many executions:
parser.add_argument('-n', help="set number of iteration", default = 200)
N = int(parser.parse_args().n)

verbose = parser.parse_args().verbose

if verbose:
    def log(argument):
        print(argument)
else:
    def log(argument):
        return

# now the real code begins

import multiprocessing
import os

import numpy as np

from matplotlib import pyplot
from plotutilities import plot

from environment import Environment
from tsgaussbid import TSLearnerGauss
from P1utilities import get_best_bid_price_possible_reward

from operator import add

env = Environment()

# day of algorithm execution
T = 395

#bids and prices range
prices = np.linspace(1, 10, num=10)
bids = np.linspace(0.1, 1, num=10)

best_daily_bid, prices, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
prices = [prices]

def iterate_days(results_queue, idx=0):
    """
    Execute the algorithm at the given day. Function required for parallel programming
    :param results_queue: queue of previous results
    :param idx: execution identifier, allows to recognize the iteration number
    :return: nothing. The results are pushed into the queue
    """
    # Declaration of learners and results' vectors
    tsgauss_learner = TSLearnerGauss(len(bids))

    vector_daily_bids_loc = []
    vector_daily_user_per_class_loc = []

    print('Starting execution ' + str(idx))

    # For every day:
    for t in range(T):
        if t % 20 == 0:
            log("Iteration day: {:3d} - execution: {:3d}".format(t, idx))

        #choose daily arm
        daily_arm = tsgauss_learner.pull_arm()
        daily_bid = bids[daily_arm]
        vector_daily_bids_loc.append(daily_bid)

        # Get new users in the day t and their costs
        [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(daily_bid)
        new_users = [new_user_1, new_user_2, new_user_3]

        vector_daily_user_per_class_loc.append(new_users)

        [cost1, cost2, cost3] = env.get_all_cost_per_click(daily_bid)
        cost = [cost1, cost2, cost3]

        # Get the total cost
        total_cost = 0
        for user in range(len(new_users)):
            total_cost += new_users[user] * cost[user]

        # Calculate the number of bought items
        daily_bought_items_per_class = [0, 0, 0]

        for user in range(len(new_users)):
            for c in range(new_users[user]):
                daily_bought_items_per_class[user] += env.buy(prices[0], user + 1)

        # Sum up the n. of bought items
        daily_bought_items = sum(daily_bought_items_per_class)

        # Calculate the revenue
        daily_revenue = daily_bought_items * env.get_margin(prices[0]) - total_cost

        # Get delayed rewards
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class[user - 1], prices[0],user)))

        #update observations
        tsgauss_learner.update_observations(daily_arm, daily_revenue, next_30_days)

    # put results in the given queue
    results_queue.put((vector_daily_bids_loc, vector_daily_user_per_class_loc, tsgauss_learner.collected_rewards))

    print('Ending execution ' + str(idx))

def to_np_arr_and_then_mean(list_of_lists):
    """
    Mean of every value of the list, based on the index
    :param list_of_lists: list containing the results for every day in a list for every iteration
    :return: an array of the mean based on values' index
    """
    # print(list_of_lists)
    np_arr = np.array(list_of_lists)
    return np_arr.mean(axis=0)

def to_np_arr_and_then_mean_per_class(list_of_lists_of_lists):
    '''like to_np_arr_and_then_mean, but divided per class'''

    final = [ [ [ 0 for i in range(T) ] for j in range(N) ] for k in range(3)]

    #from N*T*3 to 3*N*T
    for i in range(N):
        for j in range(T):
            for k in range(3):
                final[k][i][j] = list_of_lists_of_lists[i][j][k]

    mean = []

    for i in range(3):
        mean.append(to_np_arr_and_then_mean(final[i]))

    return mean


if __name__ == '__main__':
    log('N = ' + str(N))

    #initializations of results list
    bids = [] * N
    user_per_class = [] * N
    revenue = [] * N

    # Multiprocessing initializations
    processes = []
    results = [] * N
    m = multiprocessing.Manager()
    q = m.Queue()
    # Start the execution
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)
    multi_results = [pool.apply_async(iterate_days, args=(q, i,)) for i in range(N)]

    # collect the results
    for p in multi_results:
        ret = q.get()
        results.append(ret)
    # wait for the processes to end
    for i in range(len(processes)):
        processes[i].join()

    # merge the results in a list of lists
    for i in range(len(results)):
        bids.insert(i, results[i][0])
        user_per_class.insert(i, results[i][1])
        revenue.insert(i, results[i][2])

    # calculate the mean values
    mean_bids = to_np_arr_and_then_mean(bids)

    mean_user_per_class = to_np_arr_and_then_mean_per_class(user_per_class)

    mean_revenue = to_np_arr_and_then_mean(revenue)

    cwd = os.getcwd()
    print("Current working directory: " + cwd)
    plots_folder = os.path.join(cwd, "plotsp5")
    print("Plots folder: " + plots_folder)

    # Plot mean bids

    plot([mean_bids,[best_daily_bid for i in range(T)]],
            ['Bids'], 'Bids', plots_folder)

    # Plot mean user per class

    plot(mean_user_per_class,
            ['User of class 1', 'User of class 2', 'User of class 3'], 'User per class', plots_folder)

    # Plot mean revenue

    plot([mean_revenue, [best_possible_reward for i in range(T)]],
            ['Revenue'], 'Revenue', plots_folder)
