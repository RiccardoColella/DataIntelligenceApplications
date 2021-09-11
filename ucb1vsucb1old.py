# the following lines just add verbose option and others command line options

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")

# how many executions:
parser.add_argument('-n', help="set number of iteration", default = 1)
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

from operator import add

import numpy as np

from plotutilities import plot

from environment import Environment
from ucb1 import UCB1Learner
from ucb1old import UCB1Learnerold
from regretcalculator import regret_calculator
from P1utilities import get_best_bid_price_possible_reward

env = Environment()

bids = env.bids
prices = env.prices

bids, best_daily_price, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
bids = [bids]

# day of algorithm execution
T = 395

def iterate_days(results_queue, idx=0):
    """
    Execute the algorithm at the given day. Function required for parallel programming
    :param results_queue: queue of previous results
    :param idx: execution identifier, allows to recognize the iteration number
    :return: nothing. The results are pushed into the queue
    """
    # Declaration of learners and results' vectors
    ucb1_learner = UCB1Learner(len(prices))
    ucb1_old_learner = UCB1Learnerold(len(prices))
    vector_daily_price_ucb1_loc = []
    vector_daily_price_ucb1_old_loc = []
    print('Starting execution ' + str(idx))
    # For every day:
    for t in range(T):
        if t % 20 == 0:
            log("Iteration day: {:3d} - execution: {:3d}".format(t, idx))

        # Get new users in the day t and their costs
        [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
        new_users = [new_user_1, new_user_2, new_user_3]
        [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
        cost = [cost1, cost2, cost3]

        # Get the total cost
        total_cost = 0
        for user in range(len(new_users)):
            total_cost += new_users[user] * cost[user]

        # Choose the arm and thus the price for UCB1
        daily_arm_ucb1 = ucb1_learner.pull_arm()
        daily_price_ucb1 = prices[daily_arm_ucb1]
        vector_daily_price_ucb1_loc.append(daily_price_ucb1)

        # Choose the arm and thus the price for ucb1_old
        daily_arm_ucb1_old = ucb1_old_learner.pull_arm()
        daily_price_ucb1_old = prices[daily_arm_ucb1_old]
        vector_daily_price_ucb1_old_loc.append(daily_price_ucb1_old)

        # Calculate the number of bought items
        daily_bought_items_per_class_ucb1 = [0, 0, 0]
        daily_bought_items_per_class_ucb1_old = [0, 0, 0]

        for user in range(len(new_users)):
            for c in range(new_users[user]):
                daily_bought_items_per_class_ucb1[user] += env.buy(daily_price_ucb1, user + 1)
                daily_bought_items_per_class_ucb1_old[user] += env.buy(daily_price_ucb1_old, user + 1)

        # Sum up the n. of bought items
        daily_bought_items_ucb1 = sum(daily_bought_items_per_class_ucb1)
        daily_bought_items_ucb1_old = sum(daily_bought_items_per_class_ucb1_old)

        # Calculate the revenue
        daily_revenue_ucb1 = daily_bought_items_ucb1 * env.get_margin(daily_price_ucb1) - total_cost
        daily_revenue_ucb1_old = daily_bought_items_ucb1_old * env.get_margin(daily_price_ucb1_old) - total_cost

        # Get delayed rewards UCB1
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ucb1[user - 1], daily_price_ucb1,
                                                            user)))

        ucb1_learner.update_observations(daily_arm_ucb1, daily_revenue_ucb1, next_30_days)

        # Get delayed rewards UCB1 old
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ucb1_old[user - 1], daily_price_ucb1_old,
                                                            user)))

        ucb1_old_learner.update_observations(daily_arm_ucb1_old, daily_revenue_ucb1_old, next_30_days)

    print('Ending execution ' + str(idx))

    # put results in the given queue
    results_queue.put((ucb1_learner.collected_rewards, ucb1_old_learner.collected_rewards, vector_daily_price_ucb1_loc, vector_daily_price_ucb1_old_loc))

def to_np_arr_and_then_mean(list_of_lists):
    """
    Mean of every value of the list, based on the index
    :param list_of_lists: list containing the results for every day in a list for every iteration
    :return: an array of the mean based on values' index
    """
    # print(list_of_lists)
    np_arr = np.array(list_of_lists)
    return np_arr.mean(axis=0)

if __name__ == '__main__':

    log('N = ' + str(N))

    collected_rewards_ucb1 = [] * N
    collected_rewards_ucb1_old = [] * N
    vector_daily_price_ucb1 = [] * N
    vector_daily_price_ucb1_old = [] * N

    # Multiprocessing initializations
    processes = []
    results = [] * N
    m = multiprocessing.Manager()
    q = m.Queue()
    # Start the execution
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
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
        collected_rewards_ucb1.insert(i, results[i][0])
        collected_rewards_ucb1_old.insert(i, results[i][1])
        vector_daily_price_ucb1.insert(i, results[i][2])
        vector_daily_price_ucb1_old.insert(i, results[i][3])

    # calculate the mean values
    mean_collected_rewards_ucb1 = to_np_arr_and_then_mean(collected_rewards_ucb1)
    mean_collected_rewards_ucb1_old = to_np_arr_and_then_mean(collected_rewards_ucb1_old)
    mean_vector_daily_price_ucb1 = to_np_arr_and_then_mean(vector_daily_price_ucb1)
    mean_vector_daily_price_ucb1_old = to_np_arr_and_then_mean(vector_daily_price_ucb1_old)

    cwd = os.getcwd()
    print("Current working directory: " + cwd)
    plots_folder = os.path.join(cwd, "plotsucb1vsucb1old")
    print("Plots folder: " + plots_folder)

    # Plot collected rewards
    plot([mean_collected_rewards_ucb1, mean_collected_rewards_ucb1_old, [best_possible_reward for i in range(T)]],
            ['UCB1', 'UCB1old', 'clairvoyant'], 'Collected reward', plots_folder, 2)

    # Plot daily prices

    plot([mean_vector_daily_price_ucb1, mean_vector_daily_price_ucb1_old, [best_daily_price for i in range(T)]],
            ['UCB1', 'UCB1old', 'clairvoyant'], 'Daily prices', plots_folder, 2)

    #calculate and plot regret
    instantaneous_regret_ucb1, cumulative_regret_ucb1 = regret_calculator(best_possible_reward, mean_collected_rewards_ucb1)

    instantaneous_regret_ucb1_old, cumulative_regret_ucb1_old = regret_calculator(best_possible_reward, mean_collected_rewards_ucb1_old)

    plot([instantaneous_regret_ucb1, instantaneous_regret_ucb1_old],
            ['instantaneous_regret_ucb1', 'instantaneous_regret_ucb1_old'], 'Instantaneous regret comparison', plots_folder, 2)

    plot([cumulative_regret_ucb1, cumulative_regret_ucb1_old],
            ['cumulative_regret_ucb1', 'cumulative_regret_ucb1_old'], 'Cumulative regret comparison', plots_folder, 2)
