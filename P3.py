# the following lines just add verbose option and others command line options

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
#argument to plot learned curve
parser.add_argument('-p', help="plot learned curve, automatically set n=1", action="store_true")

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

plot_l_t =  parser.parse_args().p

# now the real code begins

import multiprocessing
import os
from matplotlib import pyplot

from operator import add

import numpy as np

from plotutilities import plot
from plotutilities import plot_learned_curve

from ucb1 import UCB1Learner
from environment import Environment
from tsgaussprice import TSLearnerGauss
from P1utilities import get_best_bid_price_possible_reward
from P1utilities import get_bid_and_price_revenue
from regretcalculator import *

env = Environment()

bids = env.bids
prices = env.prices

bids, best_daily_price, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
bids = [bids]

# day of algorithm execution
T = 395

if plot_l_t== True:
    N=1
    cwd = os.getcwd()
    plots_folder = os.path.join(cwd, "plotsp3")
    plots_folder = os.path.join(plots_folder, "learnedcurve")
    real = [get_bid_and_price_revenue(bids[0], prices[i], 1) + get_bid_and_price_revenue(bids[0], prices[i], 2) + get_bid_and_price_revenue(bids[0], prices[i], 3)
            for i in range(len(prices))]

def iterate_days(results_queue, idx=0):
    """
    Execute the algorithm at the given day. Function required for parallel programming
    :param results_queue: queue of previous results
    :param idx: execution identifier, allows to recognize the iteration number
    :return: nothing. The results are pushed into the queue
    """
    # Declaration of learners and results' vectors
    ucb1_learner = UCB1Learner(len(prices))
    tsgauss_learner = TSLearnerGauss(len(prices))
    vector_daily_price_ucb1_loc = []
    vector_daily_revenue_ucb1_loc = []
    vector_daily_price_ts_loc = []
    vector_daily_revenue_ts_loc = []

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

        # Choose the arm and thus the price for Thomson Sampling
        daily_arm_ts = tsgauss_learner.pull_arm()
        daily_price_ts = prices[daily_arm_ts]
        vector_daily_price_ts_loc.append(daily_price_ts)

        # Calculate the number of bought items
        daily_bought_items_per_class_ucb1 = [0, 0, 0]
        daily_bought_items_per_class_ts = [0, 0, 0]

        for user in range(len(new_users)):
            for c in range(new_users[user]):
                daily_bought_items_per_class_ucb1[user] += env.buy(daily_price_ucb1, user + 1)
                daily_bought_items_per_class_ts[user] += env.buy(daily_price_ts, user + 1)

        # Sum up the n. of bought items
        daily_bought_items_ucb1 = sum(daily_bought_items_per_class_ucb1)
        daily_bought_items_ts = sum(daily_bought_items_per_class_ts)

        # Calculate the revenue
        daily_revenue_ucb1 = daily_bought_items_ucb1 * env.get_margin(daily_price_ucb1) - total_cost
        daily_revenue_ts = daily_bought_items_ts * env.get_margin(daily_price_ts) - total_cost

        # Add to the vector the daily revenue
        vector_daily_revenue_ucb1_loc.append(daily_revenue_ucb1)
        vector_daily_revenue_ts_loc.append(daily_revenue_ts)

        # Get delayed rewards
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ucb1[user - 1], daily_price_ucb1,
                                                            user)))

        ucb1_learner.update_observations(daily_arm_ucb1, daily_revenue_ucb1, next_30_days)

        # Get delayed rewards
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ts[user - 1], daily_price_ts,
                                                            user)))
        tsgauss_learner.update_observations(daily_arm_ts, daily_revenue_ts, next_30_days)

        if plot_l_t == True and t>=29:
            plot_learned_curve(tsgauss_learner.mu, tsgauss_learner.tau, real, tsgauss_learner.n_pulled_arms, plots_folder, t)

    print('Ending execution ' + str(idx))

    # put results in the given queue
    results_queue.put((ucb1_learner.collected_rewards, tsgauss_learner.collected_rewards, vector_daily_price_ucb1_loc,
                       vector_daily_revenue_ucb1_loc, vector_daily_price_ts_loc, vector_daily_revenue_ts_loc, tsgauss_learner.mu, tsgauss_learner.tau, tsgauss_learner.n_pulled_arms))


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
    collected_rewards_ts = [] * N
    vector_daily_price_ucb1 = [] * N
    vector_daily_revenue_ucb1 = [] * N
    vector_daily_price_ts = [] * N
    vector_daily_revenue_ts = [] * N
    vector_mu = [] * N
    vector_tau = [] * N
    vector_n_pulled_arms = [] * N

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
        collected_rewards_ts.insert(i, results[i][1])
        vector_daily_price_ucb1.insert(i, results[i][2])
        vector_daily_revenue_ucb1.insert(i, results[i][3])
        vector_daily_price_ts.insert(i, results[i][4])
        vector_daily_revenue_ts.insert(i, results[i][5])
        vector_mu.insert(i, results[i][6])
        vector_tau.insert(i, results[i][7])
        vector_n_pulled_arms.insert(i, results[i][8])

    # calculate the mean values
    mean_collected_rewards_ucb1 = to_np_arr_and_then_mean(collected_rewards_ucb1)
    mean_collected_rewards_ts = to_np_arr_and_then_mean(collected_rewards_ts)
    mean_vector_daily_price_ucb1 = to_np_arr_and_then_mean(vector_daily_price_ucb1)
    mean_vector_daily_revenue_ucb1 = to_np_arr_and_then_mean(vector_daily_revenue_ucb1)
    mean_vector_daily_price_ts = to_np_arr_and_then_mean(vector_daily_price_ts)
    mean_vector_daily_revenue_ts = to_np_arr_and_then_mean(vector_daily_revenue_ts)
    mean_mu = to_np_arr_and_then_mean(vector_mu)
    mean_tau = to_np_arr_and_then_mean(vector_tau)
    mean_n_pulled_arms = to_np_arr_and_then_mean(vector_n_pulled_arms)

    cwd = os.getcwd()
    print("Current working directory: " + cwd)
    plots_folder = os.path.join(cwd, "plotsp3")
    print("Plots folder: " + plots_folder)

    # Plot collected rewards

    plot([mean_collected_rewards_ucb1, mean_collected_rewards_ts, [best_possible_reward for i in range(T)]],
            ['UCB1', 'TS', 'Clairvoyant'], 'Collected reward', plots_folder, 3)

    # Plot daily prices

    plot([mean_vector_daily_price_ucb1, mean_vector_daily_price_ts, [best_daily_price for i in range(T)]],
            ['UCB1', 'TS', 'Clairvoyant'], 'Daily prices', plots_folder, 3, yticks=prices)

    # Plot UCB1 price and revenue comparison

    plot([mean_collected_rewards_ucb1, [best_possible_reward for i in range(T)], [i * 100 for i in mean_vector_daily_price_ucb1]],
            ['Collected reward', 'Clairvoyant', 'Price * 100'], 'UCB1 price and revenue comparison', plots_folder, 1)

    # Plot TS price and revenue comparison

    plot([mean_collected_rewards_ts, [best_possible_reward for i in range(T)], [i * 100 for i in mean_vector_daily_price_ts]],
            ['Collected reward', 'Clairvoyant', 'Price * 100'], 'TS price and revenue comparison', plots_folder, 1)

    # Plot UCB1 and TS revenue
    plot([mean_collected_rewards_ucb1, [best_possible_reward for i in range(T)]],
    ['Collected reward', 'Clairvoyant'], 'UCB1 revenue', plots_folder, 5, yvalues=[0, best_possible_reward*1.1])

    plot([mean_collected_rewards_ts, [best_possible_reward for i in range(T)]],
     ['Collected reward', 'Clairvoyant'], 'TS revenue', plots_folder, 4, yvalues=[0, best_possible_reward*1.1])

    #calculate and plot regret
    instantaneous_regret_ucb1, cumulative_regret_ucb1 = regret_calculator(best_possible_reward, mean_collected_rewards_ucb1)

    instantaneous_regret_ts, cumulative_regret_ts = regret_calculator(best_possible_reward, mean_collected_rewards_ts)

    #upper_bound_ucb1 = regret_upper_bound_ucb1(bids[0], prices, best_daily_price, best_possible_reward, T)

    plot([instantaneous_regret_ucb1, instantaneous_regret_ts],
            ['Instantaneous regret ucb1', 'Instantaneous regret ts'], 'Instantaneous regret comparison', plots_folder, 3)

    plot([cumulative_regret_ucb1, cumulative_regret_ts],
            ['Cumulative regret ucb1', 'Cumulative regret ts'], 'Cumulative regret comparison', plots_folder, 3)

    plot([cumulative_regret_ucb1, regret_upper_bound_ucb1(bids[0], prices, best_daily_price, best_possible_reward, T)],
            ['Cumulative regret ucb1', 'Upper bound'], 'Cumulative regret ucb1', plots_folder, 5)

    real = [get_bid_and_price_revenue(bids[0], prices[i], 1) + get_bid_and_price_revenue(bids[0], prices[i], 2) + get_bid_and_price_revenue(bids[0], prices[i], 3)
            for i in range(len(prices))]

    plot_learned_curve(mean_mu, mean_tau, real, mean_n_pulled_arms, plots_folder)
