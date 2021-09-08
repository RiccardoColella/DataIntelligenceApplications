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

from plotutilities import multi_plot

from operator import add

from environment import Environment
from tsgaussprice import TSLearnerGauss as TSLearnerGaussPrices
from tsgaussbid import TSLearnerGauss as TSLearnerGaussBids

env = Environment()

# day of algorithm execution
T = 395
#bids and prices range
bids = env.bids
prices = env.prices

def iterate_days(results_queue, idx=0):
    """
    Execute the algorithm at the given day. Function required for parallel programming
    :param results_queue: queue of previous results
    :param idx: execution identifier, allows to recognize the iteration number
    :return: nothing. The results are pushed into the queue
    """
    # Declaration of learners and results' vectors
    tsgauss_learner_prices = []
    tsgauss_learner_bids = []
    for i in range(3):
        tsgauss_learner_prices.append(TSLearnerGaussPrices(len(prices)))
        tsgauss_learner_bids.append(TSLearnerGaussBids(len(bids)))

    vector_daily_prices_loc = []
    vector_daily_bids_loc = []
    vector_daily_user_per_class_loc = []

    print('Starting execution ' + str(idx))

    # For every day:
    for t in range(T):
        if t % 20 == 0:
            log("Iteration day: {:3d} - execution: {:3d}".format(t, idx))

        #choose daily arm
        #price:
        daily_arm_price = [tsgauss_learner_prices[i].pull_arm() for i in range(3)]
        daily_price = [prices[daily_arm_price[i]] for i in range(3)]
        vector_daily_prices_loc.append(daily_price)

        #bid:

        daily_arm_bid = [tsgauss_learner_bids[i].pull_arm() for i in range(3)]
        daily_bid = [bids[daily_arm_bid[i]] for i in range(3)]
        vector_daily_bids_loc.append(daily_bid)

        # Get new users in the day t and their costs
        new_users = []
        for i in range(1,4):
            new_users.append(env.get_new_users_daily(daily_bid[i-1],i))

        vector_daily_user_per_class_loc.append(new_users)

        cost = []
        for i in range(1,4):
            cost.append(env.get_cost_per_click(daily_bid[i-1],i))

        # Get the total cost
        total_cost = 0
        for user in range(len(new_users)):
            total_cost += new_users[user] * cost[user]

        # Calculate the number of bought items
        daily_bought_items_per_class = [0, 0, 0]

        for user in range(len(new_users)):
            for c in range(new_users[user]):
                daily_bought_items_per_class[user] += env.buy(daily_price[user], user + 1)

        # Calculate the revenue
        daily_revenue = [0, 0, 0]
        for i in range(3):
            daily_revenue[i] = daily_bought_items_per_class[i] * env.get_margin(daily_price[i]) - cost[i]

        # Get delayed rewards
        next_30_days = []

        for user in range(1, 4):
            next_30_days.append(env.get_next_30_days(daily_bought_items_per_class[user - 1], daily_price[user-1],user))
        #print(f'{next_30_days=}')
        #update observations
        for i in range(3):
            tsgauss_learner_prices[i].update_observations(daily_arm_price[i], daily_revenue[i], next_30_days[i])
            tsgauss_learner_bids[i].update_observations(daily_arm_bid[i], daily_revenue[i], next_30_days[i])

    # put results in the given queue

    revenue_loc=[]
    for k in range(T-30):
        revenue_loc.append([])
        for i in range(3):
            revenue_loc[k].append(tsgauss_learner_prices[i].collected_rewards[k])

    results_queue.put((vector_daily_prices_loc, vector_daily_bids_loc, revenue_loc, vector_daily_user_per_class_loc))

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
        for j in range(len(list_of_lists_of_lists[0])):
            for k in range(3):
                final[k][i][j] = list_of_lists_of_lists[i][j][k]

    mean = []

    for i in range(3):
        mean.append(to_np_arr_and_then_mean(final[i]))

    return mean

if __name__ == '__main__':
    log('N = ' + str(N))

    #initializations of results list
    prices = [] * N
    bids = [] * N
    revenue = [] * N
    user = [] * N

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
        prices.insert(i, results[i][0])
        bids.insert(i, results[i][1])
        revenue.insert(i, results[i][2])
        user.insert(i, results[i][3])

    # calculate the mean values
    mean_price = to_np_arr_and_then_mean_per_class(prices)
    mean_bids = to_np_arr_and_then_mean_per_class(bids)
    mean_revenue = to_np_arr_and_then_mean_per_class(revenue)
    mean_user = to_np_arr_and_then_mean_per_class(user)

    cwd = os.getcwd()
    print("Current working directory: " + cwd)
    plots_folder = os.path.join(cwd, "plotsp7")
    print("Plots folder: " + plots_folder)

    multi_plot(mean_price,'price', plots_folder)
    multi_plot(mean_bids,'bid', plots_folder)
    multi_plot(mean_revenue,'revenue', plots_folder)
    multi_plot(mean_user,'user', plots_folder)
