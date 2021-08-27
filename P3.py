import multiprocessing
import os
from operator import add

import numpy
from matplotlib import pyplot

from ucb1 import UCB1Learner
from environment import Environment
from tsgauss import TSLearnerGauss

env = Environment()

#prices range
prices = numpy.linspace(1, 10, num=10)
# bids range
bids = [0.6]
# day of algorithm execution
T = 394
# How many computation exectute
N = 50


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

    # For every day:
    for t in range(T):
        if t % 20 == 0:
            print("Iteration day: {:3d} - execution: {:3d}".format(t, idx))
        # Get new users in the day t and their costs
        [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
        new_users = [new_user_1, new_user_2, new_user_3]
        [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
        cost = [cost1, cost2, cost3]

        # Get the total cost
        total_cost = 0
        for user in range(len(new_users)):
            total_cost += new_users[user] * cost[user]

        # Choose the arm and thus the price for UCB
        daily_arm_ucb1 = ucb1_learner.pull_arm()
        daily_price_ucb1 = prices[daily_arm_ucb1]
        vector_daily_price_ucb1_loc.append(daily_price_ucb1)

        # Choose the arm and thus the price for Thomson Sampling
        daily_arm_ts = tsgauss_learner.pull_arm()
        daily_price_ts = prices[daily_arm_ts]
        vector_daily_price_ts_loc.append(daily_price_ts)

        # Get n bought items
        daily_bought_items_per_class_ucb1 = [0, 0, 0]
        daily_bought_items_per_class_ts = [0, 0, 0]

        # Calculate the number of real bought items
        for user in range(len(new_users)):
            for c in range(new_users[user]):
                daily_bought_items_per_class_ucb1[user] += env.buy(daily_price_ucb1, user + 1)
                daily_bought_items_per_class_ts[user] += env.buy(daily_price_ts, user + 1)
            '''daily_bought_items_per_class_ucb1[user] = round(env.get_conversion_rate(daily_price_ucb1, user + 1) * new_users[user])
            daily_bought_items_per_class_ts[user] = round(env.get_conversion_rate(daily_price_ts, user + 1) * new_users[user])'''


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
        # point-wise list sum
        ucb1_learner.update_observations(daily_arm_ucb1, daily_revenue_ucb1, next_30_days)

        # print("Daily revenue UCB1:               {:.5f}".format(daily_revenue_ucb1 + sum(next_30_days)))
        # print("Daily revenue Thomson Sampling:   {:.5f}".format(daily_revenue_ts + sum(next_30_days)))

        # get earnings in the following 30 days
        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ts[user - 1], daily_price_ts,
                                                            user)))
        tsgauss_learner.update_observations(daily_arm_ts, daily_revenue_ts, next_30_days)

    # put results in the given queue
    results_queue.put((ucb1_learner.collected_rewards, tsgauss_learner.collected_rewards, vector_daily_price_ucb1_loc,
                       vector_daily_revenue_ucb1_loc, vector_daily_price_ts_loc, vector_daily_revenue_ts_loc))


def to_np_arr_and_then_mean(list_of_lists):
    """
    Mean of every value of the list, based on the index
    :param list_of_lists: list containing the results for every day in a list for every iteration
    :return: an array of the mean based on values' index
    """
    # print(list_of_lists)
    np_arr = numpy.array(list_of_lists)
    return np_arr.mean(axis=0)


if __name__ == '__main__':
    # mettere P1 come funzione per poter usare ottimo per calcolare il regret

    collected_rewards_ucb1 = [] * 10
    collected_rewards_ts = [] * N
    vector_daily_price_ucb1 = [] * N
    vector_daily_revenue_ucb1 = [] * N
    vector_daily_price_ts = [] * N
    vector_daily_revenue_ts = [] * N

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
        collected_rewards_ucb1.insert(i, results[i][0])
        collected_rewards_ts.insert(i, results[i][1])
        vector_daily_price_ucb1.insert(i, results[i][2])
        vector_daily_revenue_ucb1.insert(i, results[i][3])
        vector_daily_price_ts.insert(i, results[i][4])
        vector_daily_revenue_ts.insert(i, results[i][5])
    # calculate the mean values
    mean_collected_rewards_ucb1 = to_np_arr_and_then_mean(collected_rewards_ucb1)
    mean_collected_rewards_ts = to_np_arr_and_then_mean(collected_rewards_ts)
    mean_vector_daily_price_ucb1 = to_np_arr_and_then_mean(vector_daily_price_ucb1)
    mean_vector_daily_revenue_ucb1 = to_np_arr_and_then_mean(vector_daily_revenue_ucb1)
    mean_vector_daily_price_ts = to_np_arr_and_then_mean(vector_daily_price_ts)
    mean_vector_daily_revenue_ts = to_np_arr_and_then_mean(vector_daily_revenue_ts)

    cwd = os.getcwd()
    print("Current working directory: " + cwd)
    plots_folder = os.path.join(cwd, "plots")
    print("Plots folder: " + plots_folder)

    value_line_to_plot = 904
    plot_line = True

    # Plot collected rewards
    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ucb1)
    pyplot.plot(mean_collected_rewards_ts)
    pyplot.xlim([0, T - 30])
    pyplot.legend(['UCB1', 'TS'])
    if plot_line:
        pyplot.plot([value_line_to_plot for i in range(T)])
    pyplot.title('Collected reward')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'Collected rewards.png'))

    # Plot daily prices
    pyplot.figure()
    pyplot.plot(mean_vector_daily_price_ucb1)
    pyplot.plot(mean_vector_daily_price_ts)
    pyplot.xlim([0, T - 30])
    pyplot.legend(['UCB1', 'TS'])
    pyplot.title('daily prices')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'Daily prices.png'))

    # Plot UCB1 price and revenue comparison
    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ucb1)
    pyplot.plot([i * 100 for i in mean_vector_daily_price_ucb1])
    pyplot.xlim([0, T - 30])
    pyplot.legend(['collected reward', 'price * 100'])
    if plot_line:
        pyplot.plot([value_line_to_plot for i in range(T)])
    pyplot.title('UCB1 confronto prezzo revenue')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'UCB1 confronto prezzo-revenue.png'))

    # Plot TS price and revenue comparison
    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ts)
    pyplot.plot([i * 100 for i in mean_vector_daily_price_ts])
    pyplot.xlim([0, T - 30])
    pyplot.legend(['collected reward', 'price * 100'])
    if plot_line:
        pyplot.plot([value_line_to_plot for i in range(T)])
    pyplot.title('TS confronto prezzo revenue')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'TS confronto prezzo revenue.png'))

    #calculate and plot mean regret
    best_possible_reward = 904
    mean_regret_ts = [best_possible_reward * x for x in range(1,T-29)]
    mean_regret_ts = numpy.array(mean_regret_ts)

    mean_regret_ts = numpy.add( mean_regret_ts , -1 * numpy.cumsum(mean_collected_rewards_ts))

    mean_regret_ucb1 = [best_possible_reward * x for x in range(1,T-29)]

    mean_regret_ucb1 = mean_regret_ucb1 - numpy.cumsum(mean_collected_rewards_ucb1)

    pyplot.figure()
    pyplot.plot(mean_regret_ts)
    pyplot.plot(mean_regret_ucb1)

    pyplot.legend(['mean_regret_ts', 'mean_regret_ucb1'])
    pyplot.xlim([0, T - 30])
    pyplot.title('confronto regret')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'confronto regret.png'))
