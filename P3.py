import os
from multiprocessing import Process, Queue
from operator import add

import numpy
from matplotlib import pyplot

from ucb1 import UCB1Learner
from environment import Environment
from tsgauss import TSLearnerGauss

env = Environment()

prices = numpy.linspace(1, 10, num=10)
bids = [0.7]
T = 365
N = 2


def iterate_days(results_queue, idx=0):
    ucb1_learner = UCB1Learner(len(prices))
    tsgauss_learner = TSLearnerGauss(len(prices))
    vector_daily_price_ucb1_loc = []
    vector_daily_revenue_ucb1_loc = []
    vector_daily_price_ts_loc = []
    vector_daily_revenue_ts_loc = []

    for t in range(T):
        if t % 10 == 0:
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
            next_30_days = [
                next_30_days[j] + env.get_next_30_days(daily_bought_items_per_class_ucb1[user - 1], daily_price_ucb1,
                                                       user)[j]
                for j in range(len(next_30_days))]
        # point-wise list sum
        ucb1_learner.update_observations(daily_arm_ucb1, daily_revenue_ucb1, next_30_days)

        # print("Daily revenue UCB1:               {:.5f}".format(daily_revenue_ucb1 + sum(next_30_days)))
        # print("Daily revenue Thomson Sampling:   {:.5f}".format(daily_revenue_ts + sum(next_30_days)))

        next_30_days = [0] * 30
        for user in range(1, 4):
            next_30_days = list(
                map(add, next_30_days, env.get_next_30_days(daily_bought_items_per_class_ts[user - 1], daily_price_ts,
                                                            user)))
        tsgauss_learner.update_observations(daily_arm_ts, daily_revenue_ts, next_30_days)

    results_queue.put((ucb1_learner.collected_rewards, tsgauss_learner.collected_rewards, vector_daily_revenue_ucb1_loc,
                       vector_daily_revenue_ucb1_loc, vector_daily_price_ts_loc, vector_daily_revenue_ts_loc))


def to_np_arr_and_then_mean(list_of_lists):
    print(list_of_lists)
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

    processes = [] * N
    results = [] * N
    q = Queue()

    for i in range(N):
        print("Starting Thread N. {:3d}".format(i))
        p = Process(target=iterate_days, args=(q, i))
        processes.append(p)
        p.start()

    for p in processes:
        ret = q.get()
        results.append(ret)

    for i in range(len(processes)):
        processes[i].join()

    for i in range(len(results)):
        collected_rewards_ucb1.insert(i, results[i][0])
        collected_rewards_ts.insert(i, results[i][1])
        vector_daily_price_ucb1.insert(i, results[i][2])
        vector_daily_revenue_ucb1.insert(i, results[i][3])
        vector_daily_price_ts.insert(i, results[i][4])
        vector_daily_revenue_ts.insert(i, results[i][5])

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

    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ucb1)
    pyplot.xlim([0, T - 30])
    pyplot.legend(['UCB1', 'TS'])
    pyplot.title('Collected reward')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'Collected rewards.png'))

    pyplot.figure()
    pyplot.plot(mean_vector_daily_price_ucb1)
    pyplot.plot(vector_daily_price_ts)
    pyplot.xlim([0, T - 30])
    pyplot.legend(['UCB1', 'TS'])
    pyplot.title('daily prices')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'Daily prices.png'))

    pyplot.figure()
    pyplot.plot(mean_vector_daily_revenue_ucb1)
    pyplot.plot(mean_vector_daily_revenue_ts)
    pyplot.xlim([0, T - 30])
    pyplot.legend(['UCB1 ', ' TS '])
    pyplot.title('daily revenue')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'CDaily revenue.png'))

    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ucb1)
    # plt.plot([i * 1000 for i in vector_daily_price_ucb1])
    pyplot.xlim([0, T - 30])
    pyplot.title('UCB1 confronto prezzo revenue')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'UCB1 confronto prezzo-revenue.png'))

    pyplot.figure()
    pyplot.plot(mean_collected_rewards_ts)
    # plt.plot([i * 1000 for i in vector_daily_price_ts])
    pyplot.xlim([0, T - 30])
    pyplot.title('TS confronto prezzo revenue')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'TS confronto prezzo revenue.png'))
