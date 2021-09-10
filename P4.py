# the following lines just add verbose option and others command line options

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
parser.add_argument('-p', help="make plot, automatically set n=1", action="store_true")
# how many executions:
parser.add_argument('-n', help="set number of iteration", default = 10)
N = int(parser.parse_args().n)

verbose = parser.parse_args().verbose

if verbose:
    def log(argument):
        print(argument)
else:
    def log(argument):
        return

plot_this =  parser.parse_args().p
'''if plot_this == True:
    N=1'''

# now the real code begins

import numpy as np
from scipy.stats import t as tstudent
import os
from matplotlib import pyplot

from environment import Environment
from tsgausspricecontextgeneration import TSLearnerGauss
from P1utilities import get_best_bid_price_possible_reward

# A --> B + C
# C --> D + E

confidence = 0.99

T = 365

env = Environment()

#bids and prices range
bids = env.bids
prices = env.prices

bids, best_daily_price, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
bids = [bids]

mu0 = [1200, 800, 400, 300, 100]
tau_list = [15, 10, 5, 3, 1]
sigma0 = 15

# split --> context_a_split --> splitting
#       --> context_c_split --> splitting

def splitting(p1, mu1, p2, mu2, muzero):
    'return true if we need to split the context, false otherwise'

    log(f' {p1 = } {mu1 = } {p2 = } {mu2 = } {muzero = }')
    log('splitting: ' + str(p1 * mu1 + p2 * mu2 > muzero))

    return p1 * mu1 + p2 * mu2 > muzero

def n_pulled_arm_and_reward_per_arm_counter(rev_per_class, d_arm_per_class, us_per_class,classes):
    '''return n_pulled_arm and reward_per_arm'''
    '''classes is a list of class ex: [0,1,2]'''
    reward_per_arm = [0] * n_arms
    n_pulled_arm = [0] * n_arms

    for i in range(len(rev_per_class)):
        n_pulled_arm[d_arm_per_class[i][classes[0]]] += 1
        reward_per_arm[d_arm_per_class[i][classes[0]]] += sum([rev_per_class[i][clas] for clas in classes])/sum([us_per_class[i][clas] for clas in classes])

    return n_pulled_arm, reward_per_arm

def best_arm_and_mean_best_arm(n_pulled_arm, reward_per_arm):
    '''return the best arm and the mean of this arm'''
    mean_per_arm = [a / b if b!=0 else b for a, b in zip(reward_per_arm, n_pulled_arm)]
    mean_best_arm = max(mean_per_arm)
    best_arm = mean_per_arm.index(mean_best_arm)

    return best_arm, mean_best_arm

def find_context_probability(us_per_class, classes, classes_tot):
    '''return the probability of a context'''

    users = 0
    users_tot = 0

    for i in range(len(us_per_class)):

        for clas in classes:
            users += us_per_class[i][clas]

        for clas in classes_tot:
            users_tot += us_per_class[i][clas]

    p = users / users_tot

    p = p - np.sqrt( - np.log (confidence) / (2 * (users_tot)) )

    return p

def find_lower_bound(rev_per_class, d_arm_per_class, us_per_class, n_pulled_arm, best_arm, mean_best_arm, classes):
    '''compute the lower bound of mu'''

    rewards_best_arm = np.empty(0)

    for j in range(len(d_arm_per_class)):
        if d_arm_per_class[j][classes[0]] == best_arm:
            rewards_best_arm = np.append( rewards_best_arm, sum([rev_per_class[j][i] for i in classes])/sum([us_per_class[j][i] for i in classes]))

    var = np.var(rewards_best_arm)

    return mean_best_arm - tstudent.ppf(confidence, 1 if n_pulled_arm[best_arm]==0 or n_pulled_arm[best_arm]==1 else n_pulled_arm[best_arm] - 1, loc=0, scale=1) * np.sqrt(var / n_pulled_arm[best_arm])

def context_split(rev_per_class, d_arm_per_class, us_per_class, context):
    'return true if we need to split the context, false otherwise'

    # based on the context set the classes of the two new context

    if context == 1:
        classes_1 = [0]
        classes_2 = [1,2]
    if context == 2:
        classes_1 = [1]
        classes_2 = [2]

    classes_tot = classes_1 + classes_2

    # find the best arm for the first new context

    n_pulled_arm_1, reward_per_arm_1 = n_pulled_arm_and_reward_per_arm_counter(rev_per_class, d_arm_per_class, us_per_class, classes_1)

    best_arm_1, mean_best_arm_1 = best_arm_and_mean_best_arm(n_pulled_arm_1, reward_per_arm_1)

    # find the best arm for the second new context

    n_pulled_arm_2, reward_per_arm_2 = n_pulled_arm_and_reward_per_arm_counter(rev_per_class, d_arm_per_class, us_per_class, classes_2)

    best_arm_2, mean_best_arm_2 = best_arm_and_mean_best_arm(n_pulled_arm_2, reward_per_arm_2)

    # find the best arm total

    n_pulled_arm_tot, reward_per_arm_tot = n_pulled_arm_and_reward_per_arm_counter(rev_per_class, d_arm_per_class, us_per_class, classes_tot)

    best_arm_tot, mean_best_arm_tot = best_arm_and_mean_best_arm(n_pulled_arm_tot, reward_per_arm_tot)

    #find the lower bounds of the probabilities of the two new context

    p_1 = find_context_probability(us_per_class, classes_1, classes_tot)

    p_2 = find_context_probability(us_per_class, classes_2, classes_tot)

    # find lower bound mu_1, mu_2 mu_0

    mu_1 = find_lower_bound(rev_per_class, d_arm_per_class, us_per_class, n_pulled_arm_1, best_arm_1, mean_best_arm_1, classes_1)
    mu_2 = find_lower_bound(rev_per_class, d_arm_per_class, us_per_class, n_pulled_arm_2, best_arm_2, mean_best_arm_2, classes_2)
    mu_tot = find_lower_bound(rev_per_class, d_arm_per_class, us_per_class, n_pulled_arm_tot, best_arm_tot, mean_best_arm_tot, classes_tot)

    log(f'{best_arm_1,best_arm_2,best_arm_tot=}')
    log(f'{mu_1,mu_2,mu_tot=}')
    log(f'{p_1,p_2=}')

    if best_arm_1 != best_arm_2:
        return splitting(p_1, mu_1, p_2, mu_2, mu_tot)
    else:
        return False

def split(split_context, rev_per_class, d_arm_per_class, us_per_class):
    'decide if a split is needed by calling context_a_split or context_c_split and return the best context'

    if split_context == 1:
        if context_split(rev_per_class, d_arm_per_class, us_per_class, split_context):
            return 2
        else:
            return 1

    elif split_context == 2:
        if context_split(rev_per_class, d_arm_per_class, us_per_class, split_context):
            return 3
        else:
            return 2

    else:
        return 3

def multi_plot(list_of_things_to_plot, name, yticks=False):
    'plot 3 list of mean: one for every class'

    pyplot.figure()
    for i in range(len(list_of_things_to_plot)):
        pyplot.plot(list_of_things_to_plot[i])
    pyplot.axvline(x=time_first_split, color='k')
    pyplot.axvline(x=time_second_split, color='k')
    if type(yticks)!=type(False):
        pyplot.yticks(yticks)
    pyplot.xlim([0, 365])
    pyplot.legend(['Class 1', 'Class 2', 'Class 3'])
    pyplot.title(str(name) + ' per class')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, str(iter) + str(name) + ' per class.png'))
    pyplot.close()

if __name__ == '__main__':

    for iter in range(N):

        users_per_class = []
        revenue_per_class = []
        daily_arm_per_class = []
        last30dayschoice = []

        time_first_split = 0
        time_second_split = 0

        context = 1

        n_arms = len(prices)
        tau=tau_list[0]
        tsgauss_learner = TSLearnerGauss(n_arms, [], [mu0[0]] * n_arms, [tau] * n_arms, sigma0, [], [], np.zeros(n_arms), 0, [0]*n_arms)

        for t in range(T):
            if t%7==0:
                print("Iteration day: " + str(t))
            # Get new users in the day t and their costs
            [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
            new_users = [new_user_1, new_user_2, new_user_3]

            users_per_class.append(new_users)

            [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
            cost = [cost1, cost2, cost3]

            # Get the total cost
            total_cost = 0
            for i in range(len(new_users)):
                total_cost += new_users[i] * cost[i]

            # In the first days we won't split for sure
            if t < 67:
                daily_arm = tsgauss_learner.pull_arm()
                daily_price = [prices[daily_arm]] * 3
                daily_arm_per_class.append([daily_arm] * 3)

            # Check if it is better to split
            else:
                if t % 7 == 0:
                    context_old = context
                    context = split(context_old, revenue_per_class[0:-30], daily_arm_per_class[0:-30], users_per_class[0:-30])

                    # Create new learners if a split has happened
                    if context > context_old:

                        if context == 2:
                            print('A -- > B + C at day: ' + str (t) + '--------------------------------------------------------------------------------------------------------------------------------' + str(iter))
                            time_first_split = t

                            #compute tau_b and mu_b then create the new tsgauss_learner_b
                            reward_per_arm_b = [0] * n_arms
                            n_pulled_arm_b = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_b[daily_arm_per_class[i][0]] += 1
                                reward_per_arm_b[daily_arm_per_class[i][0]] += revenue_per_class[i][0]

                            mean_per_arm_b = [a / b if b!=0 else b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]  # element

                            n_pulled_arm_b = np.array(n_pulled_arm_b)
                            mean_per_arm_b = np.array(mean_per_arm_b)
                            tau = tau_list[1]
                            mu_b = n_pulled_arm_b * tau**2 * mean_per_arm_b / (n_pulled_arm_b * tau**2 + sigma0**2) + sigma0**2 * mu0[1] / (n_pulled_arm_b * tau**2 + sigma0**2)

                            tau_b = (sigma0 * tau)**2 / (n_pulled_arm_b * tau**2 + sigma0**2)

                            n_pulled_arm_b=n_pulled_arm_b.tolist()
                            mean_per_arm_b=mean_per_arm_b.tolist()
                            mu_b=mu_b.tolist()
                            tau_b=tau_b.tolist()
                            # print(mean_per_arm_b)
                            # print(mu_b)
                            # print(tau_b)
                            k = 31
                            tsgauss_learner_b = TSLearnerGauss(n_arms, [revenue_per_class[i][0] for i in range(len(revenue_per_class)-k)], mu_b, tau_b, sigma0, [daily_arm_per_class[i][0] for i in range(t-k,t)], [revenue_per_class[i][0] for i in range(t-k,t)], np.array(reward_per_arm_b), t, n_pulled_arm_b)

                            #compute tau_c and mu_c then create the new tsgauss_learner_c
                            reward_per_arm_c = [0] * n_arms
                            n_pulled_arm_c = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_c[daily_arm_per_class[i][1]] += 1
                                reward_per_arm_c[daily_arm_per_class[i][1]] += revenue_per_class[i][1] + revenue_per_class[i][2]

                            mean_per_arm_c = [a / b if b!=0 else b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]  # element wise division python

                            n_pulled_arm_c = np.array(n_pulled_arm_c)
                            mean_per_arm_c = np.array(mean_per_arm_c)
                            tau=tau_list[2]
                            mu_c = n_pulled_arm_c * tau**2 * mean_per_arm_c / (n_pulled_arm_c * tau**2 + sigma0**2) + sigma0**2 * mu0[2] / (n_pulled_arm_c * tau**2 + sigma0**2)
                            tau_c = (sigma0 * tau)**2 / (n_pulled_arm_c * tau**2 + sigma0**2)

                            n_pulled_arm_c=n_pulled_arm_c.tolist()
                            mean_per_arm_c=mean_per_arm_c.tolist()
                            mu_c=mu_c.tolist()
                            tau_c=tau_c.tolist()
                            # print(mean_per_arm_c)
                            # print(mu_c)
                            # print(tau_c)

                            tsgauss_learner_c = TSLearnerGauss(n_arms, [revenue_per_class[i][1] + revenue_per_class[i][2] for i in range(len(revenue_per_class)-k)], mu_c, tau_c, sigma0, [daily_arm_per_class[i][0] for i in range(t-k,t)], [revenue_per_class[i][1] + revenue_per_class[i][2] for i in range(t-k,t)], np.array(reward_per_arm_c), t, n_pulled_arm_c)

                        if context == 3:
                            print('C -- > D + E at day: ' + str (t) + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + str(iter))
                            time_second_split = t

                            #compute tau_d and mu_d then create the new tsgauss_learner_d
                            reward_per_arm_d = [0] * n_arms
                            n_pulled_arm_d = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_d[daily_arm_per_class[i][1]] += 1
                                reward_per_arm_d[daily_arm_per_class[i][1]] += revenue_per_class[i][1]

                            mean_per_arm_d = [a / b if b!=0 else b for a, b in zip(reward_per_arm_d, n_pulled_arm_d)]  # element

                            n_pulled_arm_d = np.array(n_pulled_arm_d)
                            mean_per_arm_d = np.array(mean_per_arm_d)
                            tau=tau_list[3]
                            mu_d = n_pulled_arm_d * tau**2 * mean_per_arm_d / (n_pulled_arm_d * tau**2 + sigma0**2) + sigma0**2 * mu0[3] / (n_pulled_arm_d * tau**2 + sigma0**2)

                            tau_d = (sigma0 * tau)**2 / (n_pulled_arm_d * tau**2 + sigma0**2)


                            n_pulled_arm_d=n_pulled_arm_d.tolist()
                            mean_per_arm_d=mean_per_arm_d.tolist()
                            mu_d=mu_d.tolist()
                            tau_d=tau_d.tolist()
                            # print(mean_per_arm_d)
                            # print(mu_d)
                            # print(tau_d)

                            tsgauss_learner_d = TSLearnerGauss(n_arms, [revenue_per_class[i][1] for i in range(len(revenue_per_class)-k)], mu_d, tau_d, sigma0, [daily_arm_per_class[i][1] for i in range(t-k,t)], [revenue_per_class[i][1] for i in range(t-k,t)], np.array(reward_per_arm_d), t, n_pulled_arm_d)

                            #compute tau_e and mu_e then create the new tsgauss_learner_e
                            reward_per_arm_e = [0] * n_arms
                            n_pulled_arm_e = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_e[daily_arm_per_class[i][2]] += 1
                                reward_per_arm_e[daily_arm_per_class[i][2]] += revenue_per_class[i][2]

                            mean_per_arm_e = [a / b if b!=0 else b for a, b in zip(reward_per_arm_e, n_pulled_arm_e)]  # element wise division python

                            n_pulled_arm_e = np.array(n_pulled_arm_e)
                            mean_per_arm_e = np.array(mean_per_arm_e)
                            tau=tau_list[4]
                            mu_e = n_pulled_arm_e * tau**2 * mean_per_arm_e / (n_pulled_arm_e * tau**2 + sigma0**2) + sigma0**2 * mu0[4] / (n_pulled_arm_e * tau**2 + sigma0**2)
                            tau_e = (sigma0 * tau)**2 / (n_pulled_arm_e * tau**2 + sigma0**2)

                            n_pulled_arm_e=n_pulled_arm_e.tolist()
                            mean_per_arm_e=mean_per_arm_e.tolist()
                            mu_e=mu_e.tolist()
                            tau_e=tau_e.tolist()
                            # print(mean_per_arm_e)
                            # print(mu_e)
                            # print(tau_e)

                            tsgauss_learner_e = TSLearnerGauss(n_arms, [revenue_per_class[i][2] for i in range(len(revenue_per_class)-k)], mu_e, tau_e, sigma0, [daily_arm_per_class[i][2] for i in range(t-k,t)], [revenue_per_class[i][2] for i in range(t-k,t)], np.array(reward_per_arm_e), t, n_pulled_arm_e)


                # according to the context, use the right learners to choose the daily price
                if context == 1:
                    daily_arm = tsgauss_learner.pull_arm()
                    daily_price = [prices[daily_arm]] * 3
                    daily_arm_per_class.append([daily_arm] * 3)

                if context == 2:
                    daily_arm_b = tsgauss_learner_b.pull_arm()
                    daily_arm_c = tsgauss_learner_c.pull_arm()
                    daily_price = [prices[daily_arm_b], prices[daily_arm_c], prices[daily_arm_c]]
                    daily_arm_per_class.append([daily_arm_b, daily_arm_c, daily_arm_c])

                if context == 3:
                    daily_arm_b = tsgauss_learner_b.pull_arm()
                    daily_arm_d = tsgauss_learner_d.pull_arm()
                    daily_arm_e = tsgauss_learner_e.pull_arm()
                    daily_price = [prices[daily_arm_b], prices[daily_arm_d], prices[daily_arm_e]]
                    daily_arm_per_class.append([daily_arm_b, daily_arm_d, daily_arm_e])

            # Calculate the number of bought items, the revenue and the next 30 days
            daily_bought_items_perclass = [0, 0, 0]

            for i in range(len(new_users)):
                for c in range(new_users[i]):
                    daily_bought_items_perclass[i] += env.buy(daily_price[i], i + 1)

            margin = [env.get_margin(int(price)) for price in daily_price]

            revenue_per_class_today = []
            for i in range(len(margin)):
                revenue_per_class_today.append(margin[i] * daily_bought_items_perclass[i] - cost[i] * new_users[i])

            next_30_days = [env.get_next_30_days(daily_bought_items_perclass[0], daily_price[0], 1), env.get_next_30_days(daily_bought_items_perclass[1], daily_price[1], 2), env.get_next_30_days(daily_bought_items_perclass[2], daily_price[2], 3)]

            sum_next_30_days = [sum(next_30_days[i]) for i in range(len(next_30_days))]

            # according to the context, update right learners
            if context == 1:
                tsgauss_learner.update_observations(daily_arm, sum(revenue_per_class_today), [next_30_days[0][i] + next_30_days[1][i] + next_30_days[2][i] for i in range(len(next_30_days[0]))])

            if context == 2:
                tsgauss_learner_b.update_observations(daily_arm_b, revenue_per_class_today[0], next_30_days[0])
                tsgauss_learner_c.update_observations(daily_arm_c, revenue_per_class_today[1] + revenue_per_class_today[2], [next_30_days[1][i] + next_30_days[2][i] for i in range(len(next_30_days[0]))])

            if context == 3:
                tsgauss_learner_b.update_observations(daily_arm_b, revenue_per_class_today[0], next_30_days[0])
                tsgauss_learner_d.update_observations(daily_arm_b, revenue_per_class_today[1], next_30_days[1])
                tsgauss_learner_e.update_observations(daily_arm_b, revenue_per_class_today[2], next_30_days[2])

            revenue_per_class.append([revenue_per_class_today[i] + sum_next_30_days[i] for i in range(len(revenue_per_class_today))])

        if plot_this == True and time_first_split!=0 and time_second_split!=0:
            cwd = os.getcwd()
            print("Current working directory: " + cwd)
            plots_folder = os.path.join(cwd, "plotsp4")
            print("Plots folder: " + plots_folder)

            revenue_per_class_new = [[[] for i in range(len(revenue_per_class))] for i in range(len(revenue_per_class[0]))]
            for i in range(len(revenue_per_class)):
                for j in range(len(revenue_per_class[0])):
                    revenue_per_class_new[j][i] = revenue_per_class[i][j]

            multi_plot(revenue_per_class_new, 'Revenue')

            daily_price_per_class = [[[] for i in range(len(daily_arm_per_class))] for i in range(len(daily_arm_per_class[0]))]
            for i in range(len(daily_arm_per_class)):
                for j in range(len(daily_arm_per_class[0])):
                    daily_price_per_class[j][i] = prices[daily_arm_per_class[i][j]]

            multi_plot(daily_price_per_class, 'Price', prices)
