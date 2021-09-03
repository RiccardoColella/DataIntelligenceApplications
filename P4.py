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

import numpy as np
from scipy.stats import t as tstudent

from environment import Environment
from tsgausspricecontextgeneration import TSLearnerGauss
from P1asutilities import get_best_bid_price_possible_reward

# A --> B + C
# C --> D + E

confidence = 0.99

T = 365

#bids and prices range
prices = np.linspace(1, 10, num=10)
bids = np.linspace(0.1, 1, num=10)

bids, best_daily_price, best_possible_reward = get_best_bid_price_possible_reward(bids, prices)
bids = [bids]

mu0 = 1000
tau = 13.5
sigma0 = 13.5

# split --> context_a_split --> splitting
#       --> context_c_split --> splitting

def splitting(p1, mu1, p2, mu2, muzero):
    'return true if we need to split the context, false otherwise'

    log(f' {p1 = } {mu1 = } {p2 = } {mu2 = } {muzero = }')
    log('splitting: ' + str(p1 * mu1 + p2 * mu2 > muzero))

    return p1 * mu1 + p2 * mu2 > muzero

'''
def daily_normalized_rev(rev_per_class_daily, us_per_class_daily):
    'normalize the rev per class for the users'

    return sum([a/b for a,b in zip(rev_per_class_daily,us_per_class_daily)])
'''

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

    ''''print('mean, t_student')
    print(mean_best_arm, tstudent.ppf(confidence, 1 if n_pulled_arm[best_arm]==0 or n_pulled_arm[best_arm]==1 else n_pulled_arm[best_arm] - 1, loc=0, scale=1) * np.sqrt(var / n_pulled_arm[best_arm]  ))'''

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

    '''print(f'{best_arm_1,best_arm_2,best_arm_tot=}')
    print(f'{mu_1,mu_2,mu_tot=}')
    print(f'{p_1,p_2=}')'''

    if best_arm_1 != best_arm_2:
        return splitting(p_1, mu_1, p_2, mu_2, mu_tot)
    else:
        return False

#the following two functions are outdated and no longer necessary
'''
def context_a_split(rev_per_class, d_arm_per_class, us_per_class):
    'return true if we need to split the context a, false otherwise'

    day = len(rev_per_class)

    # Here we find the best arm for b and we compute some parameters
    reward_per_arm_b = [0] * n_arms
    n_pulled_arm_b = [0] * n_arms
    for i in range(day):
        n_pulled_arm_b[d_arm_per_class[i][0]] += 1
        reward_per_arm_b[d_arm_per_class[i][0]] += rev_per_class[i][0]

    mean_per_arm_b = [a / b if b!=0 else b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]
    mean_best_arm_b = max(mean_per_arm_b)
    best_arm_b = mean_per_arm_b.index(mean_best_arm_b)

    # Here we find the best arm for c and we compute some parameters
    reward_per_arm_c = [0] * n_arms
    n_pulled_arm_c = [0] * n_arms
    for i in range(day):
        n_pulled_arm_c[d_arm_per_class[i][1]] += 1
        reward_per_arm_c[d_arm_per_class[i][1]] += rev_per_class[i][1] + rev_per_class[i][2]

    mean_per_arm_c = [a / b if b!=0 else b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]
    mean_best_arm_c = max(mean_per_arm_c)
    best_arm_c = mean_per_arm_c.index(mean_best_arm_c)

    # here we find the best arm in total
    reward_per_arm_tot = [0] * n_arms
    n_pulled_arm_tot = [0] * n_arms
    for i in range(day):
        n_pulled_arm_tot[d_arm_per_class[i][0]] += 1
        reward_per_arm_tot[d_arm_per_class[i][0]] += sum(rev_per_class[i])

    mean_per_arm_tot = [a / b if b!=0 else b for a, b in zip(reward_per_arm_tot, n_pulled_arm_tot)]
    mean_best_arm_tot = max(mean_per_arm_tot)
    best_arm_tot = mean_per_arm_tot.index(mean_best_arm_tot)

    print(f'{best_arm_tot=}')

    # find probability of context b and c, then compute the lower bounds
    b_users = 0
    c_users = 0

    for i in range(day):
        b_users += us_per_class[i][0]
        c_users += us_per_class[i][1] + us_per_class[i][2]

    pb = b_users / (b_users + c_users)
    pc = c_users / (b_users + c_users)

    # pb and pc are lower bounds
    pb = pb - np.sqrt( - np.log (confidence) / (2 * (b_users + c_users)) )
    pc = pc - np.sqrt( - np.log (confidence) / (2 * (b_users + c_users)) )

    # compute variance revenue for best arm b, c, and tot
    rewards_best_arm_b = np.empty(0)
    rewards_best_arm_c = np.empty(0)
    rewards_best_arm_tot = np.empty(0)

    for j in range(day):

        if d_arm_per_class[j][0] == best_arm_b:
            rewards_best_arm_b = np.append(rewards_best_arm_b, rev_per_class[j][0])

        if d_arm_per_class[j][1] == best_arm_c:
            rewards_best_arm_c = np.append(rewards_best_arm_c, (rev_per_class[j][1] + rev_per_class[j][2]))

        if d_arm_per_class[j][0] == best_arm_tot:
            rewards_best_arm_tot = np.append(rewards_best_arm_tot, (sum(rev_per_class[j])))

    var_b = np.var(rewards_best_arm_b)
    var_c = np.var(rewards_best_arm_c)
    var_tot = np.var(rewards_best_arm_tot)

    # find lower bound mub, muc mu0

    mub = mean_best_arm_b - tstudent.ppf(confidence, 1 if n_pulled_arm_b[best_arm_b]==0 or n_pulled_arm_b[best_arm_b]==1 else n_pulled_arm_b[best_arm_b] - 1, loc=0, scale=1) * np.sqrt(
        var_b / n_pulled_arm_b[best_arm_b])
    muc = mean_best_arm_c - tstudent.ppf(confidence, 1 if n_pulled_arm_c[best_arm_c]==0 or n_pulled_arm_c[best_arm_c]==1 else n_pulled_arm_c[best_arm_c] - 1, loc=0, scale=1) * np.sqrt(
        var_c / n_pulled_arm_c[best_arm_c])
    muzero = mean_best_arm_tot - tstudent.ppf(confidence, 1 if n_pulled_arm_tot[best_arm_tot]==0 or n_pulled_arm_tot[best_arm_tot]==1 else n_pulled_arm_tot[best_arm_tot] - 1, loc=0, scale=1) * np.sqrt(
        var_tot / n_pulled_arm_tot[best_arm_tot])

    ''''''log('rewards_best_arm_b:' + str(rewards_best_arm_b))
    log('var_b:' +str(var_b))
    log('mean_per_arm_b:' +str(mean_per_arm_b))
    log('best_arm_b: ' + str(best_arm_b))
    log('pulled arm times_b:' + str(n_pulled_arm_b[best_arm_b]))


    log('rewards_best_arm_c:' + str(rewards_best_arm_c))
    log('var_c:' +str(var_c))
    log('mean_per_arm_c:' +str(mean_per_arm_c))
    log('best_arm_c: ' + str(best_arm_c))
    log('pulled arm times_c:' + str(n_pulled_arm_c[best_arm_c]))

    log('rewards_best_arm_tot:' + str(rewards_best_arm_tot))
    log('var_tot:' +str(var_tot))
    log('mean_per_arm_tot:' +str(mean_per_arm_tot))
    log('best_arm_tot: ' + str(best_arm_tot))
    log('pulled arm times_tot:' + str(n_pulled_arm_tot[best_arm_tot]))

    print(f'{[mean_best_arm_b+mean_best_arm_c,mean_best_arm_tot]=}')
    print(f'{[mub+muc,muzero]=}') ''''''

    if best_arm_b != best_arm_c:
        return splitting(pb, mub, pc, muc, muzero)
    else:
        return False

def context_c_split(rev_per_class, d_arm_per_class, us_per_class):
    'return true if we need to split the context c, false otherwise'

    day = len(rev_per_class)

    # Here we find the best arm for d and we compute some parameters
    reward_per_arm_d = [0] * n_arms
    n_pulled_arm_d = [0] * n_arms
    for i in range(day):
        n_pulled_arm_d[d_arm_per_class[i][1]] += 1
        reward_per_arm_d[d_arm_per_class[i][1]] += rev_per_class[i][1]

    mean_per_arm_d = [a / b if b!=0 else b for a, b in zip(reward_per_arm_d, n_pulled_arm_d)]  # element wise division python
    mean_best_arm_d = max(mean_per_arm_d)
    best_arm_d = mean_per_arm_d.index(mean_best_arm_d)

    # Here we find the best arm for e and we compute some parameters
    reward_per_arm_e = [0] * n_arms
    n_pulled_arm_e = [0] * n_arms
    for i in range(day):
        n_pulled_arm_e[d_arm_per_class[i][2]] += 1
        reward_per_arm_e[d_arm_per_class[i][2]] += rev_per_class[i][2]

    mean_per_arm_e = [a / b if b!=0 else b for a, b in zip(reward_per_arm_e, n_pulled_arm_e)]  # element wise division python
    mean_best_arm_e = max(mean_per_arm_e)
    best_arm_e = mean_per_arm_e.index(mean_best_arm_e)

    # here we find the best arm in total
    reward_per_arm_tot = [0] * n_arms
    n_pulled_arm_tot = [0] * n_arms

    for i in range(day):
        n_pulled_arm_tot[d_arm_per_class[i][1]] += 1
        reward_per_arm_tot[d_arm_per_class[i][1]] += rev_per_class[i][1] + rev_per_class[i][2]

    mean_per_arm_tot = [a / b if b!=0 else b for a, b in zip(reward_per_arm_tot, n_pulled_arm_tot)]  # element wise division python
    mean_best_arm_tot = max(mean_per_arm_tot)
    best_arm_tot = mean_per_arm_tot.index(mean_best_arm_tot)

    # find probability of context d and e, then compute the lower bounds
    d_users = 0
    e_users = 0

    for i in range(day):
        d_users += us_per_class[i][1]
        e_users += us_per_class[i][2]

    pd = d_users / (d_users + e_users)
    pe = e_users / (d_users + e_users)

    # pd and pe are lower bounds
    pd = pd - np.sqrt( - np.log (confidence) / (2 * (d_users + e_users)) )
    pe = pe - np.sqrt( - np.log (confidence) / (2 * (d_users + e_users)) )

    # compute variance revenue for best arm d, e, and tot
    rewards_best_arm_d = np.empty(0)
    rewards_best_arm_e = np.empty(0)
    rewards_best_arm_tot = np.empty(0)

    for j in range(day):

        if d_arm_per_class[j][1] == best_arm_d:
            rewards_best_arm_d = np.append(rewards_best_arm_d, rev_per_class[j][1])

        if d_arm_per_class[j][2] == best_arm_e:
            rewards_best_arm_e = np.append(rewards_best_arm_e, (rev_per_class[j][2]))

        if d_arm_per_class[j][1] == best_arm_tot:
            rewards_best_arm_tot = np.append(rewards_best_arm_tot, rev_per_class[j][1] + rev_per_class[j][2] )

    var_d = np.var(rewards_best_arm_d)
    var_e = np.var(rewards_best_arm_e)
    var_tot = np.var(rewards_best_arm_tot)

    # find lower bound mud, mue, mu0
    mud = mean_best_arm_d - tstudent.ppf(confidence, 1 if n_pulled_arm_d[best_arm_d]==0 or n_pulled_arm_d[best_arm_d]==1 else n_pulled_arm_d[best_arm_d] - 1, loc=0, scale=1) * np.sqrt(
        var_d / n_pulled_arm_d[best_arm_d])
    mue = mean_best_arm_e - tstudent.ppf(confidence, 1 if n_pulled_arm_e[best_arm_e]==0 or n_pulled_arm_e[best_arm_e]==1 else n_pulled_arm_e[best_arm_e] - 1, loc=0, scale=1) * np.sqrt(
        var_e / n_pulled_arm_e[best_arm_e])
    muzero = mean_best_arm_tot - tstudent.ppf(confidence, 1 if n_pulled_arm_tot[best_arm_tot]==0 or n_pulled_arm_tot[best_arm_tot]==1 else n_pulled_arm_tot[best_arm_tot] - 1, loc=0, scale=1) * np.sqrt(
        var_tot / n_pulled_arm_tot[best_arm_tot])

    log('best_arm_d: ' + str(best_arm_d))
    log('best_arm_e: ' + str(best_arm_e))
    log('best_arm_tot: ' + str(best_arm_tot))

    if best_arm_d!=best_arm_e:
        return splitting(pd, mud, pe, mue, muzero)
    else:
        return False '''

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

## TODO: parallelization like P3

if __name__ == '__main__':

    for iter in range(N):

        env = Environment()

        users_per_class = []
        revenue_per_class = []
        daily_arm_per_class = []
        last30dayschoice = []

        context = 1

        n_arms = len(prices)
        tsgauss_learner = TSLearnerGauss(n_arms, [], [mu0] * n_arms, [tau] * n_arms, sigma0, [], [], np.zeros(n_arms), 0)

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

                            #compute tau_b and mu_b then create the new tsgauss_learner_b
                            reward_per_arm_b = [0] * n_arms
                            n_pulled_arm_b = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_b[daily_arm_per_class[i][0]] += 1
                                reward_per_arm_b[daily_arm_per_class[i][0]] += revenue_per_class[i][0]

                            mean_per_arm_b = [a / b if b!=0 else b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]  # element

                            n_pulled_arm_b = np.array(n_pulled_arm_b)
                            mean_per_arm_b = np.array(mean_per_arm_b)

                            mu_b = n_pulled_arm_b * tau**2 * mean_per_arm_b / (n_pulled_arm_b * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_b * tau**2 + sigma0**2)

                            tau_b = (sigma0 * tau)**2 / (n_pulled_arm_b * tau**2 + sigma0**2)

                            n_pulled_arm_b=n_pulled_arm_b.tolist()
                            mean_per_arm_b=mean_per_arm_b.tolist()
                            mu_b=mu_b.tolist()
                            tau_b=tau_b.tolist()

                            k = 31
                            tsgauss_learner_b = TSLearnerGauss(n_arms, [revenue_per_class[i][0] for i in range(len(revenue_per_class)-k)], mu_b, tau_b, sigma0, [daily_arm_per_class[i][0] for i in range(t-k,t)], [revenue_per_class[i][0] for i in range(t-k,t)], np.array(reward_per_arm_b), t)

                            #compute tau_c and mu_c then create the new tsgauss_learner_c
                            reward_per_arm_c = [0] * n_arms
                            n_pulled_arm_c = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_c[daily_arm_per_class[i][1]] += 1
                                reward_per_arm_c[daily_arm_per_class[i][1]] += revenue_per_class[i][1] + revenue_per_class[i][2]

                            mean_per_arm_c = [a / b if b!=0 else b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]  # element wise division python

                            n_pulled_arm_c = np.array(n_pulled_arm_c)
                            mean_per_arm_c = np.array(mean_per_arm_c)

                            mu_c = n_pulled_arm_c * tau**2 * mean_per_arm_c / (n_pulled_arm_c * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_c * tau**2 + sigma0**2)
                            tau_c = (sigma0 * tau)**2 / (n_pulled_arm_c * tau**2 + sigma0**2)

                            n_pulled_arm_c=n_pulled_arm_c.tolist()
                            mean_per_arm_c=mean_per_arm_c.tolist()
                            mu_c=mu_c.tolist()
                            tau_c=tau_c.tolist()

                            tsgauss_learner_c = TSLearnerGauss(n_arms, [revenue_per_class[i][1] + revenue_per_class[i][2] for i in range(len(revenue_per_class)-k)], mu_c, tau_c, sigma0, [daily_arm_per_class[i][0] for i in range(t-k,t)], [revenue_per_class[i][1] + revenue_per_class[i][2] for i in range(t-k,t)], np.array(reward_per_arm_c), t)

                        if context == 3:
                            print('C -- > D + E at day: ' + str (t) + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + str(iter))

                            #compute tau_d and mu_d then create the new tsgauss_learner_d
                            reward_per_arm_d = [0] * n_arms
                            n_pulled_arm_d = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_d[daily_arm_per_class[i][1]] += 1
                                reward_per_arm_d[daily_arm_per_class[i][1]] += revenue_per_class[i][1]

                            mean_per_arm_d = [a / b if b!=0 else b for a, b in zip(reward_per_arm_d, n_pulled_arm_d)]  # element

                            n_pulled_arm_d = np.array(n_pulled_arm_d)
                            mean_per_arm_d = np.array(mean_per_arm_d)

                            mu_d = n_pulled_arm_d * tau**2 * mean_per_arm_d / (n_pulled_arm_d * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_d * tau**2 + sigma0**2)

                            tau_d = (sigma0 * tau)**2 / (n_pulled_arm_d * tau**2 + sigma0**2)


                            n_pulled_arm_d=n_pulled_arm_d.tolist()
                            mean_per_arm_d=mean_per_arm_d.tolist()
                            mu_d=mu_d.tolist()
                            tau_d=tau_d.tolist()

                            tsgauss_learner_d = TSLearnerGauss(n_arms, [revenue_per_class[i][1] for i in range(len(revenue_per_class)-k)], mu_d, tau_d, sigma0, [daily_arm_per_class[i][1] for i in range(t-k,t)], [revenue_per_class[i][1] for i in range(t-k,t)], np.array(reward_per_arm_d), t)

                            #compute tau_e and mu_e then create the new tsgauss_learner_e
                            reward_per_arm_e = [0] * n_arms
                            n_pulled_arm_e = [0] * n_arms
                            for i in range(t-30):
                                n_pulled_arm_e[daily_arm_per_class[i][2]] += 1
                                reward_per_arm_e[daily_arm_per_class[i][2]] += revenue_per_class[i][2]

                            mean_per_arm_e = [a / b if b!=0 else b for a, b in zip(reward_per_arm_e, n_pulled_arm_e)]  # element wise division python

                            n_pulled_arm_e = np.array(n_pulled_arm_e)
                            mean_per_arm_e = np.array(mean_per_arm_e)

                            mu_e = n_pulled_arm_e * tau**2 * mean_per_arm_e / (n_pulled_arm_e * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_e * tau**2 + sigma0**2)
                            tau_e = (sigma0 * tau)**2 / (n_pulled_arm_e * tau**2 + sigma0**2)

                            n_pulled_arm_e=n_pulled_arm_e.tolist()
                            mean_per_arm_e=mean_per_arm_e.tolist()
                            mu_e=mu_e.tolist()
                            tau_e=tau_e.tolist()

                            tsgauss_learner_e = TSLearnerGauss(n_arms, [revenue_per_class[i][2] for i in range(len(revenue_per_class)-k)], mu_e, tau_e, sigma0, [daily_arm_per_class[i][2] for i in range(t-k,t)], [revenue_per_class[i][2] for i in range(t-k,t)], np.array(reward_per_arm_e), t)


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
