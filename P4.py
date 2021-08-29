import numpy as np
from scipy.stats import t as tstudent

from environment import Environment
from tsgaussp4 import TSLearnerGauss

# A --> B + C
# C --> D + E

confidence = 0.99

T = 365

prices = np.linspace(1, 10, num=10)
bids = [0.6]

mu0 = 800
tau = 10
sigma0 = 5

def splitting(p1, mu1, p2, mu2, muzero):
    # return true if we need to split the context, false otherwise
    print(p1)
    print(mu1)
    print(p2)
    print(mu2)
    print(muzero)
    print(str(p1 * mu1 + p2 * mu2) + '>' + str(muzero))
    print(p1 * mu1 + p2 * mu2 > muzero)
    return p1 * mu1 + p2 * mu2 > muzero

def context_a_split(rev_per_class, d_arm_per_class, us_per_class):
    'return true if we need to split the context a, false otherwise'

    '''print('rev_per_class')
    print(rev_per_class)
    print('d_arm_per_class')
    print(d_arm_per_class)
    print('us_per_class')
    print(us_per_class)'''

    day = len(rev_per_class)

    # Here we find the best arm for b and we compute some parameters
    reward_per_arm_b = [0] * n_arms
    n_pulled_arm_b = [0] * n_arms
    for i in range(day):
        n_pulled_arm_b[d_arm_per_class[i][0]] += 1
        reward_per_arm_b[d_arm_per_class[i][0]] += rev_per_class[i][0]

    '''print('n_pulled_arm_b')
    print(n_pulled_arm_b)
    print('reward_per_arm_b')
    print(reward_per_arm_b)'''

    mean_per_arm_b = [a / b if b!=0 else b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]  # element wise division python
    mean_best_arm_b = max(mean_per_arm_b)
    best_arm_b = mean_per_arm_b.index(mean_best_arm_b)

    '''print('mean_per_arm_b')
    print(mean_per_arm_b)
    print('mean_best_arm_b')
    print(mean_best_arm_b)
    print('best_arm_b')
    print(best_arm_b)'''

    # Here we find the best arm for c and we compute some parameters
    reward_per_arm_c = [0] * n_arms
    n_pulled_arm_c = [0] * n_arms
    for i in range(day):
        n_pulled_arm_c[d_arm_per_class[i][1]] += 1
        reward_per_arm_c[d_arm_per_class[i][1]] += rev_per_class[i][1] + rev_per_class[i][2]

    mean_per_arm_c = [a / b if b!=0 else b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]  # element wise division python
    mean_best_arm_c = max(mean_per_arm_c)
    best_arm_c = mean_per_arm_c.index(mean_best_arm_c)

    # here we find the best arm in total
    reward_per_arm_tot = [0] * n_arms
    n_pulled_arm_tot = [0] * n_arms
    for i in range(day):
        n_pulled_arm_tot[d_arm_per_class[i][0]] += 1
        reward_per_arm_tot[d_arm_per_class[i][0]] += sum(rev_per_class[i])

    mean_per_arm_tot = [a / b if b!=0 else b for a, b in zip(reward_per_arm_tot, n_pulled_arm_tot)]  # element wise division python
    mean_best_arm_tot = max(mean_per_arm_tot)
    best_arm_tot = mean_per_arm_tot.index(mean_best_arm_tot)

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

    print('varianza:')
    print(rewards_best_arm_b)
    print(var_b)

    # find lower bound mub, muc mu0
    '''print('tstudent')
    print(tstudent.ppf(confidence, (n_pulled_arm_b[best_arm_b] - 1), loc=0, scale=1) * np.sqrt(
        var_b / n_pulled_arm_b[best_arm_b]))'''
    '''print('var_b')
    print(var_b)'''
    mub = mean_best_arm_b - tstudent.ppf(confidence, (n_pulled_arm_b[best_arm_b] - 1), loc=0, scale=1) * np.sqrt(
        var_b / n_pulled_arm_b[best_arm_b])
    muc = mean_best_arm_c - tstudent.ppf(confidence, (n_pulled_arm_c[best_arm_c] - 1), loc=0, scale=1) * np.sqrt(
        var_c / n_pulled_arm_c[best_arm_c])
    muzero = mean_best_arm_tot - tstudent.ppf(confidence, (n_pulled_arm_tot[best_arm_tot] - 1), loc=0, scale=1) * np.sqrt(
        var_tot / n_pulled_arm_tot[best_arm_tot])

    print(best_arm_b)
    print(best_arm_c)
    print(best_arm_tot)

    if best_arm_b != best_arm_c:
        return splitting(pb, mub, pc, muc, muzero)
    else:
        return False

def context_c_split(rev_per_class, d_arm_per_class, us_per_class):
    'return true if we need to split the context c, false otherwise'

    day = len(rev_per_class) - 30

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
        reward_per_arm_tot[d_arm_per_class[i][1]] += sum(rev_per_class[i][1]) + sum(rev_per_class[i][2])

    mean_per_arm_tot = [a / b if b!=0 else b for a, b in zip(reward_per_arm_tot, n_pulled_arm_tot)]  # element wise division python
    mean_best_arm_tot = max(mean_per_arm_tot)
    best_arm_tot = mean_per_arm_tot.index(mean_best_arm_tot)

    # find probability of context b and c, then compute the lower bounds
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

    # compute variance revenue for best arm b, c, and tot
    rewards_best_arm_d = np.empty(0)
    rewards_best_arm_e = np.empty(0)
    rewards_best_arm_tot = np.empty(0)

    for j in range(day):

        if d_arm_per_class[j][1] == best_arm_d:
            rewards_best_arm_d = np.append(rewards_best_arm_d, rev_per_class[j][1])

        if d_arm_per_class[j][2] == best_arm_e:
            rewards_best_arm_e = np.append(rewards_best_arm_e, (rev_per_class[j][2]))

        if d_arm_per_class[j][1] == best_arm_tot:
            rewards_best_arm_tot = np.append(rewards_best_arm_tot, (sum(rev_per_class[j][1]) + sum(rev_per_class[j][2]) ) )

    var_d = np.var(rewards_best_arm_d, ddof = 1)
    var_e = np.var(rewards_best_arm_e, ddof = 1)
    var_tot = np.var(rewards_best_arm_tot, ddof = 1)

    # find lower bound mub, muc mu0
    mud = mean_best_arm_d - tstudent.ppf(confidence, (n_pulled_arm_d[best_arm_d] - 1), loc=0, scale=1) * np.sqrt(
        var_d / n_pulled_arm_d[best_arm_d])
    mue = mean_best_arm_e - tstudent.ppf(confidence, (n_pulled_arm_e[best_arm_e] - 1), loc=0, scale=1) * np.sqrt(
        var_e / n_pulled_arm_e[best_arm_e])
    muzero = mean_best_arm_tot - tstudent.ppf(confidence, (n_pulled_arm_tot[best_arm_tot] - 1), loc=0, scale=1) * np.sqrt(
        var_tot / n_pulled_arm_tot[best_arm_tot])

    if best_arm_d!=best_arm_e:
        return splitting(pd, mud, pe, mue, muzero)
    else:
        return False

def split(split_context, rev_per_class, d_arm_per_class, us_per_class):
    'decide if a split is needed and return the best context'

    if split_context == 1:
        if context_a_split(rev_per_class, d_arm_per_class, us_per_class):
            return 2
        else:
            return 1

    elif split_context == 2:
        if context_c_split(rev_per_class, d_arm_per_class, us_per_class):
            return 3
        else:
            return 2

    else:
        return 3

## TODO: parallelization like P3

for iter in range(20):

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
        print(new_users)

        [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
        cost = [cost1, cost2, cost3]

        # Get the total cost
        total_cost = 0
        for i in range(len(new_users)):
            total_cost += new_users[i] * cost[i]

        # In the first days we won't split for sure
        if t < 100:
            daily_arm = tsgauss_learner.pull_arm()
            daily_price = [prices[daily_arm]] * 3
            daily_arm_per_class.append([daily_arm] * 3)

        else:
            if t % 7 == 0:
                context_old = context
                context = split(context_old, revenue_per_class[0:-30], daily_arm_per_class[0:-30], users_per_class[0:-30])

                print('context:')
                print(context_old)
                print(context)

                if context > context_old:
                    ## TODO: all
                    if context == 2:
                        print('A -- > B + C at day: ' + str (t) + '--------------------------------------------------------------------------------------------------------------------------------' + str(iter))

                        #compute tau_b and mu_b then create the new tsgauss_learner_b
                        reward_per_arm_b = [0] * n_arms
                        n_pulled_arm_b = [0] * n_arms
                        for i in range(t-30):
                              n_pulled_arm_b[daily_arm_per_class[i][0]] += 1
                              reward_per_arm_b[daily_arm_per_class[i][0]] += revenue_per_class[i][0]

                        mean_per_arm_b = [a / b if b!=0 else b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]  # element wise division python

                        mu_b = n_pulled_arm_b * tau**2 * mean_per_arm_b / (n_pulled_arm_b * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_b * tau**2 + sigma0**2)

                        tau_b = (sigma0 * tau)**2 / (n_pulled_arm_b * tau**2 + sigma0**2)
                        k = 30 #magic parameter

                        tsgauss_learner_b = TSLearnerGauss(n_arms, [revenue_per_class[i][0] for i in range(len(revenue_per_class)-k)], mu_b, tau_b, sigma0, [daily_arm_per_class[0][i] for i in range(t-k,t)], [revenue_per_class[0][i] for i in range(t-k,t)], reward_per_arm_b, t)

                        #compute tau_c and mu_c then create the new tsgauss_learner_c
                        reward_per_arm_c = [0] * n_arms
                        n_pulled_arm_c = [0] * n_arms
                        for i in range(t-30):
                            n_pulled_arm_c[daily_arm_per_class[i][1]] += 1
                            reward_per_arm_c[daily_arm_per_class[i][1]] += revenue_per_class[i][1] + revenue_per_class[i][2]

                        mean_per_arm_c = [a / b if b!=0 else b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]  # element wise division python

                        mu_c = n_pulled_arm_c * tau**2 * mean_per_arm_c / (n_pulled_arm_c * tau**2 + sigma0**2) + sigma0**2 * mu0 / (n_pulled_arm_c * tau**2 + sigma0**2)
                        tau_c = (sigma0 * tau)**2 / (n_pulled_arm_c * tau**2 + sigma0**2)

                        tsgauss_learner_c = TSLearnerGauss(n_arms, [revenue_per_class[i][1] + revenue_per_class[i][2] for i in range(len(revenue_per_class)-k)], mu_b, tau_b, sigma0, [daily_arm_per_class[0][i] for i in range(t-k,t)], [revenue_per_class[1][i] + revenue_per_class[2][i] for i in range(t-k,t)], reward_per_arm_c, t)

                    ## TODO: if context == 3

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

        daily_bought_items_perclass = [0, 0, 0]
        # Calculate the number of real bought items
        for i in range(len(new_users)):
            for c in range(new_users[i]):
                daily_bought_items_perclass[i] += env.buy(daily_price[i], i + 1)

        '''print(new_users)'''
        '''print(daily_price)'''
        '''print(daily_bought_items_perclass)'''

        margin = [env.get_margin(int(price)) for price in daily_price]

        '''print(margin)'''

        revenue_per_class_today = []
        for i in range(len(margin)):
            revenue_per_class_today.append(margin[i] * daily_bought_items_perclass[i] - cost[i] * new_users[i])

        '''print(revenue_per_class_today)'''

        next_30_days = [env.get_next_30_days(daily_bought_items_perclass[0], daily_price[0], 1), env.get_next_30_days(daily_bought_items_perclass[1], daily_price[1], 2), env.get_next_30_days(daily_bought_items_perclass[2], daily_price[2], 3)]
        '''print('next_30_days:')
        print(next_30_days)'''
        sum_next_30_days = [sum(next_30_days[i]) for i in range(len(next_30_days))]
        '''print('sum_next_30_days')
        print(sum_next_30_days)'''

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

        '''print(revenue_per_class[-1])'''
