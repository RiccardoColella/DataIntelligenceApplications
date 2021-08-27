import numpy as np
from scipy.stats import t

from environment import Environment
from tsgaussp4 import TSLearnerGauss

# A --> B + C
# C --> D + E

confidence = 0.9

T = 365

prices = np.linspace(1, 10, num=10)
bids = [0.7]

users_per_class = []
revenue_per_class = []
daily_arm_per_class = []

context = 1


def split(split_context, rev_per_class, d_arm_per_class, us_per_class):
    ## TODO: mi sa che sti if else sono da sistemare lol
    if split_context == 1 and context_a_split(rev_per_class, d_arm_per_class, us_per_class) is False:
        return 1

    elif split_context == 2 and context_a_split(rev_per_class, d_arm_per_class, us_per_class) is False:
        return 2

    else:
        return 3


def splitting(p1, mu1, p2, mu2, mu0):
    # return true if we need to split the context, false otherwise
    return p1 * mu1 + p2 * mu2 > mu0


def context_a_split(rev_per_class, d_arm_per_class, us_per_class):
    # return true if we need to split the context a, false otherwise'''

    day = len(rev_per_class)
    n_arms = max(d_arm_per_class)

    # Here we find the best arm for b and we compute some parameters
    reward_per_arm_b = [0] * n_arms
    n_pulled_arm_b = [0] * n_arms
    for i in range(day):
        n_pulled_arm_b[d_arm_per_class[i][0]] += 1
        reward_per_arm_b[d_arm_per_class[i][0]] += rev_per_class[i][0]

    mean_per_arm_b = [a / b for a, b in zip(reward_per_arm_b, n_pulled_arm_b)]  # element wise division python
    mean_best_arm_b = max(mean_per_arm_b)
    best_arm_b = mean_per_arm_b.index(mean_best_arm_b)

    # Here we find the best arm for c and we compute some parameters
    reward_per_arm_c = [0] * n_arms
    n_pulled_arm_c = [0] * n_arms
    for i in range(day):
        n_pulled_arm_c[d_arm_per_class[i][1]] += 1
        reward_per_arm_c[d_arm_per_class[i][1]] += rev_per_class[i][1] + rev_per_class[i][2]

    mean_per_arm_c = [a / b for a, b in zip(reward_per_arm_c, n_pulled_arm_c)]  # element wise division python
    mean_best_arm_c = max(mean_per_arm_c)
    best_arm_c = mean_per_arm_c.index(mean_best_arm_c)

    # here we find the best arm in total
    reward_per_arm_tot = [0] * n_arms
    n_pulled_arm_tot = [0] * n_arms
    for i in range(day):
        n_pulled_arm_tot[d_arm_per_class[i][0]] += 1
        reward_per_arm_tot[d_arm_per_class[i][0]] += sum(rev_per_class[i])

    mean_per_arm_tot = [a / b for a, b in zip(reward_per_arm_tot, n_pulled_arm_tot)]  # element wise division python
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
    rewards_best_arm_b = np.array()
    rewards_best_arm_c = np.array()
    rewards_best_arm_tot = np.array()

    for j in range(day):

        if d_arm_per_class[j][0] == best_arm_b:
            np.append(rewards_best_arm_b, rev_per_class[j][0])

        if d_arm_per_class[j][1] == best_arm_c:
            np.append(rewards_best_arm_c, (rev_per_class[j][1] + rev_per_class[j][2]))

        if d_arm_per_class[j][0] == best_arm_tot:
            np.append(rewards_best_arm_tot, (sum(rev_per_class[j])))

    var_b = np.var(rewards_best_arm_b, ddof=1)
    var_c = np.var(rewards_best_arm_c, ddof=1)
    var_tot = np.var(rewards_best_arm_tot, ddof=1)

    # find lower bound mub, muc mu0
    mub = mean_best_arm_b - t.ppf(confidence, (n_pulled_arm_b[best_arm_b] - 1), loc=0, scale=1) * np.sqrt(
        var_b / n_pulled_arm_b[best_arm_b])
    muc = mean_best_arm_c - t.ppf(confidence, (n_pulled_arm_c[best_arm_c] - 1), loc=0, scale=1) * np.sqrt(
        var_c / n_pulled_arm_c[best_arm_c])
    mu0 = mean_best_arm_tot - t.ppf(confidence, (n_pulled_arm_tot[best_arm_tot] - 1), loc=0, scale=1) * np.sqrt(
        var_tot / n_pulled_arm_tot[best_arm_tot])

    return splitting(pb, mub, pc, muc, mu0)


def context_c_split(rev_per_class, d_arm_per_class, us_per_class):
    # TODO: Should we group all the splitted context in one single function?
    return False


if __name__ == '__main__':
    env = Environment()
    tsgauss_learner = TSLearnerGauss(len(prices))

    for t in range(T):

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
        if t < 50:
            daily_arm = tsgauss_learner.pull_arm()
            daily_price = [prices[daily_arm]] * 3
            daily_arm_per_class.append([daily_arm] * 3)

        else:
            if t % 7 == 0:
                context_old = context
                # TODO: manage delay
                context = split(context_old, revenue_per_class, daily_arm_per_class, users_per_class)

                if context > context_old:
                    print('todo')
                    # TODO: change ts gauss in use

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
                buy = env.buy(daily_price[i], i + 1)
                daily_bought_items_perclass[i] += buy

        # TODO: manage delay

        margin = [env.get_margin(i) for i in daily_price]

        revenue_per_class_today = []
        for i in range(margin):
            revenue_per_class_today.append(margin * daily_bought_items_perclass[i] - cost[i] * new_users[i])

        revenue_per_class.append(revenue_per_class_today)
