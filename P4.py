from environment import Environment
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from tsgaussp4 import TSLearnerGauss
from scipy.stats import t

confidence = 0.9


env = Environment()

T = 365

prices = np.linspace(1, 10, num=10)
bids = [0.7]

tsgauss_learner = TSLearnerGauss(len(prices))

vector_daily_price = []
vector_daily_revenue = []

for t in range(T):

    print("Iteration day: " + str(t))
    # Get new users in the day t and their costs
    [new_user_1, new_user_2, new_user_3] = env.get_all_new_users_daily(bids[0])
    new_users = [new_user_1, new_user_2, new_user_3]
    [cost1, cost2, cost3] = env.get_all_cost_per_click(bids[0])
    cost = [cost1, cost2, cost3]

    # Get the total cost
    total_cost = 0
    for i in range(len(new_users)):
        total_cost += new_users[i] * cost[i]

    #per i primi 50 giorni non splittiamo di sicuro lol
    if t<50:
        daily_arm = tsgauss_learner.pull_arm()
        daily_price = prices[daily_arm]
        vector_daily_price.append(daily_price)

    ### TODO: capiamo se conviene splittare o no (panic)
    else:




    daily_bought_items_perclass = [0, 0, 0]
    daily_bought_items_perfeature = [[0, 0] [0, 0]]
    # Calculate the number of real bought items
    for i in range(len(new_users)):
        for c in range(new_users[i]):
            buy = env.buy(daily_price, i+1)
            feature = env.feature(i + 1)
            daily_bought_items_perclass[i] += buy
            daily_bought_items_perfeature[ feature[0] ][ feature[1] ] += buy

    daily_bought_items = sum(daily_bought_items_perclass)

    daily_revenue = daily_bought_items * env.get_margin( daily_price ) - total_cost



def split(context, revenue_perclass, daily_arm_perclass, users_per_class):


    if ( len(context)==1 and !context_a_split(revenue_perclass, daily_arm_perclass,users_per_class) ):

        return(['a'])

    elif ( len(context)==2 and !context_a_split(revenue_perclass, daily_arm_perclass,users_per_class) ):

        return(['b' 'c'] )

    else:

        return(['b' 'd' 'e'])


def splitting(p1, mu1, p2, mu2, mu0):
    #return true if we need to split the context, false otherwise
    return ( p1 * mu1 + p2 * mu2 > mu0)

def context_a_split(revenue_perclass, daily_arm_perclass, users_per_class):
    #return true if we need to split the context a, false otherwise'''

    day = len(revenue_perclass)
    n_arms = max(daily_arm_perclass)

    #trovo l'arma migliore per b e calcolo un po' di parametri
    reward_per_arm_b = [0] * n_arms
    n_pulled_arm_b = [0] * n_arms
        for i in range(day):
            n_pulled_arm_b[daily_arm_perclass[i][0]] += 1
            reward_per_arm_b[daily_arm_perclass[i][0]] += revenue_perclass[i][0]

    mean_per_arm_b = [a/b for a,b in zip(reward_per_arm_b, n_pulled_arm_b)] #element wise division python
    mean_best_arm_b = max(mean_per_arm_b)
    best_arm_b = mean_per_arm_b.index(mean_best_arm_b)

    #trovo l'arma migliore per c e calcolo un po' di parametri
    reward_per_arm_c = [0] * n_arms
    n_pulled_arm_c = [0] * n_arms
        for i in range(day):
            n_pulled_arm_c[daily_arm_perclass[i][1]] += 1
            reward_per_arm_c[daily_arm_perclass[i][1]] += revenue_perclass[i][1] + revenue_perclass[i][2]

    mean_per_arm_c = [a/b for a,b in zip(reward_per_arm_c, n_pulled_arm_c)] #element wise division python
    mean_best_arm_c = max(mean_per_arm_c)
    best_arm_c = mean_per_arm_c.index(mean_best_arm_c)

    #trovo best arm totale
    reward_per_arm_tot = [0] * n_arms
    n_pulled_arm_tot = [0] * n_arms
        for i in range(day):
            n_pulled_arm_tot[daily_arm_perclass[i][0]] += 1
            reward_per_arm_tot[daily_arm_perclass[i][0]] += sum(revenue_perclass[i])

    mean_per_arm_tot = [a/b for a,b in zip(reward_per_arm_tot, n_pulled_arm_tot)] #element wise division python
    mean_best_arm_tot = max(mean_per_arm_tot)
    best_arm_tot = mean_per_arm_tot.index(mean_best_arm_tot)

    #trovo probabilità context b e c
    ## TODO: trovare i lower bound sulle probabilità
    b_users = 0
    c_users = 0

    for i in range(day):
        b_users += users_per_class[i][0]
        c_users += users_per_class[i][1] + users_per_class[i][2]

    pb = b_users / ( b_users + c_users)
    pc = c_users / ( b_users + c_users)

    #calcolo varianza revenue per best arm b, c e tot

    rewards_best_arm_b = np.array([])
    rewards_best_arm_c = np.array([])
    rewards_best_arm_tot = np.array([])

    for i in range(day):

        if (daily_arm_perclass[i][0] == best_arm_b):
            np.append(rewards_best_arm_b , revenue_perclass[i][0])

        if (daily_arm_perclass[i][1] == best_arm_c):
            np.append(rewards_best_arm_c , (revenue_perclass[i][1] + revenue_perclass[i][2]))

        if (daily_arm_perclass[i][0] == best_arm_tot):
            np.append(rewards_best_arm_tot , (sum(revenue_perclass[i]))

    var_b = np.var(rewards_best_arm_b, ddof=1)
    var_c = np.var(rewards_best_arm_c, ddof=1)
    var_tot = np.var(rewards_best_arm_tot, ddof=1)

    #trovo i lower bound mub, muc mu0
    mub = mean_best_arm_b - t.ppf(confidence, (n_pulled_arm_b[best_arm_b] - 1), loc=0, scale=1) * np.sqrt( var_b / n_pulled_arm_b[best_arm_b] )
    muc = mean_best_arm_c - t.ppf(confidence, (n_pulled_arm_c[best_arm_c] - 1), loc=0, scale=1) * np.sqrt( var_c / n_pulled_arm_c[best_arm_c] )
    mu0 = mean_best_arm_tot - t.ppf(confidence, (n_pulled_arm_tot[best_arm_tot] - 1), loc=0, scale=1) * np.sqrt( var_tot / n_pulled_arm_tot[best_arm_tot] )

    return (splitting(pb, mub, pc, muc, mu0))

def context_c_split(revenue_perclass, daily_arm_perclass, users_per_class):

    ## TODO: capire se connviene raggruppare tutti gli splitted context in una sola funzione
