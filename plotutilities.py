from matplotlib import pyplot
import os

import numpy as np
from scipy.stats import t as tstudent

def plot(list_of_things_to_plot, legend, title, plots_folder, color=0):
    '''plot the list of things to plot in a plot with the given legend, title and colors,
        then save the plot in the given folder as title.png'''

    pyplot.figure()

    palette_list = [['#006A4E', '#FFF154','#FF00FF', '#87CEEB'],
                    ['r' ,'gold' , 'k'],
                    ['r', 'deepskyblue', 'k']]
    if color == 0:
        for things_to_plot in list_of_things_to_plot:
            pyplot.plot(things_to_plot)
    else:
        col = palette_list[color-1]
        for i in range(len(list_of_things_to_plot)):
            pyplot.plot(list_of_things_to_plot[i],col[i])

    pyplot.xlim([0, 365])
    pyplot.legend(legend)
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, title + '.png'))
    pyplot.close()

def multi_plot(list_of_mean, name, plots_folder, best = False):
    'plot 3 list of mean: one for every class'

    pyplot.figure()
    for i in range(len(list_of_mean)):
        pyplot.plot(list_of_mean[i])
    if type(best)!=type(False):
        for i in range(len(list_of_mean)):
            pyplot.plot([best[i] for a in range(len(list_of_mean[i]))],'k')
    pyplot.xlim([0, 364])
    pyplot.legend(['Mean ' + str(name) + ' class 1', 'Mean ' + str(name) + ' of class 2', 'Mean ' + str(name) + ' of class 3'])
    pyplot.title('Mean ' + str(name) + ' per class')
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, 'Mean ' + str(name) + ' per class.png'))
    pyplot.close()

def plot_learned_curve(mu, tau, real, n_pulled_arms, plots_folder, t=0):
    'plot the curve learned by ts_gauss'

    pyplot.figure()

    pyplot.plot(mu,'o-')
    pyplot.plot(real,'o-')

    confidence = 0.99
    x = [i for i in range(len(mu))]
    n = [1 if n_pulled_arms[i]==0 or n_pulled_arms[i]==1 else n_pulled_arms[i] - 1 for i in range(len(n_pulled_arms))]

    sup = [mu[i] + tstudent.ppf(confidence, n[i], loc=0, scale=1) * tau[i] / np.sqrt(n[i]) for i in range(len(mu))]
    inf = [mu[i] - tstudent.ppf(confidence, n[i], loc=0, scale=1) * tau[i] / np.sqrt(n[i]) for i in range(len(mu))]

    pyplot.fill_between(x, inf, sup, alpha = 0.5)
    pyplot.ylim(-1000,3000)
    pyplot.ylabel('Rewards')
    pyplot.xlabel('Arms')
    pyplot.legend(['Learned', 'Real'])
    pyplot.title('Learned curve')

    if t==0:
        pyplot.savefig(os.path.join(plots_folder, 'Learned curve.png'))
    else:
        pyplot.savefig(os.path.join(plots_folder, str(t)+'Learned curve.png'))

    pyplot.close()
