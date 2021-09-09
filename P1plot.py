from matplotlib import pyplot
from matplotlib import cm
import os

from environment import Environment

from P1utilities import *

env = Environment()

bids = env.bids
prices = env.prices

cwd = os.getcwd()
plots_folder = os.path.join(cwd, "plotsp1")

class_list = ['Class 1', 'Class 2', 'Class 3']

title = 'Mean daily clicks of new customers disaggregated'
pyplot.figure()
for i in range(1,4):
    pyplot.plot(bids, [env.get_mean_new_users_daily(bid,i) for bid in bids],'o-')
pyplot.legend(class_list)
pyplot.xlabel('Bids')
pyplot.ylabel('New customers')
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))

title = 'Mean daily clicks of new customers aggregated'
pyplot.figure()
pyplot.plot(bids, [env.get_mean_new_users_daily(bid,1)+env.get_mean_new_users_daily(bid,2)+env.get_mean_new_users_daily(bid,3)  for bid in bids],'o-k')
pyplot.xlabel('Bids')
pyplot.ylabel('New customers')
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))

title = 'Conversion rate'
pyplot.figure()
for i in range(1,4):
    pyplot.plot(prices,[env.get_conversion_rate(price, i) for price in prices], 'o-')
pyplot.legend(class_list)
pyplot.xlabel('Prices')
pyplot.ylabel('Conversion rate')
pyplot.xticks([prices[i] for i in range(1,len(prices),2)])
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))

for classe in range(1,4):
    title = 'Revenue function class ' + str(classe)
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(bids, prices)
    Z = np.array([[get_bid_and_price_revenue(bid, price, classe) for bid in bids] for price in prices])
    if classe == 1:
        ax.plot_surface(X, Y, Z, cmap = cm.Blues, linewidth=0, antialiased=False)
    elif classe == 2:
        ax.plot_surface(X, Y, Z, cmap = cm.Oranges, linewidth=0, antialiased=False)
    else:
        ax.plot_surface(X, Y, Z, cmap = cm.Greens, linewidth=0, antialiased=False)
    ax.view_init(elev=30., azim=40)
    pyplot.yticks([prices[i] for i in range(1,len(prices),2)])
    pyplot.xlabel('Bids')
    pyplot.ylabel('Prices')
    pyplot.title(title)
    pyplot.savefig(os.path.join(plots_folder, title + '.png'))
