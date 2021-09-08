from matplotlib import pyplot
import os

from environment import Environment

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
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
