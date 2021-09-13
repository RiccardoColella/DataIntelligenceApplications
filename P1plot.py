from matplotlib import pyplot
from matplotlib import cm
from matplotlib.lines import Line2D
import os
from scipy.stats import beta
from scipy.stats import norm

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
pyplot.close()

title = 'Mean daily clicks of new customers aggregated'
pyplot.figure()
pyplot.plot(bids, [env.get_mean_new_users_daily(bid,1)+env.get_mean_new_users_daily(bid,2)+env.get_mean_new_users_daily(bid,3)  for bid in bids],'o-k')
pyplot.xlabel('Bids')
pyplot.ylabel('New customers')
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

title = 'Mean cost per click'
pyplot.figure()
for i in range(1,4):
    pyplot.plot(bids, [env.get_mean_cost_per_click(bid,i) for bid in bids], 'o-')
pyplot.xlabel('Bids')
pyplot.ylabel('Cost per click')
pyplot.legend(class_list)
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

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
pyplot.close()

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
    pyplot.close()

title = 'Revenue function'
fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(bids, prices)
Z = np.array([[sum([get_bid_and_price_revenue(bid, price, classe) for classe in range(1,4)]) for bid in bids] for price in prices])
ax.plot_surface(X, Y, Z, cmap = cm.Greys, linewidth=0, antialiased=False)
ax.view_init(elev=30., azim=40)
pyplot.yticks([prices[i] for i in range(1,len(prices),2)])
pyplot.xlabel('Bids')
pyplot.ylabel('Prices')
pyplot.title(title)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

title = 'Percentage of bid probability density function'
pyplot.figure()
x = np.linspace(0,1,100)
for clas in range(1,4):
    alpha = env.customer_classes[clas - 1].a_cost_per_click
    bheta = env.customer_classes[clas - 1].b_cost_per_click
    min = env.customer_classes[clas - 1].min_cost_per_click
    y = [beta.pdf(i,alpha, bheta) for i in x]
    scaled_x = [i*(1-min) + min for i in x]
    pyplot.plot(scaled_x, y)

pyplot.xlabel('Percentage of bid')
pyplot.ylabel('Probability density function')
pyplot.title(title)
pyplot.legend(class_list)
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

title = 'Number of comebacks probability density function'
pyplot.figure()

x = np.linspace(4,11,1000)
fig, (ax1, ax2, ax3) = pyplot.subplots(3)
fig.suptitle(title)
for clas in range(1,4):
    mean = env.customer_classes[clas - 1].mean_n_times_comeback
    dev = env.customer_classes[clas - 1].dev_n_times_comeback
    y = [norm.pdf(i,loc=mean, scale=dev) for i in x]
    if clas == 1:
        ax1.plot(x,y,'b')
        ax1.legend([class_list[clas-1]])
    if clas == 2:
        ax2.plot(x,y,'orange')
        ax2.legend([class_list[clas-1]])
    if clas == 3:
        ax3.plot(x,y,'g')
        ax3.legend([class_list[clas-1]])

ax2.set(ylabel='Probability density function')
ax3.set(xlabel='Number of comebacks')

pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

title = 'Alghoritm comparison'

N=21
X, Y = np.meshgrid([i for i in range(1,N)], [i for i in range(1,N)])
Z_brute = np.array([[x*y for x in range(1,N)] for y in range(1,N)])
Z_our = np.array([[x+y for x in range(1,N)] for y in range(1,N)])

fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})

ax.plot_surface(X, Y, Z_brute, cmap = cm.autumn, linewidth=0, antialiased=False)
ax.plot_surface(X, Y, Z_our, cmap = cm.winter, linewidth=0, antialiased=False)

pyplot.xlabel('Number of Bids')
pyplot.xticks([1,5,10,15,20])
pyplot.ylabel('Number of Prices')
pyplot.yticks([1,5,10,15,20])
pyplot.title(title)
ax.view_init(elev=20., azim=105)
ax.legend([Line2D([0], [0], color=cm.autumn(0.3), lw=4), Line2D([0], [0], color=cm.winter(1), lw=4)],
    ['Brute force alghoritm (P*X)', 'Our alghoritm (P+X)'])
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()

title = 'Margin'
pyplot.figure()
pyplot.title(title)
margin = [env.get_margin(a) for a in prices]
pyplot.plot(prices, margin, 'ok-')
pyplot.xticks(prices)
pyplot.yticks(margin)
pyplot.xlabel('Price')
pyplot.ylabel('Margin')
pyplot.savefig(os.path.join(plots_folder, title + '.png'))
pyplot.close()
